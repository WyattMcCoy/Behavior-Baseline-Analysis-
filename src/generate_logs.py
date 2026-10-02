"""
generate_logs.py  (version 2: realistic test)
Creates fake user activity logs for the Digital Behavioral Baseline project.

Version 1 was too easy: normal users never touched sensitive pages, so the model
scored a perfect 1.0. Version 2 adds the messiness of a real workplace:
  - Roles: finance, managers, and IT use sensitive pages as part of their jobs
  - Occasional users who log in rarely, at random times
  - Hybrid workers with two normal locations
  - Users who switch to remote partway through the test period
  - Users returning from leave (failed logins, then a password change)
  - Quiet attackers who move at human speed and mostly stay on normal pages

Training period: normal behavior only. Test period: normal + about 3% attacks.
The answer key is saved separately in data/labels/.

Run from the project folder:  python src/generate_logs.py
"""

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
SEED = 42
N_USERS = 50
TRAIN_START = datetime(2026, 8, 1)
TRAIN_DAYS = 30
TEST_DAYS = 14
ATTACK_RATE = 0.03             # Travis's call
CONTEXT_SHIFT_CHANCE = 0.5     # chance each attack gets each context clue
QUIET_ATTACK_CHANCE = 0.5      # share of Explorers/Bots that are the quiet version

# Normal-life noise
TRAVEL_CHANCE = 0.015
NEW_DEVICE_CHANCE = 0.02
LEAVE_RETURN_CHANCE = 0.01     # forgot password after time off
QUICK_EXPORT_CHANCE = 0.15     # finance users going straight to export
HYBRID_SHARE = 0.2             # share of regular users who split office/home
N_REMOTE_SWITCHERS = 2
REMOTE_SWITCH_TEST_DAY = 5     # they go remote on day 5 of the test period

# Share of users in each role
ROLES = {"staff": 0.50, "finance": 0.15, "manager": 0.12, "it": 0.10, "occasional": 0.13}
# Sensitive pages that are part of each role's normal job
ROLE_PAGES = {"staff": [], "finance": ["/billing", "/export"],
              "manager": ["/export"], "it": ["/admin", "/admin/users"]}

rng = np.random.default_rng(SEED)          # main dice
noise_rng = np.random.default_rng(SEED + 1)  # dice for normal-life surprises

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LABEL_DIR = PROJECT_ROOT / "data" / "labels"

NORMAL_PAGES = ["/home", "/dashboard", "/inbox", "/calendar", "/reports",
                "/team", "/schedule", "/docs", "/tickets", "/profile"]
SENSITIVE_PAGES = ["/admin", "/admin/users", "/billing", "/export",
                   "/settings/security"]

HOME_LOCATIONS = [("US", "San Francisco"), ("US", "Oakland"), ("US", "San Jose"),
                  ("US", "Sacramento"), ("US", "Daly City")]
OTHER_LOCATIONS = [("DE", "Frankfurt"), ("NL", "Amsterdam"), ("BR", "Sao Paulo"),
                   ("RO", "Bucharest"), ("SG", "Singapore"), ("RU", "Moscow")]

USER_AGENTS = ["Chrome/Windows", "Edge/Windows", "Safari/macOS",
               "Chrome/macOS", "Firefox/Linux"]

ATTACK_TYPES = ["explorer", "bot", "guesser"]


def random_ip(first):
    return f"{first}.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}"


def other_home_location(city):
    options = [loc for loc in HOME_LOCATIONS if loc[1] != city]
    return options[rng.integers(len(options))]


# ------------------------------------------------------------------
# STEP 1: Users, each with a role and a personal normal
# ------------------------------------------------------------------
def make_user_profiles(n_users):
    role_names, role_probs = list(ROLES), list(ROLES.values())
    profiles = []
    for i in range(n_users):
        role = str(rng.choice(role_names, p=role_probs))
        country, city = HOME_LOCATIONS[rng.integers(len(HOME_LOCATIONS))]
        profile = {
            "user_id": f"user_{i:03d}",
            "role": role,
            "country": country, "city": city,
            "ip": f"10.{i}.{rng.integers(0, 256)}.{rng.integers(1, 255)}",
            "device_id": f"dev_{i:03d}",
            "user_agent": USER_AGENTS[rng.integers(len(USER_AGENTS))],
            "start_hour": float(np.clip(rng.normal(9, 1.5), 6, 12)),
            "hour_spread": 1.0,
            "sessions_per_day": rng.uniform(1, 3),
            "actions_per_session": rng.uniform(8, 25),
            "seconds_between_actions": rng.uniform(10, 40),
            "download_rate": rng.uniform(0.02, 0.08),
            "works_weekends": bool(rng.random() < 0.2),
            "second_ctx": None,     # hybrid workers get a second normal
            "remote_from": None,    # remote switchers get a date
        }
        if role == "occasional":
            # Rare logins at random hours, short visits to inbox and reports.
            profile.update(sessions_per_day=rng.uniform(0.02, 0.12),
                           actions_per_session=rng.uniform(3, 7),
                           favorite_pages=["/inbox", "/reports", "/home"],
                           works_weekends=True)
        else:
            extra = ROLE_PAGES[role]
            pool = [p for p in NORMAL_PAGES if p not in extra]
            profile["favorite_pages"] = [str(p) for p in rng.choice(pool, size=4, replace=False)] + extra
            if role == "finance":
                profile["download_rate"] = rng.uniform(0.10, 0.25)
        profiles.append(profile)

    regular = [p for p in profiles if p["role"] != "occasional"]
    order = rng.permutation(len(regular))
    n_hybrid = int(round(HYBRID_SHARE * len(regular)))

    # Hybrid: a second location and device that are ALSO normal for them.
    for idx in order[:n_hybrid]:
        p = regular[idx]
        country, city = other_home_location(p["city"])
        p["second_ctx"] = {"country": country, "city": city, "ip": random_ip(192),
                           "device_id": p["device_id"] + "_home",
                           "user_agent": p["user_agent"]}
        p["hour_spread"] = 2.0

    # Remote switchers: everything changes partway through the test period.
    test_start = TRAIN_START + timedelta(days=TRAIN_DAYS)
    for idx in order[n_hybrid:n_hybrid + N_REMOTE_SWITCHERS]:
        p = regular[idx]
        country, city = other_home_location(p["city"])
        p["remote_from"] = test_start + timedelta(days=REMOTE_SWITCH_TEST_DAY)
        p["remote_ctx"] = {"country": country, "city": city, "ip": random_ip(192),
                           "device_id": p["device_id"] + "_remote",
                           "user_agent": p["user_agent"]}
        p["remote_start_hour"] = float(np.clip(p["start_hour"] + rng.choice([-2.5, 2.5]), 5, 14))
    return profiles


def home_context(profile):
    return {key: profile[key] for key in ["ip", "user_agent", "device_id", "country", "city"]}


def make_event(profile, session_id, ts, action, page, ctx, success=True):
    return {
        "user_id": profile["user_id"],
        "timestamp": ts,
        "action_type": action,
        "page_url": page,
        "ip_address": ctx["ip"],
        "user_agent": ctx["user_agent"],
        "device_id": ctx["device_id"],
        "country": ctx["country"],
        "city": ctx["city"],
        "session_id": session_id,
        "response_time_ms": int(max(30, rng.normal(250, 60))),
        "success_flag": success,
    }


def pick_action(download_rate):
    roll = rng.random()
    if roll < download_rate:
        return "download_file"
    return "view_page" if roll < 0.65 else "click_button"


# ------------------------------------------------------------------
# STEP 2: Normal sessions, including the messy-but-legitimate ones
# ------------------------------------------------------------------
def normal_session(profile, session_id, start_time):
    ctx = home_context(profile)
    scenario = []   # records WHY an unusual normal session looks unusual

    if profile["remote_from"] is not None and start_time >= profile["remote_from"]:
        ctx.update(profile["remote_ctx"])
        scenario.append("remote_switch")
    elif profile["second_ctx"] is not None and noise_rng.random() < 0.4:
        ctx.update(profile["second_ctx"])
        scenario.append("hybrid_home")

    if noise_rng.random() < TRAVEL_CHANCE:
        if noise_rng.random() < 0.7:
            country, city = other_home_location(profile["city"])
        else:
            country, city = OTHER_LOCATIONS[noise_rng.integers(len(OTHER_LOCATIONS))]
        ctx.update(country=country, city=city, ip=random_ip(172))
        scenario.append("travel")
    if noise_rng.random() < NEW_DEVICE_CHANCE:
        other_agents = [ua for ua in USER_AGENTS if ua != profile["user_agent"]]
        ctx.update(device_id=f"dev_new_{session_id}",
                   user_agent=other_agents[noise_rng.integers(len(other_agents))])
        scenario.append("new_device")

    events = []
    t = start_time

    if noise_rng.random() < LEAVE_RETURN_CHANCE:
        # Back from leave: forgot the password. Human pace between tries.
        scenario.append("leave_return")
        for _ in range(noise_rng.integers(2, 5)):
            events.append(make_event(profile, session_id, t, "login", "/login", ctx, success=False))
            t += timedelta(seconds=noise_rng.uniform(8, 40))
        events.append(make_event(profile, session_id, t, "login", "/login", ctx))
        t += timedelta(seconds=noise_rng.uniform(10, 40))
        events.append(make_event(profile, session_id, t, "change_password", "/settings/security", ctx))
    else:
        if rng.random() < 0.03:   # ordinary typo
            events.append(make_event(profile, session_id, t, "login", "/login", ctx, success=False))
            t += timedelta(seconds=rng.uniform(5, 20))
        events.append(make_event(profile, session_id, t, "login", "/login", ctx))

    if profile["role"] == "finance" and noise_rng.random() < QUICK_EXPORT_CHANCE:
        # Travis's example: straight to export, pull the data, done.
        scenario.append("quick_export")
        for _ in range(noise_rng.integers(3, 8)):
            t += timedelta(seconds=noise_rng.uniform(2, 6))
            events.append(make_event(profile, session_id, t, "download_file", "/export", ctx))
    else:
        n_actions = max(2, rng.poisson(profile["actions_per_session"]))
        for _ in range(n_actions):
            t += timedelta(seconds=rng.exponential(profile["seconds_between_actions"]))
            if rng.random() < 0.85:
                page = str(rng.choice(profile["favorite_pages"]))
            else:
                page = str(rng.choice(NORMAL_PAGES))
            events.append(make_event(profile, session_id, t,
                                     pick_action(profile["download_rate"]), page, ctx))

    if rng.random() < 0.005:
        t += timedelta(seconds=rng.uniform(10, 60))
        events.append(make_event(profile, session_id, t, "change_password", "/settings/security", ctx))

    t += timedelta(seconds=rng.uniform(5, 30))
    events.append(make_event(profile, session_id, t, "logout", "/logout", ctx))
    return events, "+".join(scenario) if scenario else "routine"


# ------------------------------------------------------------------
# STEP 3: Attackers, loud and quiet
# ------------------------------------------------------------------
def attack_session(profile, session_id, start_time, attack_type):
    shifts = {name: bool(rng.random() < CONTEXT_SHIFT_CHANCE)
              for name in ["odd_hour", "new_location", "new_device"]}
    quiet = attack_type in ("explorer", "bot") and rng.random() < QUIET_ATTACK_CHANCE
    variant = "quiet" if quiet else "loud"

    ctx = home_context(profile)
    if shifts["odd_hour"]:
        start_time = start_time.replace(hour=int(rng.integers(0, 5)),
                                        minute=int(rng.integers(0, 60)))
    if shifts["new_location"]:
        country, city = OTHER_LOCATIONS[rng.integers(len(OTHER_LOCATIONS))]
        ctx.update(country=country, city=city, ip=random_ip(185))
    if shifts["new_device"]:
        other_agents = [ua for ua in USER_AGENTS if ua != profile["user_agent"]]
        ctx.update(device_id=f"dev_new_{session_id}",
                   user_agent=other_agents[rng.integers(len(other_agents))])

    events = []
    t = start_time

    if attack_type == "guesser":
        for _ in range(rng.integers(3, 9)):
            events.append(make_event(profile, session_id, t, "login", "/login", ctx, success=False))
            t += timedelta(seconds=rng.uniform(1, 4))
    events.append(make_event(profile, session_id, t, "login", "/login", ctx))

    if attack_type == "explorer" and not quiet:
        # Loud: slow, wandering, half the visits on sensitive pages.
        for _ in range(int(rng.integers(15, 40))):
            t += timedelta(seconds=rng.exponential(profile["seconds_between_actions"] * 3))
            pool = SENSITIVE_PAGES if rng.random() < 0.5 else NORMAL_PAGES
            action = "download_file" if rng.random() < 0.1 else "view_page"
            events.append(make_event(profile, session_id, t, action, str(rng.choice(pool)), ctx))

    elif attack_type == "explorer" and quiet:
        # Quiet: the victim's own pages at the victim's pace, a few sensitive stops.
        for _ in range(max(5, rng.poisson(profile["actions_per_session"]))):
            t += timedelta(seconds=rng.exponential(profile["seconds_between_actions"] * rng.uniform(1.0, 1.5)))
            if rng.random() < 0.85:
                page = str(rng.choice(profile["favorite_pages"]))
            else:
                page = str(rng.choice(SENSITIVE_PAGES))
            action = "download_file" if rng.random() < 0.05 else "view_page"
            events.append(make_event(profile, session_id, t, action, page, ctx))

    elif attack_type == "bot" and not quiet:
        # Loud: very fast mass download.
        for _ in range(int(rng.integers(40, 120))):
            t += timedelta(seconds=rng.uniform(0.3, 2.0))
            page = str(rng.choice(["/export", "/reports", "/docs"]))
            action = "download_file" if rng.random() < 0.8 else "view_page"
            events.append(make_event(profile, session_id, t, action, page, ctx))

    elif attack_type == "bot" and quiet:
        # Low and slow: downloads at a human pace.
        for _ in range(int(rng.integers(15, 40))):
            t += timedelta(seconds=rng.uniform(8, 30))
            page = str(rng.choice(["/export", "/reports", "/docs"]))
            action = "download_file" if rng.random() < 0.6 else "view_page"
            events.append(make_event(profile, session_id, t, action, page, ctx))

    else:  # guesser, after getting in
        for _ in range(int(rng.integers(3, 10))):
            t += timedelta(seconds=rng.uniform(3, 20))
            events.append(make_event(profile, session_id, t, "view_page",
                                     str(rng.choice(NORMAL_PAGES)), ctx))
        t += timedelta(seconds=rng.uniform(5, 30))
        events.append(make_event(profile, session_id, t, "change_password",
                                 "/settings/security", ctx))

    t += timedelta(seconds=rng.uniform(1, 10))
    events.append(make_event(profile, session_id, t, "logout", "/logout", ctx))
    return events, shifts, variant


# ------------------------------------------------------------------
# STEP 4: Run the calendar
# ------------------------------------------------------------------
def session_start_time(profile, day, k):
    if profile["role"] == "occasional":
        hour = rng.uniform(6, 22)   # whenever they get a minute
    else:
        center = profile["start_hour"]
        if profile["remote_from"] is not None and day >= profile["remote_from"]:
            center = profile["remote_start_hour"]
        hour = float(np.clip(rng.normal(center + 3 * k, profile["hour_spread"]), 0, 23.5))
    return day + timedelta(hours=hour)


def generate_period(profiles, start_date, n_days, attack_rate, prefix):
    events, labels = [], []
    counter = 0
    for day_offset in range(n_days):
        day = start_date + timedelta(days=day_offset)
        is_weekend = day.weekday() >= 5
        for profile in profiles:
            if is_weekend and not profile["works_weekends"]:
                continue
            for k in range(rng.poisson(profile["sessions_per_day"])):
                counter += 1
                session_id = f"{prefix}_{counter:06d}"
                start_time = session_start_time(profile, day, k)
                label = {"session_id": session_id, "user_id": profile["user_id"],
                         "role": profile["role"]}

                if rng.random() < attack_rate:
                    attack_type = str(rng.choice(ATTACK_TYPES))
                    session_events, shifts, variant = attack_session(
                        profile, session_id, start_time, attack_type)
                    label.update(is_compromised=1, attack_type=attack_type, variant=variant,
                                 normal_scenario="none", **shifts)
                else:
                    session_events, scenario = normal_session(profile, session_id, start_time)
                    label.update(is_compromised=0, attack_type="none", variant="none",
                                 normal_scenario=scenario,
                                 odd_hour=False, new_location=False, new_device=False)
                labels.append(label)
                events.extend(session_events)

    logs = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)
    return logs, pd.DataFrame(labels)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    profiles = make_user_profiles(N_USERS)

    train_logs, _ = generate_period(profiles, TRAIN_START, TRAIN_DAYS, 0.0, "train")
    test_start = TRAIN_START + timedelta(days=TRAIN_DAYS)
    test_logs, test_labels = generate_period(profiles, test_start, TEST_DAYS, ATTACK_RATE, "test")

    train_logs.to_csv(RAW_DIR / "logs_train.csv", index=False)
    test_logs.to_csv(RAW_DIR / "logs_test.csv", index=False)
    test_labels.to_csv(LABEL_DIR / "test_labels.csv", index=False)
    pd.DataFrame(profiles)[["user_id", "role"]].to_csv(LABEL_DIR / "user_roles.csv", index=False)

    roles = pd.Series([p["role"] for p in profiles]).value_counts()
    n_attacks = int(test_labels["is_compromised"].sum())
    attacks = test_labels[test_labels["is_compromised"] == 1]
    print("Users by role: " + ", ".join(f"{r} {n}" for r, n in roles.items()))
    print(f"Hybrid users: {sum(p['second_ctx'] is not None for p in profiles)}, "
          f"remote switchers: {sum(p['remote_from'] is not None for p in profiles)}")
    print(f"Train: {len(train_logs):,} events, {train_logs['session_id'].nunique():,} sessions")
    print(f"Test:  {len(test_logs):,} events, {len(test_labels):,} sessions")
    print(f"Planted attacks: {n_attacks} ({n_attacks / len(test_labels):.1%} of test sessions)")
    print(attacks.groupby(["attack_type", "variant"]).size().to_string())


if __name__ == "__main__":
    main()
