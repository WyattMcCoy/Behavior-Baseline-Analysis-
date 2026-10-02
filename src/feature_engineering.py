"""
feature_engineering.py  (version 2)
Turns raw log rows into one row of behavior measurements per session.

Three time windows:
  History (training days 1-20): used ONLY to learn each user's normal.
  Fit     (training days 21-30): normal sessions the model will train on.
  Test    (the 14 test days):    sessions the model will score.

New in version 2 (Travis's call): if a user has NO sessions in the History
window, there is no normal to compare against. Their sessions are marked
needs_manual_review = 1 and go to an analyst instead of the model.

Run from the project folder:  python src/feature_engineering.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


HISTORY_DAYS = 20

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SENSITIVE_PAGES = {"/admin", "/admin/users", "/billing", "/export", "/settings/security"}

FEATURE_COLUMNS = [
    "avg_gap_seconds",         # 1  speed
    "speed_vs_normal",         # 2  speed compared to this user
    "peak_actions_per_min",    # 3  bursts
    "n_actions",               # 4  volume
    "download_share",          # 5  volume
    "failed_logins",           # 6  logins
    "password_changed",        # 7  logins
    "sensitive_share",         # 8  pages
    "new_page_share",          # 9  pages compared to this user
    "hours_from_usual_login",  # 10 context compared to this user
    "new_city",                # 11 context compared to this user
    "new_device",              # 12 context compared to this user
    "failed_login_pace",       # 13 seconds between failed logins (Travis's call)
]

# Travis's call: sessions with fewer than 2 failed logins get 60 ("calm, human"),
# and real values are capped at 60 (anything slower than a minute is a person).
FAILED_PACE_CAP = 60


def load_logs(path):
    logs = pd.read_csv(path, parse_dates=["timestamp"])
    return logs.sort_values(["session_id", "timestamp"]).reset_index(drop=True)


# ------------------------------------------------------------------
# PART 1: Measurements that only need the session itself
# ------------------------------------------------------------------
def raw_session_features(logs):
    logs = logs.copy()
    logs["gap_seconds"] = logs.groupby("session_id")["timestamp"].diff().dt.total_seconds()
    logs["is_activity"] = ~logs["action_type"].isin(["login", "logout"])
    logs["is_download"] = logs["action_type"] == "download_file"
    logs["is_sensitive"] = logs["page_url"].isin(SENSITIVE_PAGES) & logs["is_activity"]
    logs["is_failed_login"] = (logs["action_type"] == "login") & (~logs["success_flag"])
    logs["is_pw_change"] = logs["action_type"] == "change_password"
    logs["minute"] = logs["timestamp"].dt.floor("min")

    feats = logs.groupby("session_id").agg(
        user_id=("user_id", "first"),
        start_time=("timestamp", "min"),
        city=("city", "first"),
        device_id=("device_id", "first"),
        n_actions=("is_activity", "sum"),
        avg_gap_seconds=("gap_seconds", "mean"),
        n_downloads=("is_download", "sum"),
        n_sensitive=("is_sensitive", "sum"),
        failed_logins=("is_failed_login", "sum"),
        password_changed=("is_pw_change", "max"),
    )

    activity = logs[logs["is_activity"]]
    per_minute = activity.groupby(["session_id", "minute"]).size()
    feats["peak_actions_per_min"] = per_minute.groupby(level="session_id").max()
    feats["peak_actions_per_min"] = feats["peak_actions_per_min"].fillna(0)

    safe_n = feats["n_actions"].replace(0, np.nan)
    feats["download_share"] = (feats["n_downloads"] / safe_n).fillna(0)
    feats["sensitive_share"] = (feats["n_sensitive"] / safe_n).fillna(0)

    # Pace of failed logins: a script retries in seconds, a person takes longer.
    failed = logs[logs["is_failed_login"]]
    fail_gaps = failed.groupby("session_id")["timestamp"].diff().dt.total_seconds()
    feats["failed_login_pace"] = fail_gaps.groupby(failed["session_id"]).mean()
    feats["failed_login_pace"] = (feats["failed_login_pace"]
                                  .fillna(FAILED_PACE_CAP)
                                  .clip(upper=FAILED_PACE_CAP))

    feats["login_hour"] = feats["start_time"].dt.hour + feats["start_time"].dt.minute / 60
    feats["password_changed"] = feats["password_changed"].astype(int)
    return feats


# ------------------------------------------------------------------
# PART 2: Learn each user's normal from the History window
# ------------------------------------------------------------------
def learn_user_normals(history_logs, history_feats):
    normals = history_feats.groupby("user_id").agg(
        usual_gap=("avg_gap_seconds", "median"),
        usual_login_hour=("login_hour", "median"),
        history_sessions=("n_actions", "size"),
    )
    known = {
        "pages": history_logs.groupby("user_id")["page_url"].apply(set),
        "cities": history_logs.groupby("user_id")["city"].apply(set),
        "devices": history_logs.groupby("user_id")["device_id"].apply(set),
    }
    return normals, known


# ------------------------------------------------------------------
# PART 3: Compare each session to that user's normal
# ------------------------------------------------------------------
def add_relative_features(feats, logs, normals, known):
    feats = feats.join(normals, on="user_id")

    # No history = nothing to compare to. Send to a person instead of the model.
    feats["history_sessions"] = feats["history_sessions"].fillna(0).astype(int)
    feats["needs_manual_review"] = (feats["history_sessions"] == 0).astype(int)

    feats["speed_vs_normal"] = feats["avg_gap_seconds"] / feats["usual_gap"]

    diff = (feats["login_hour"] - feats["usual_login_hour"]).abs()
    feats["hours_from_usual_login"] = np.minimum(diff, 24 - diff)

    feats["new_city"] = [int(c not in known["cities"].get(u, set()))
                         for u, c in zip(feats["user_id"], feats["city"])]
    feats["new_device"] = [int(d not in known["devices"].get(u, set()))
                           for u, d in zip(feats["user_id"], feats["device_id"])]

    activity = logs[~logs["action_type"].isin(["login", "logout"])]
    is_new_page = pd.Series(
        [p not in known["pages"].get(u, set())
         for u, p in zip(activity["user_id"], activity["page_url"])],
        index=activity.index,
    )
    feats["new_page_share"] = is_new_page.groupby(activity["session_id"]).mean()
    feats["new_page_share"] = feats["new_page_share"].fillna(0)
    return feats


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_logs = load_logs(RAW_DIR / "logs_train.csv")
    test_logs = load_logs(RAW_DIR / "logs_test.csv")

    cutoff = train_logs["timestamp"].min().normalize() + pd.Timedelta(days=HISTORY_DAYS)
    session_start = train_logs.groupby("session_id")["timestamp"].transform("min")
    history_logs = train_logs[session_start < cutoff]
    fit_logs = train_logs[session_start >= cutoff]

    history_feats = raw_session_features(history_logs)
    normals, known = learn_user_normals(history_logs, history_feats)

    for name, logs in [("fit", fit_logs), ("test", test_logs)]:
        feats = raw_session_features(logs)
        feats = add_relative_features(feats, logs, normals, known)
        feats.to_parquet(PROCESSED_DIR / f"features_{name}.parquet")

        review = feats["needs_manual_review"] == 1
        missing = int(feats.loc[~review, FEATURE_COLUMNS].isna().sum().sum())
        print(f"{name}: {len(feats):,} sessions | {int(review.sum())} need manual review "
              f"({feats.loc[review, 'user_id'].nunique()} users) | "
              f"{missing} missing values among the rest")

    thin = normals[normals["history_sessions"] < 5]
    print(f"Users with 1-4 History sessions (thin baseline): {len(thin)}")


if __name__ == "__main__":
    main()
