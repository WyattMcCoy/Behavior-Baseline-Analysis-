"""
triage.py
Runs AFTER the model. For every flagged session, attaches plain-English reasons
and a priority (HIGH / MEDIUM / LOW) so the analyst knows where to start.

Why: the model finds UNUSUAL, not BAD. These rules add human judgment about
which kinds of unusual are dangerous.

Like train_and_score.py, this never opens the answer key.

Run from the project folder:  python src/triage.py
"""

from pathlib import Path

import pandas as pd

# Rule thresholds (plain numbers an analyst could argue about and change)
RAPID_FAIL_PACE = 5          # seconds between failed logins: faster = script
FAST_RATIO = 0.25            # 4x faster than this user's usual pace
HEAVY_DOWNLOAD_SHARE = 0.5   # half or more of actions are downloads
NEW_PAGE_SHARE = 0.1         # 10%+ of visits to pages this user never used
ODD_HOUR_GAP = 4             # 4+ hours from usual login time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


def reasons_for(row):
    """Turn the numbers into short evidence tags an analyst can read."""
    reasons = []
    if row["failed_logins"] >= 2:
        if row["failed_login_pace"] < RAPID_FAIL_PACE:
            reasons.append("RAPID_FAILED_LOGINS")
        else:
            reasons.append("HUMAN_PACE_FAILED_LOGINS")
    if row["speed_vs_normal"] < FAST_RATIO:
        reasons.append("MUCH_FASTER_THAN_USUAL")
    if row["download_share"] >= HEAVY_DOWNLOAD_SHARE:
        reasons.append("HEAVY_DOWNLOADS")
    if row["new_page_share"] >= NEW_PAGE_SHARE:
        reasons.append("NEW_PAGES_FOR_USER")
    if row["sensitive_share"] > 0:
        reasons.append("SENSITIVE_PAGES")
    if row["new_city"]:
        reasons.append("NEW_CITY")
    if row["new_device"]:
        reasons.append("NEW_DEVICE")
    if row["hours_from_usual_login"] >= ODD_HOUR_GAP:
        reasons.append("UNUSUAL_HOUR")
    if row["password_changed"]:
        reasons.append("PASSWORD_CHANGED")
    return reasons


def priority_for(reasons):
    known_place = "NEW_CITY" not in reasons and "NEW_DEVICE" not in reasons

    # Travis's rule: forgot-password pattern from the user's own laptop and city.
    if "HUMAN_PACE_FAILED_LOGINS" in reasons and known_place:
        return "LOW"
    # Patterns that point to an attack, not a person having an odd day.
    if "RAPID_FAILED_LOGINS" in reasons:
        return "HIGH"
    if "MUCH_FASTER_THAN_USUAL" in reasons and "HEAVY_DOWNLOADS" in reasons:
        return "HIGH"
    if "NEW_PAGES_FOR_USER" in reasons and "SENSITIVE_PAGES" in reasons:
        return "HIGH"
    return "MEDIUM"


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.read_parquet(PROCESSED_DIR / "scores_test.parquet")

    results["reasons"] = ""
    results["priority"] = "NOT_FLAGGED"

    flagged = results["flagged"] == 1
    tags = results[flagged].apply(reasons_for, axis=1)
    results.loc[flagged, "reasons"] = tags.apply(", ".join)
    results.loc[flagged, "priority"] = tags.apply(priority_for)
    results.loc[results["needs_manual_review"] == 1, "priority"] = "MANUAL_REVIEW"

    results.to_parquet(PROCESSED_DIR / "triage_test.parquet")

    # The analyst's work list: highest priority first, then highest score.
    order = {"HIGH": 0, "MANUAL_REVIEW": 1, "MEDIUM": 2, "LOW": 3}
    queue = results[results["priority"].isin(order)].copy()
    queue["_order"] = queue["priority"].map(order)
    queue = queue.sort_values(["_order", "suspicion_score"], ascending=[True, False])
    columns = ["priority", "user_id", "start_time", "suspicion_score", "city", "device_id", "reasons"]
    queue[columns].to_csv(REPORTS_DIR / "analyst_queue.csv")

    print("Analyst queue by priority:")
    print(queue["priority"].value_counts().reindex(order).dropna().astype(int).to_string())
    print(f"\nSaved to {REPORTS_DIR / 'analyst_queue.csv'}")
    print("\nFirst 8 items in the queue:")
    print(queue[["priority", "user_id", "reasons"]].head(8).to_string())


if __name__ == "__main__":
    main()
