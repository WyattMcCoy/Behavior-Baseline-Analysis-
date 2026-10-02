"""
evaluate.py  (version 2)
Opens the answer key and checks the model's flags against it.

Reports:
  1. Results at three flag rates (1%, 3.75%, 5%) for model-scored sessions
  2. Caught vs. planted by attack type and loud/quiet variant, at 3.75%
  3. The attacks that slipped through
  4. False alarms, with the real-life reason each normal session looked odd
  5. The manual review queue
  6. A chart of scores for normal vs. attack sessions

Run from the project folder:  python src/evaluate.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

RATES_TO_COMPARE = [0.01, 0.0375, 0.05]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LABEL_DIR = PROJECT_ROOT / "data" / "labels"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.read_parquet(PROCESSED_DIR / "scores_test.parquet")
    labels = pd.read_csv(LABEL_DIR / "test_labels.csv", index_col="session_id")
    labels = labels.drop(columns="user_id").rename(columns={
        "odd_hour": "planted_odd_hour",
        "new_location": "planted_new_location",
        "new_device": "planted_new_device",
    })
    everything = results.join(labels)

    review = everything[everything["needs_manual_review"] == 1]
    df = everything[everything["needs_manual_review"] == 0]
    is_attack = df["is_compromised"] == 1
    total_attacks = int(is_attack.sum())

    # ---- 1. Compare flag rates ----
    rows = []
    for rate in RATES_TO_COMPARE:
        cutoff = df["suspicion_score"].quantile(1 - rate)
        flagged = df["suspicion_score"] >= cutoff
        caught = int((flagged & is_attack).sum())
        rows.append({
            "flag_rate": f"{rate:.2%}",
            "flagged": int(flagged.sum()),
            "caught": caught,
            "missed": total_attacks - caught,
            "false_alarms": int((flagged & ~is_attack).sum()),
            "precision": caught / flagged.sum(),
            "recall": caught / total_attacks,
        })
    comparison = pd.DataFrame(rows)
    auc = roc_auc_score(df["is_compromised"], df["suspicion_score"])

    print(f"Model-scored sessions: {len(df):,}   Attacks among them: {total_attacks}")
    print(f"ROC-AUC: {auc:.3f}  (1.0 = perfect ranking, 0.5 = coin flip)\n")
    print(comparison.round(2).to_string(index=False))
    comparison.to_csv(REPORTS_DIR / "flag_rate_comparison.csv", index=False)

    # ---- 2. By attack type and variant ----
    attacks = df[is_attack]
    by_type = attacks.groupby(["attack_type", "variant"]).agg(
        planted=("flagged", "size"), caught=("flagged", "sum"))
    print("\nAt 3.75%, by attack type:")
    print(by_type.to_string())

    # ---- 3. Missed attacks ----
    missed = attacks[attacks["flagged"] == 0].sort_values("rank")
    print(f"\nMissed attacks ({len(missed)}):")
    if len(missed):
        print(missed[["user_id", "role", "attack_type", "variant", "rank",
                      "planted_odd_hour", "planted_new_location", "planted_new_device"]].to_string())

    # ---- 4. False alarms and why ----
    false_alarms = df[(df["flagged"] == 1) & ~is_attack].sort_values("rank")
    print(f"\nFalse alarms ({len(false_alarms)}), by real-life reason:")
    print(false_alarms["normal_scenario"].value_counts().to_string())
    print("\nFalse alarm details:")
    print(false_alarms[["user_id", "role", "normal_scenario", "rank", "failed_logins",
                        "new_city", "new_device", "hours_from_usual_login",
                        "history_sessions"]].round(2).to_string())

    # ---- 5. Manual review queue ----
    print(f"\nManual review queue: {len(review)} sessions from "
          f"{review['user_id'].nunique()} users with no history, "
          f"{int(review['is_compromised'].sum())} of them attacks")

    # ---- 5b. Triage check (only if triage.py has been run) ----
    triage_path = PROCESSED_DIR / "triage_test.parquet"
    if triage_path.exists():
        triage = pd.read_parquet(triage_path)[["priority", "reasons"]].join(labels)
        queue = triage[triage["priority"] != "NOT_FLAGGED"]
        queue = queue.assign(truth=queue["is_compromised"].map({1: "attack", 0: "normal"}))
        print("\nTriage priority vs. truth (flagged sessions only):")
        print(pd.crosstab(queue["priority"], queue["truth"]).to_string())
        print("\nNormal sessions by priority and real-life reason:")
        normals = queue[queue["truth"] == "normal"]
        print(pd.crosstab(normals["normal_scenario"], normals["priority"]).to_string())

    # ---- 6. Chart ----
    cutoff = df.loc[df["flagged"] == 1, "suspicion_score"].min()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df.loc[~is_attack, "suspicion_score"], bins=40, alpha=0.6, label="Normal")
    ax.hist(df.loc[is_attack, "suspicion_score"], bins=40, alpha=0.8, label="Attack")
    ax.axvline(cutoff, linestyle="--", color="black", label="3.75% flag line")
    ax.set_xlabel("Suspicion score (higher = more suspicious)")
    ax.set_ylabel("Number of sessions")
    ax.set_title("Suspicion scores: normal vs. planted attacks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "score_distribution.png", dpi=150)
    print(f"\nChart saved to {REPORTS_DIR / 'score_distribution.png'}")


if __name__ == "__main__":
    main()
