"""
train_and_score.py  (version 2)
Trains an Isolation Forest on normal sessions (the Fit window), then scores
test sessions and flags the top 3.75%.

Sessions marked needs_manual_review (users with no history) skip the model
and go straight to an analyst.

This script never opens the answer key. Checking results is evaluate.py.

Run from the project folder:  python src/train_and_score.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from feature_engineering import FEATURE_COLUMNS

FLAG_RATE = 0.0375   # Travis's call
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    fit = pd.read_parquet(PROCESSED_DIR / "features_fit.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "features_test.parquet")

    # The model can only learn from, and score, sessions that have a baseline.
    fit = fit[fit["needs_manual_review"] == 0]
    scored = test[test["needs_manual_review"] == 0].copy()
    review = test[test["needs_manual_review"] == 1].copy()

    model = IsolationForest(n_estimators=200, random_state=SEED)
    model.fit(fit[FEATURE_COLUMNS])
    joblib.dump(model, MODELS_DIR / "isolation_forest.joblib")

    scored["suspicion_score"] = -model.score_samples(scored[FEATURE_COLUMNS])
    threshold = scored["suspicion_score"].quantile(1 - FLAG_RATE)
    scored["flagged"] = (scored["suspicion_score"] >= threshold).astype(int)
    scored["rank"] = scored["suspicion_score"].rank(ascending=False).astype(int)

    review["suspicion_score"] = np.nan
    review["flagged"] = 0
    review["rank"] = -1

    results = pd.concat([scored, review])
    results.to_parquet(PROCESSED_DIR / "scores_test.parquet")

    print(f"Trained on {len(fit):,} normal sessions")
    print(f"Model flagged {scored['flagged'].sum()} of {len(scored):,} scored sessions "
          f"(top {FLAG_RATE:.2%}, score >= {threshold:.3f})")
    print(f"Sent to manual review (no history): {len(review)}")
    print(f"Total analyst queue: {scored['flagged'].sum() + len(review)}")

    print("\nTop 10 most suspicious sessions:")
    show = ["user_id", "suspicion_score", "speed_vs_normal", "download_share",
            "sensitive_share", "failed_logins", "new_city", "new_device"]
    print(scored.sort_values("suspicion_score", ascending=False)[show].head(10).round(2).to_string())


if __name__ == "__main__":
    main()
