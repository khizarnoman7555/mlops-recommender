import json
import joblib
import numpy as np
from surprise import SVD, NormalPredictor, Dataset, Reader
from surprise.model_selection import cross_validate

# ── 1. Load data into Surprise format ─────────────────────────
import pandas as pd

ratings = pd.read_csv(
    "data/ratings.dat",
    sep="\t",
    names=["userId", "movieId", "rating", "timestamp"]
)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

print("Data loaded into Surprise format ✓")
print(f"Ratings: {len(ratings):,}")

# ── 2. Baseline model — random predictor ──────────────────────
print("\n" + "="*50)
print("BASELINE — NormalPredictor (random)")
print("="*50)

baseline = NormalPredictor()
cv_baseline = cross_validate(
    baseline, data,
    measures=["RMSE", "MAE"],
    cv=5, verbose=False
)
baseline_rmse = cv_baseline["test_rmse"].mean()
baseline_mae  = cv_baseline["test_mae"].mean()

print(f"RMSE: {baseline_rmse:.4f} (+/- {cv_baseline['test_rmse'].std():.4f})")
print(f"MAE:  {baseline_mae:.4f} (+/- {cv_baseline['test_mae'].std():.4f})")

# ── 3. SVD model ───────────────────────────────────────────────
print("\n" + "="*50)
print("SVD COLLABORATIVE FILTER")
print("="*50)

model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
cv_svd = cross_validate(
    model, data,
    measures=["RMSE", "MAE"],
    cv=5, verbose=False
)
svd_rmse = cv_svd["test_rmse"].mean()
svd_mae  = cv_svd["test_mae"].mean()

print(f"RMSE: {svd_rmse:.4f} (+/- {cv_svd['test_rmse'].std():.4f})")
print(f"MAE:  {svd_mae:.4f} (+/- {cv_svd['test_mae'].std():.4f})")

# ── 4. Compare ─────────────────────────────────────────────────
print("\n" + "="*50)
print("COMPARISON")
print("="*50)
improvement = ((baseline_rmse - svd_rmse) / baseline_rmse) * 100
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"SVD RMSE:      {svd_rmse:.4f}")
print(f"Improvement:   {improvement:.1f}%")
print(f"Beat baseline: {'YES ✓' if svd_rmse < baseline_rmse else 'NO ✗'}")

# ── 5. Train on full dataset and save ─────────────────────────
print("\n" + "="*50)
print("TRAINING ON FULL DATASET")
print("="*50)

trainset = data.build_full_trainset()
model.fit(trainset)

joblib.dump(model, "models/svd_model.pkl")
print("Model saved → models/svd_model.pkl ✓")

# ── 6. Save metrics ────────────────────────────────────────────
metrics = {
    "svd_rmse":      round(svd_rmse, 4),
    "svd_mae":       round(svd_mae, 4),
    "svd_rmse_std":  round(cv_svd["test_rmse"].std(), 4),
    "baseline_rmse": round(baseline_rmse, 4),
    "improvement_pct": round(improvement, 2),
    "n_factors": 50,
    "n_epochs":  20,
    "cv_folds":  5,
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved → metrics.json ✓")

print("\n" + "="*50)
print("DONE")
print("="*50)