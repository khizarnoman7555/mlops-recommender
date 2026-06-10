import os
import json
import joblib
import pandas as pd
import mlflow
from surprise import SVD, NormalPredictor, Dataset, Reader
from surprise.model_selection import cross_validate, GridSearchCV

# ── 1. Connect to DagsHub via env vars (no OAuth) ─────────────
username = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

mlflow.set_tracking_uri(
    f"https://{username}:{password}@dagshub.com/khizarnoman7555/mlops-recommender.mlflow"
)

mlflow.set_experiment("svd-recommender")
print("Connected to DagsHub MLflow ✓")

# ── 2. Load data ───────────────────────────────────────────────
ratings = pd.read_csv(
    "data/ratings.dat",
    sep="\t",
    names=["userId", "movieId", "rating", "timestamp"]
)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
print("Data loaded ✓")

# ── 3. Log baseline run ────────────────────────────────────────
print("\nLogging baseline run...")
with mlflow.start_run(run_name="baseline-normal-predictor"):
    baseline = NormalPredictor()
    cv = cross_validate(baseline, data, measures=["RMSE", "MAE"], cv=5, verbose=False)
    rmse = cv["test_rmse"].mean()
    mae  = cv["test_mae"].mean()

    mlflow.log_param("model_type", "NormalPredictor")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse_std", cv["test_rmse"].std())
    print(f"Baseline RMSE: {rmse:.4f} ✓")

# ── 4. Grid search over SVD hyperparameters ────────────────────
print("\nRunning grid search — this will take ~5 minutes...")

param_grid = {
    "n_factors": [20, 50, 100],
    "lr_all":    [0.002, 0.005],
    "reg_all":   [0.02, 0.1],
}

gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=3, n_jobs=-1)
gs.fit(data)

print(f"Grid search complete ✓")
print(f"Best RMSE:   {gs.best_score['rmse']:.4f}")
print(f"Best params: {gs.best_params['rmse']}")

# ── 5. Log each grid search combination to MLflow ─────────────
print("\nLogging all grid search runs to DagsHub...")
results = gs.cv_results

for i in range(len(results["params"])):
    params = results["params"][i]
    mean_rmse = results["mean_test_rmse"][i]
    std_rmse  = results["std_test_rmse"][i]

    run_name = f"svd-f{params['n_factors']}-lr{params['lr_all']}-reg{params['reg_all']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "SVD")
        mlflow.log_param("n_factors",  params["n_factors"])
        mlflow.log_param("lr_all",     params["lr_all"])
        mlflow.log_param("reg_all",    params["reg_all"])
        mlflow.log_param("cv_folds",   3)
        mlflow.log_metric("rmse",      mean_rmse)
        mlflow.log_metric("rmse_std",  std_rmse)

print(f"Logged {len(results['params'])} runs ✓")

# ── 6. Train best model on full data and log it ────────────────
print("\nTraining best model on full dataset...")
best_params = gs.best_params["rmse"]

with mlflow.start_run(run_name="best-model-final"):
    best_model = SVD(
        n_factors=best_params["n_factors"],
        lr_all=best_params["lr_all"],
        reg_all=best_params["reg_all"],
        random_state=42
    )

    # 5-fold CV for reliable final metrics
    cv_final = cross_validate(best_model, data,
                              measures=["RMSE", "MAE"], cv=5, verbose=False)
    final_rmse = cv_final["test_rmse"].mean()
    final_mae  = cv_final["test_mae"].mean()

    # Train on full dataset
    trainset = data.build_full_trainset()
    best_model.fit(trainset)

    # Log everything
    mlflow.log_params(best_params)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_metric("rmse", final_rmse)
    mlflow.log_metric("mae",  final_mae)
    mlflow.log_metric("rmse_std", cv_final["test_rmse"].std())

    # Save and version model
    joblib.dump(best_model, "models/svd_best_model.pkl")
    mlflow.log_artifact("models/svd_best_model.pkl")

    print(f"Best model RMSE: {final_rmse:.4f}")
    print(f"Best model MAE:  {final_mae:.4f}")

# ── 7. Save updated metrics ────────────────────────────────────
metrics = {
    "best_rmse":   round(final_rmse, 4),
    "best_mae":    round(final_mae, 4),
    "best_params": best_params,
    "grid_search_runs": len(results["params"]),
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n" + "="*50)
print("ALL DONE")
print(f"View experiments at:")
print(f"https://dagshub.com/khizarnoman7555/mlops-recommender.mlflow")
print("="*50)