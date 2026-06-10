import pytest
import pandas as pd
import joblib
from surprise import Dataset, Reader
from src.data.load_data import load_ratings   # adjust to your actual import

# ── Data tests ──────────────────────────────────────────
def test_data_loads():
    df = load_ratings()
    assert df is not None
    assert len(df) == 100_000

def test_no_nulls():
    df = load_ratings()
    assert df.isnull().sum().sum() == 0

def test_rating_range():
    df = load_ratings()
    assert df["rating"].between(1, 5).all()

def test_no_duplicates():
    df = load_ratings()
    assert df.duplicated(subset=["user_id", "movie_id"]).sum() == 0

# ── Model quality test ───────────────────────────────────
def test_model_rmse():
    """RMSE quality gate — blocks bad models from reaching prod."""
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import cross_validate
    import pandas as pd

    df = load_ratings()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "movie_id", "rating"]], reader)

    model = SVD(n_factors=20, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    results = cross_validate(model, data, measures=["RMSE"], cv=3, verbose=False)
    mean_rmse = results["test_rmse"].mean()

    assert mean_rmse < 1.0, f"RMSE {mean_rmse:.4f} exceeds quality gate of 1.0"

# ── Model artifact test ──────────────────────────────────
def test_model_artifact_loads():
    model = joblib.load("models/svd_best_model.pkl")
    assert model is not None

def test_model_predicts():
    model = joblib.load("models/svd_best_model.pkl")
    pred = model.predict(uid="1", iid="1")
    assert 1.0 <= pred.est <= 5.0