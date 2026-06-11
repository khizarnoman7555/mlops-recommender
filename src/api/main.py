import json
from fastapi import FastAPI, HTTPException
from pathlib import Path

# ── Load pre-computed recommendations ─────────────────────────
RECS_PATH = Path("models/recommendations.json")

with open(RECS_PATH) as f:
    recommendations = {int(k): v for k, v in json.load(f).items()}

# ── Top 10 popular movies as cold-start fallback ───────────────
import pandas as pd

ratings = pd.read_csv(
    "data/ratings.dat", sep="\t",
    names=["userId", "movieId", "rating", "timestamp"]
)
movies = pd.read_csv(
    "data/movies.dat", sep="|", encoding="latin-1",
    usecols=[0, 1], names=["movieId", "title"]
)
movie_titles = dict(zip(movies["movieId"], movies["title"]))

top_popular = (
    ratings.groupby("movieId")["rating"]
    .agg(["mean", "count"])
    .query("count >= 50")
    .sort_values("mean", ascending=False)
    .head(10)
    .reset_index()
)
fallback = [
    {"movieId": int(row.movieId), "title": movie_titles.get(row.movieId, "Unknown"), "predicted_rating": round(row.mean, 3)}
    for row in top_popular.itertuples()
]

# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="Movie Recommender API",
    description="SVD collaborative filtering on MovieLens 100K",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Movie Recommender API is running"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    if user_id in recommendations:
        return {
            "user_id": user_id,
            "recommendations": recommendations[user_id],
            "source": "personalised"
        }
    else:
        return {
            "user_id": user_id,
            "recommendations": fallback,
            "source": "cold-start-fallback"
        }

@app.get("/health")
def health():
    return {"status": "healthy", "users_served": len(recommendations)}
