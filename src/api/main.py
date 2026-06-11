import json
import sqlite3
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI
import pandas as pd

# ── Load pre-computed recommendations ─────────────────────────
RECS_PATH = Path("models/recommendations.json")
with open(RECS_PATH) as f:
    recommendations = {int(k): v for k, v in json.load(f).items()}

# ── Top-10 popular movies as cold-start fallback ───────────────
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

# ── SQLite logging setup ───────────────────────────────────────
DB_PATH = Path("predictions.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            user_id     INTEGER NOT NULL,
            source      TEXT NOT NULL,
            top_movie   TEXT,
            top_score   REAL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(user_id: int, source: str, top_movie: str, top_score: float):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (timestamp, user_id, source, top_movie, top_score) VALUES (?,?,?,?,?)",
        (datetime.utcnow().isoformat(), user_id, source, top_movie, top_score)
    )
    conn.commit()
    conn.close()

init_db()

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
        recs = recommendations[user_id]
        source = "personalised"
    else:
        recs = fallback
        source = "cold-start-fallback"

    # Log the prediction
    log_prediction(
        user_id=user_id,
        source=source,
        top_movie=recs[0]["title"],
        top_score=recs[0]["predicted_rating"]
    )

    return {"user_id": user_id, "recommendations": recs, "source": source}

@app.get("/health")
def health():
    return {"status": "healthy", "users_served": len(recommendations)}

@app.get("/stats")
def stats():
    """Return prediction stats from SQLite log."""
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    personalised = conn.execute("SELECT COUNT(*) FROM predictions WHERE source='personalised'").fetchone()[0]
    cold_start = conn.execute("SELECT COUNT(*) FROM predictions WHERE source='cold-start-fallback'").fetchone()[0]
    recent = conn.execute(
        "SELECT timestamp, user_id, source, top_movie FROM predictions ORDER BY id DESC LIMIT 5"
    ).fetchall()
    conn.close()
    return {
        "total_predictions": total,
        "personalised": personalised,
        "cold_start": cold_start,
        "recent": [{"timestamp": r[0], "user_id": r[1], "source": r[2], "top_movie": r[3]} for r in recent]
    }