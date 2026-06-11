import json
import joblib
import pandas as pd
from pathlib import Path

# ── Load model and data ────────────────────────────────────────
print("Loading model and data...")
model = joblib.load("models/svd_best_model.pkl")

ratings = pd.read_csv(
    "data/ratings.dat",
    sep="\t",
    names=["userId", "movieId", "rating", "timestamp"]
)

movies = pd.read_csv(
    "data/movies.dat",
    sep="|",
    encoding="latin-1",
    usecols=[0, 1],
    names=["movieId", "title"]
)

# ── Build lookup: movieId → title ─────────────────────────────
movie_titles = dict(zip(movies["movieId"], movies["title"]))

# ── Get all user and movie IDs ─────────────────────────────────
all_users  = ratings["userId"].unique()
all_movies = ratings["movieId"].unique()

# ── Pre-compute top-10 recommendations for every user ─────────
print(f"Pre-computing recommendations for {len(all_users)} users...")
recommendations = {}

for i, user_id in enumerate(all_users):
    # Movies this user has already rated
    rated = set(ratings[ratings["userId"] == user_id]["movieId"])

    # Predict ratings for all unrated movies
    predictions = [
        (movie_id, model.predict(user_id, movie_id).est)
        for movie_id in all_movies
        if movie_id not in rated
    ]

    # Top 10 by predicted rating
    top10 = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]

    recommendations[int(user_id)] = [
        {"movieId": int(mid), "title": movie_titles.get(mid, "Unknown"), "predicted_rating": round(score, 3)}
        for mid, score in top10
    ]

    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(all_users)} users done...")

# ── Save to JSON ───────────────────────────────────────────────
Path("models").mkdir(exist_ok=True)
with open("models/recommendations.json", "w") as f:
    json.dump(recommendations, f)

print(f"Saved recommendations for {len(recommendations)} users → models/recommendations.json ✓")
