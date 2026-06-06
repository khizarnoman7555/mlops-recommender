import pandas as pd

# ── Load ratings ──────────────────────────────────────────────
ratings = pd.read_csv(
    "data/ratings.dat",
    sep="\t",
    names=["userId", "movieId", "rating", "timestamp"],
)

# ── Load movies ───────────────────────────────────────────────
movies = pd.read_csv(
    "data/movies.dat",
    sep="|",
    encoding="latin-1",
    usecols=[0, 1],
    names=["movieId", "title"],
)

# ── Basic validation ──────────────────────────────────────────
print("=" * 40)
print("RATINGS")
print("=" * 40)
print(f"Shape:      {ratings.shape}")
print(f"Nulls:      {ratings.isnull().sum().sum()}")
print(f"Duplicates: {ratings.duplicated().sum()}")
print(f"Users:      {ratings['userId'].nunique()}")
print(f"Movies:     {ratings['movieId'].nunique()}")
print(f"Rating range: {ratings['rating'].min()} – {ratings['rating'].max()}")
print()
print(ratings.head())

print()
print("=" * 40)
print("MOVIES")
print("=" * 40)
print(f"Shape: {movies.shape}")
print()
print(movies.head())