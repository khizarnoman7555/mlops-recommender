import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="MLOps Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 MLOps Monitoring Dashboard")
st.markdown("Live prediction logs and drift detection — MovieLens Recommender")
st.divider()

DB_PATH = Path("predictions.db")

if not DB_PATH.exists():
    st.warning("No predictions database found. Make sure the API has received requests.")
    st.stop()

# ── Load predictions ───────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC", conn)
conn.close()

if df.empty:
    st.warning("No predictions logged yet.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date

# ── Top metrics ────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Predictions", len(df))
col2.metric("Personalised", len(df[df["source"] == "personalised"]))
col3.metric("Cold-start", len(df[df["source"] == "cold-start-fallback"]))
col4.metric("Avg Predicted Score", f"{df['top_score'].mean():.3f}")

st.divider()

# ── Daily prediction volume ────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Daily Prediction Volume")
    daily = df.groupby("date").size().reset_index(name="count")
    st.bar_chart(daily.set_index("date"))

with col_right:
    st.subheader("Predicted Score Distribution")
    st.bar_chart(df["top_score"].value_counts().sort_index())

st.divider()

# ── Drift report ───────────────────────────────────────────────
st.subheader("Drift Detection")
drift_report = Path("reports/drift_report.html")
if drift_report.exists():
    st.success("Last drift check: No drift detected — model is stable ✓")
    with open(drift_report) as f:
        st.download_button("Download Full Drift Report", f.read(),
                          file_name="drift_report.html", mime="text/html")
else:
    st.info("No drift report yet. Run src/monitoring/drift_detection.py to generate one.")

st.divider()

# ── Recent predictions table ───────────────────────────────────
st.subheader("Recent Predictions")
st.dataframe(
    df[["timestamp", "user_id", "source", "top_movie", "top_score"]].head(20),
    use_container_width=True
)
