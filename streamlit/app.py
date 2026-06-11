import streamlit as st
import requests

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommender")
st.markdown("Powered by SVD Collaborative Filtering · MovieLens 100K")
st.divider()

API_URL = st.secrets.get("API_URL", "http://localhost:8000")

user_id = st.number_input(
    "Enter a User ID (1–943)",
    min_value=1, max_value=9999,
    value=1, step=1
)

if st.button("Get Recommendations", type="primary"):
    with st.spinner("Fetching recommendations..."):
        try:
            res = requests.get(f"{API_URL}/recommend/{user_id}", timeout=10)
            data = res.json()

            if data["source"] == "personalised":
                st.success(f"Personalised recommendations for User {user_id}")
            else:
                st.info(f"User {user_id} not found — showing popular movies instead")

            for i, movie in enumerate(data["recommendations"], 1):
                st.markdown(
                    f"**{i}.** {movie['title']} &nbsp; "
                    f"`⭐ {movie['predicted_rating']}`"
                )

        except Exception as e:
            st.error(f"Could not reach API: {e}")
            st.markdown("Make sure the FastAPI server is running.")

st.divider()
st.caption("MLOps Project · Phase 6 · Built with FastAPI + Streamlit")
