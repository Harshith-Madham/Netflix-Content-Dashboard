"""
app/streamlit_app.py
Streamlit dashboard for exploring Netflix data and getting recommendations
"""
import sys, os
# --- ensure parent folder (project root) is on Python path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.recommenders import ContentRecommender

st.set_page_config(page_title="🎬Netflix Content Dashboard", layout="wide")
st.title("🎬 Netflix Content Dashboard")

@st.cache_data
def load_recommender():
    return ContentRecommender("data/netflix_titles.csv")

recommender = load_recommender()
df = recommender.df

# =========================
# Data Explorer
# =========================
st.sidebar.header("Filters")
genre = st.sidebar.selectbox(
    "Filter by Genre",
    ["All"] + sorted(df["listed_in"].dropna().str.split(", ").explode().unique())
)
year_range = st.sidebar.slider(
    "Release Year Range",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    (2000, 2020)
)

filtered = df.copy()
if genre != "All":
    filtered = filtered[filtered["listed_in"].str.contains(genre, na=False)]
filtered = filtered[
    (filtered["release_year"] >= year_range[0]) &
    (filtered["release_year"] <= year_range[1])
]

st.subheader("Dataset Preview")
st.dataframe(filtered[["title", "release_year", "type", "listed_in", "rating"]].head(100))

# =========================
# Recommender
# =========================
st.subheader("Find Similar Titles")
user_title = st.text_input("Enter a title you like", "Breaking Bad")
if st.button("Recommend"):
    try:
        recs = recommender.recommend(user_title, topn=10)
        st.table(recs)
    except ValueError as e:
        st.error(str(e))

