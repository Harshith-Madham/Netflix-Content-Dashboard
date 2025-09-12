"""
src/recommenders.py
Content-based recommender for Netflix titles
"""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

stop = set(stopwords.words("english"))

def clean_text(s: str) -> str:
    """Lowercase, remove punctuation, strip stopwords."""
    if pd.isna(s):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [w for w in s.split() if w not in stop]
    return " ".join(tokens)

class ContentRecommender:
    """Builds a TF-IDF content similarity model on Netflix titles."""
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        # Combine description, genres, director, and cast
        self.df["meta"] = (
            self.df["description"].apply(clean_text) + " " +
            self.df["listed_in"].apply(clean_text) + " " +
            self.df["director"].fillna("").apply(clean_text) + " " +
            self.df["cast"].fillna("").apply(clean_text)
        )

        # Vectorize and compute TF-IDF matrix
        self.tfidf = TfidfVectorizer(max_features=10000)
        self.matrix = self.tfidf.fit_transform(self.df["meta"])

    def recommend(self, title: str, topn: int = 10) -> pd.DataFrame:
        """Return top-N similar titles for a given title."""
        if title not in self.df["title"].values:
            raise ValueError(f"'{title}' not found in dataset.")
        idx = self.df[self.df["title"] == title].index[0]
        sims = cosine_similarity(self.matrix[idx], self.matrix).flatten()
        sim_idx = sims.argsort()[-topn-1:-1][::-1]
        return self.df.iloc[sim_idx][["title", "release_year", "listed_in"]]
