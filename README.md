# 🎬 Netflix Content Dashboard
Netflix Content Dashboard – Data Analysis &amp; ML App End-to-end project using Python, pandas, scikit-learn, and Streamlit. Includes full EDA of 8.8k Netflix titles and a content-based recommender (TF-IDF + cosine similarity) served via an interactive web dashboard.


## Overview
* Performed **Exploratory Data Analysis (EDA)** on 8,800+ Netflix titles to understand genre trends, ratings, and durations.
* Engineered a combined **metadata text field** and built a **TF-IDF + Cosine Similarity** recommender.
* Created an interactive **Streamlit dashboard** for browsing titles, filtering by genre/year, and getting similar-title recommendations.

---

## Features
- 📊 Interactive visualizations of genres, ratings, and release trends.
- 🔍 Powerful search and filtering by genre and year.
- 🤖 Content-based movie/TV recommendations using machine learning.
- 🌐 One-click web deployment via Streamlit Community Cloud.

---

## Tech Stack
| Category            | Tools & Libraries                          |
|---------------------|--------------------------------------------|
| Programming         | Python 3.10                                |
| Data Analysis       | pandas, numpy, matplotlib, seaborn, plotly |
| Machine Learning    | scikit-learn, nltk                         |
| Web App Framework   | Streamlit                                  |
| Environment Mgmt    | Conda                                      |
| Version Control     | Git & GitHub                               |

---

## Project Structure
Netflix_Content_Dashboard/
│ README.md
│ requirements.txt
  ─ data/ # Raw data (Netflix CSV) and instructions
├─ notebooks/ # Jupyter notebooks for EDA & modeling
├─ src/ # Python modules (recommenders.py)
   ─ recommenders.py
└─ app/ # Streamlit dashboard
   ─ streamlit_app.py
