# CineMatch: Movie Recommendation System

A content-based movie recommendation platform built with Python, Scikit-learn, and Streamlit, enriched with movie metadata, posters, trailers, and ratings from The Movie Database (TMDB) API.

---

## Project Overview

This project implements a content-based filtering recommendation engine. It processes movie metadata (genres, keywords, cast, director, release year, and plot overview) from the TMDb 5000 Movies Dataset, transforms textual features using NLP techniques, and calculates pairwise Cosine Similarities to deliver real-time recommendations.

---

## Features

- Content-Based Recommendation: Calculates similarity across 4,799 movies based on plot, genres, keywords, cast, and directors.
- Fast Local Search: Substring and prefix search across catalog movie titles without unnecessary external API requests.
- Recommendation Factors: Displays metadata-based reasons for each recommendation (shared director, shared cast, matching genres).
- TMDB API Enrichment: Displays high-resolution posters, backdrops, ratings, release dates, runtime, and cast credits.
- Trailer Integration: Extracts and links official YouTube trailers and teasers where available.
- Caching: Uses st.cache_resource for model data and st.cache_data for TMDB API calls with a 24-hour TTL.
- Offline Fallback: Remains operational in local mode when TMDB API is offline or unconfigured.
- Restrained Interface: Clean, dark cinematic styling designed around artwork and typography.
- Terms of Service and Privacy Policy: Includes policy and attribution statements.

---

## Architecture

```text
       ┌───────────────────────────┐     ┌───────────────────────────┐
       │   tmdb_5000_movies.csv    │     │   tmdb_5000_credits.csv   │
       └─────────────┬─────────────┘     └─────────────┬─────────────┘
                     └───────────────┬─────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │  Data Merging & Cleaning (4,799)│
                    └────────────────┬────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │ Feature Extraction (JSON parse) │
                    │ - Top 3 Cast, Director, Genres  │
                    │ - Keywords, Year, Overview      │
                    └────────────────┬────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │ Entity Space-Collapsing         │
                    │ ('Sam Worthington' -> 'Sam...') │
                    └────────────────┬────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │ PorterStemmer Normalization     │
                    └────────────────┬────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │ CountVectorizer (5,000 features)│
                    └────────────────┬────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │ Cosine Similarity Matrix (NxN)  │
                    └────────────────┬────────────────┘
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │                 Interactive Application                 │
        │                                                         │
        │   User Input -> Local Recommender Engine (Top N)        │
        │                           │                             │
        │                           ▼                             │
        │                   TMDB API Enrichment                   │
        │            (Posters, Ratings, Trailers, Cast)           │
        │                           │                             │
        │                           ▼                             │
        │               Streamlit Presentation UI                 │
        └─────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```text
Movie Recommendation/
│
├── app.py                     # Main Streamlit application
│
├── recommender/               # Modular recommendation package
│   ├── __init__.py            # Package exports
│   ├── config.py              # Configuration & secrets management
│   ├── engine.py              # Core recommendation, search & explainability
│   ├── preprocessing.py       # Data extraction & NLP text pipeline
│   └── tmdb.py                # TMDB API client with retry & fallback
│
│
├── assets/                    # Static assets & fallback graphics
│   └── default-poster.png     # Clean fallback movie poster
│
├── tests/                     # Unit test suites (17 tests)
│   ├── test_recommender.py    # Tests ranking, search, explainability
│   └── test_tmdb.py           # Tests TMDB API client and mock fallbacks
│
├── .streamlit/
│   └── config.toml            # Theme and server configurations
│
├── movies.pickle              # Processed movie metadata (4,799 rows)
├── similarity.joblib          # Pairwise cosine similarity matrix
├── requirements.txt           # Project dependencies
├── .gitignore                 # Protection for secrets and cache
└── README.md                  # Documentation
```

---

## NLP Pipeline

1. Tag Corpus Creation: Metadata fields are concatenated into a unified document tag: Overview + Genres + Keywords + Cast + Director + Release Year.
2. Stemming: Tokens are reduced to root stems using NLTK PorterStemmer.
3. Vectorization: Scikit-learn CountVectorizer generates a 5,000-dimensional Bag-of-Words representation with English stop-words removed.
4. Cosine Similarity: Pairwise similarity matrix calculated over all 4,799 feature vectors.

---

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure TMDB API Key (Optional)
To enable live posters, ratings, and trailers, obtain a Read Access Token from themoviedb.org and configure it via:

Environment Variable:
```bash
# Windows PowerShell
$env:TMDB_ACCESS_TOKEN="your_token_here"

# Linux / macOS
export TMDB_ACCESS_TOKEN="your_token_here"
```

Streamlit Secrets (.streamlit/secrets.toml):
```toml
TMDB_ACCESS_TOKEN = "your_token_here"
```

Local .env File:
```env
TMDB_ACCESS_TOKEN=your_token_here
```

### 3. Run Application
```bash
streamlit run app.py
```

---

## Testing

Execute the test suite with:
```bash
python -m pytest tests/
```

---

## Attribution & Legal

- Movie metadata, imagery, and trailers are provided courtesy of The Movie Database (TMDB).
- This product uses the TMDB API but is not endorsed or certified by TMDB.
- Terms of Service and Privacy Policy are accessible within the application interface.
