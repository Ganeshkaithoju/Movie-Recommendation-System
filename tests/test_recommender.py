"""
Tests for the recommendation engine, ranking, explainability, search, filtering, and sorting.
"""

import numpy as np
import pandas as pd
import pytest
from recommender.engine import recommend, search_movies, explain_recommendation


@pytest.fixture
def mock_dataset():
    """Creates a mock movies dataframe and similarity matrix."""
    movies_data = {
        "id": [101, 102, 103, 104, 105, 106],
        "title": [
            "Inception",
            "Interstellar",
            "The Dark Knight",
            "Memento",
            "Titanic",
            "Avatar",
        ],
        "tags": ["dream sci-fi", "space sci-fi", "batman crime", "memory thriller", "ship romance", "alien sci-fi"],
        "release_year": [2010, 2014, 2008, 2000, 1997, 2009],
        "vote_average": [8.8, 8.6, 9.0, 8.4, 7.8, 7.5],
        "genres": [
            ["Action", "Science Fiction"],
            ["Adventure", "Drama", "Science Fiction"],
            ["Action", "Crime", "Drama"],
            ["Mystery", "Thriller"],
            ["Drama", "Romance"],
            ["Action", "Adventure", "Science Fiction"],
        ],
        "director": [
            ["Christopher Nolan"],
            ["Christopher Nolan"],
            ["Christopher Nolan"],
            ["Christopher Nolan"],
            ["James Cameron"],
            ["James Cameron"],
        ],
        "cast": [
            ["Leonardo DiCaprio", "Joseph Gordon-Levitt"],
            ["Matthew McConaughey", "Anne Hathaway"],
            ["Christian Bale", "Heath Ledger"],
            ["Guy Pearce", "Carrie-Anne Moss"],
            ["Leonardo DiCaprio", "Kate Winslet"],
            ["Sam Worthington", "Zoe Saldana"],
        ],
    }
    df = pd.DataFrame(movies_data)

    # 6x6 dummy cosine similarity matrix with 1.0 on diagonal
    sim = np.array([
        [1.00, 0.85, 0.70, 0.65, 0.40, 0.30],  # Inception (0)
        [0.85, 1.00, 0.60, 0.50, 0.35, 0.45],  # Interstellar (1)
        [0.70, 0.60, 1.00, 0.55, 0.20, 0.25],  # The Dark Knight (2)
        [0.65, 0.50, 0.55, 1.00, 0.15, 0.10],  # Memento (3)
        [0.40, 0.35, 0.20, 0.15, 1.00, 0.50],  # Titanic (4)
        [0.30, 0.45, 0.25, 0.10, 0.50, 1.00],  # Avatar (5)
    ])
    return df, sim


def test_recommend_excludes_selected_movie(mock_dataset):
    df, sim = mock_dataset
    target_idx = 0  # Inception
    recs = recommend(target_idx, sim, df, n=3)

    rec_indices = [r["index"] for r in recs]
    rec_ids = [r["id"] for r in recs]

    assert target_idx not in rec_indices
    assert df.iloc[target_idx]["id"] not in rec_ids


def test_recommend_respects_count_n(mock_dataset):
    df, sim = mock_dataset
    for n in (1, 3, 5):
        recs = recommend(0, sim, df, n=n)
        assert len(recs) == n


def test_recommend_similarity_scores_ordered_descending(mock_dataset):
    df, sim = mock_dataset
    recs = recommend(0, sim, df, n=5)
    scores = [r["similarity_score"] for r in recs]

    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


def test_recommend_sort_early_first(mock_dataset):
    df, sim = mock_dataset
    recs = recommend(0, sim, df, n=5, sort_type="Early First")
    years = [r["release_year"] for r in recs if r["release_year"] is not None]

    for i in range(len(years) - 1):
        assert years[i] <= years[i + 1]


def test_recommend_sort_top_rated(mock_dataset):
    df, sim = mock_dataset
    recs = recommend(0, sim, df, n=5, sort_type="Top Rated")
    ratings = [r["vote_average"] for r in recs]

    for i in range(len(ratings) - 1):
        assert ratings[i] >= ratings[i + 1]


def test_recommend_year_filtering(mock_dataset):
    df, sim = mock_dataset
    # Filter 2000 - 2009
    recs = recommend(0, sim, df, n=5, year_filter="2000 - 2009")
    for r in recs:
        assert 2000 <= r["release_year"] <= 2009


def test_explain_recommendation_finds_director_and_cast(mock_dataset):
    df, _ = mock_dataset
    target = df.iloc[0]
    candidate = df.iloc[4]
    reasons = explain_recommendation(target, candidate, 0.40)
    assert any("Leonardo DiCaprio" in r for r in reasons)

    nolan_cand = df.iloc[1]
    nolan_reasons = explain_recommendation(target, nolan_cand, 0.85)
    assert any("Christopher Nolan" in r for r in nolan_reasons)


def test_search_movies(mock_dataset):
    df, _ = mock_dataset
    matches = search_movies("in", df, limit=5)
    assert len(matches) >= 2
    titles = [m["title"] for m in matches]
    assert "Inception" in titles
    assert "Interstellar" in titles

    knight_matches = search_movies("knight", df, limit=5)
    assert len(knight_matches) == 1
    assert knight_matches[0]["title"] == "The Dark Knight"

    assert search_movies("", df) == []
    assert search_movies("   ", df) == []
