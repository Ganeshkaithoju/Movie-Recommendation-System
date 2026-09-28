"""
Recommendation Engine Module
Handles local similarity calculations, ranking, explainability, filtering, and movie search.
Supports sorting by Top Similar, Early First, Top Rated, and filtering by release year.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def recommend(
    movie_index: int,
    similarity_matrix: Any,
    movies_df: pd.DataFrame,
    n: int = 10,
    year_filter: str = "All Years",
    sort_type: str = "Top Similar",
    **kwargs: Any
) -> List[Dict[str, Any]]:
    """
    Computes top N recommendations for a given movie index with support for
    year filtering and custom sorting (Top Similar, Early First, Top Rated, Latest First).

    Args:
        movie_index (int): Integer row index of the target movie.
        similarity_matrix: 2D array-like pairwise similarity matrix.
        movies_df (pd.DataFrame): DataFrame containing movie records.
        n (int): Number of recommendations to return (default 10).
        year_filter (str): Filter by era ('All Years', '2010 and Newer', '2000 - 2009', '1990 - 1999', 'Before 1990').
        sort_type (str): Sorting mode ('Top Similar', 'Early First', 'Top Rated', 'Latest First').

    Returns:
        List[Dict[str, Any]]: Filtered and sorted recommendation items.
    """
    if movie_index < 0 or movie_index >= len(movies_df):
        raise IndexError(f"movie_index {movie_index} is out of bounds for dataset of size {len(movies_df)}")

    # Extract similarity row for the requested movie
    row_scores = similarity_matrix[movie_index]

    if hasattr(row_scores, "toarray"):
        row_scores = row_scores.toarray().flatten()
    elif isinstance(row_scores, np.matrix):
        row_scores = np.asarray(row_scores).flatten()

    # Sort all indices by descending similarity score: list of (index, score)
    sorted_pairs = sorted(enumerate(row_scores), reverse=True, key=lambda x: x[1])

    target_row = movies_df.iloc[movie_index]
    candidate_pool: List[Dict[str, Any]] = []

    # Gather a pool of top similar candidates (up to 200) to allow rich filtering & sorting
    pool_size = max(n * 10, 100)

    for cand_idx, score in sorted_pairs:
        if cand_idx == movie_index:
            continue

        cand_row = movies_df.iloc[cand_idx]
        score_val = float(score)

        # Release year
        raw_year = cand_row.get("release_year") if "release_year" in cand_row else None
        try:
            year_val = int(raw_year) if raw_year and not pd.isna(raw_year) else None
        except Exception:
            year_val = None

        # Year Filtering
        if year_filter == "2010 and Newer":
            if not year_val or year_val < 2010:
                continue
        elif year_filter == "2000 - 2009":
            if not year_val or not (2000 <= year_val <= 2009):
                continue
        elif year_filter == "1990 - 1999":
            if not year_val or not (1990 <= year_val <= 1999):
                continue
        elif year_filter == "Before 1990":
            if not year_val or year_val >= 1990:
                continue

        # Vote average
        raw_rating = cand_row.get("vote_average") if "vote_average" in cand_row else None
        try:
            rating_val = float(raw_rating) if raw_rating and not pd.isna(raw_rating) else 0.0
        except Exception:
            rating_val = 0.0

        raw_title = str(cand_row.get("title", ""))
        display_title = raw_title.title() if raw_title.islower() else raw_title

        # Explainability cues
        reasons = explain_recommendation(target_row, cand_row, score_val)

        candidate_pool.append({
            "index": cand_idx,
            "id": int(cand_row.get("id", 0)),
            "title": display_title,
            "similarity_score": round(score_val, 4),
            "similarity_pct": max(0, min(100, int(round(score_val * 100)))),
            "reasons": reasons,
            "release_year": year_val,
            "vote_average": rating_val,
            "overview": cand_row.get("overview") if "overview" in cand_row else None,
            "genres": cand_row.get("genres") if "genres" in cand_row else None,
            "director": cand_row.get("director") if "director" in cand_row else None,
            "cast": cand_row.get("cast") if "cast" in cand_row else None,
        })

        if len(candidate_pool) >= pool_size:
            break

    # Apply Custom Sorting
    if sort_type == "Early First":
        # Sort by release year ascending (oldest first). Unknown years placed at the end.
        candidate_pool.sort(key=lambda x: (x["release_year"] if x["release_year"] else 9999, -x["similarity_score"]))
    elif sort_type == "Top Rated":
        # Sort by rating descending, then similarity
        candidate_pool.sort(key=lambda x: (-x["vote_average"], -x["similarity_score"]))
    elif sort_type == "Latest First":
        # Sort by release year descending (newest first)
        candidate_pool.sort(key=lambda x: (-x["release_year"] if x["release_year"] else 0, -x["similarity_score"]))
    else:  # "Top Similar"
        candidate_pool.sort(key=lambda x: -x["similarity_score"])

    return candidate_pool[:n]


def search_movies(
    query: str,
    movies_df: pd.DataFrame,
    limit: int = 15
) -> List[Dict[str, Any]]:
    """
    Performs fast local search over movie titles without external network calls.
    Ranks prefix matches higher than substring matches.
    """
    clean_query = query.strip().lower()
    if not clean_query:
        return []

    prefix_matches = []
    contains_matches = []

    titles = movies_df["title"].values
    ids = movies_df["id"].values
    years = movies_df["release_year"].values if "release_year" in movies_df.columns else [None] * len(titles)

    for idx, (raw_title, m_id, year) in enumerate(zip(titles, ids, years)):
        title_lower = str(raw_title).lower()
        if title_lower.startswith(clean_query):
            display_title = str(raw_title).title() if str(raw_title).islower() else str(raw_title)
            prefix_matches.append({
                "index": idx,
                "id": int(m_id),
                "title": display_title,
                "year": int(year) if year and year > 0 else None,
            })
        elif clean_query in title_lower:
            display_title = str(raw_title).title() if str(raw_title).islower() else str(raw_title)
            contains_matches.append({
                "index": idx,
                "id": int(m_id),
                "title": display_title,
                "year": int(year) if year and year > 0 else None,
            })

        if len(prefix_matches) + len(contains_matches) >= limit * 2:
            break

    results = prefix_matches + contains_matches
    return results[:limit]


def explain_recommendation(
    target_row: Any,
    candidate_row: Any,
    similarity_score: float
) -> List[str]:
    """
    Generates transparent, explainable reasons why a movie was recommended
    based on shared directors, shared cast, shared genres, and cosine similarity.
    """
    reasons: List[str] = []

    # 1. Check shared director
    t_director = target_row.get("director") if hasattr(target_row, "get") else None
    c_director = candidate_row.get("director") if hasattr(candidate_row, "get") else None

    if isinstance(t_director, list) and isinstance(c_director, list):
        shared_directors = set(t_director).intersection(set(c_director))
        if shared_directors:
            reasons.append(f"Same Director ({', '.join(shared_directors)})")

    # 2. Check shared cast
    t_cast = target_row.get("cast") if hasattr(target_row, "get") else None
    c_cast = candidate_row.get("cast") if hasattr(candidate_row, "get") else None

    if isinstance(t_cast, list) and isinstance(c_cast, list):
        shared_cast = set(t_cast).intersection(set(c_cast))
        if shared_cast:
            reasons.append(f"Shared Star ({', '.join(shared_cast)})")

    # 3. Check shared genres
    t_genres = target_row.get("genres") if hasattr(target_row, "get") else None
    c_genres = candidate_row.get("genres") if hasattr(candidate_row, "get") else None

    if isinstance(t_genres, list) and isinstance(c_genres, list):
        shared_genres = [g for g in t_genres if g in c_genres]
        if shared_genres:
            reasons.append(f"Shared Genres: {', '.join(shared_genres[:3])}")

    # 4. Content similarity score indicator
    if similarity_score >= 0.4:
        reasons.append(f"High Plot & Theme Match ({int(round(similarity_score * 100))}% similarity)")
    elif similarity_score >= 0.2:
        reasons.append(f"Similar Story Elements ({int(round(similarity_score * 100))}% similarity)")
    else:
        reasons.append(f"Related Content ({int(round(similarity_score * 100))}% similarity)")

    return reasons
