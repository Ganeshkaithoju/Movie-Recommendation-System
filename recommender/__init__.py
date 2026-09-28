"""
Recommender package exports.
"""

from recommender.engine import recommend, search_movies, explain_recommendation
from recommender.tmdb import (
    get_movie_details,
    get_movie_credits,
    get_movie_videos,
    get_poster_url,
    get_backdrop_url,
    extract_trailer_url,
)
from recommender.config import get_tmdb_access_token, is_tmdb_configured
from recommender.preprocessing import (
    enrich_dataset_metadata,
    parse_json_names,
    parse_top_cast,
    parse_director,
    stem_text,
    collapse_entities,
)

__all__ = [
    "recommend",
    "search_movies",
    "explain_recommendation",
    "get_movie_details",
    "get_movie_credits",
    "get_movie_videos",
    "get_poster_url",
    "get_backdrop_url",
    "extract_trailer_url",
    "get_tmdb_access_token",
    "is_tmdb_configured",
    "enrich_dataset_metadata",
    "parse_json_names",
    "parse_top_cast",
    "parse_director",
    "stem_text",
    "collapse_entities",
]
