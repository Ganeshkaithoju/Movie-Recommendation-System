"""
NLP and Data Preprocessing Pipeline for the Movie Recommendation System.
Documents and provides reusable transformation routines from the initial TMDB 5000 dataset.
"""

import ast
import logging
from typing import Any, List, Optional
import pandas as pd
from nltk.stem import PorterStemmer

logger = logging.getLogger(__name__)

# Initialize single stemmer instance
_stemmer = PorterStemmer()


def parse_json_names(json_str: Any) -> List[str]:
    """Extracts 'name' values from a stringified JSON array."""
    if not isinstance(json_str, str) or not json_str.strip():
        return []
    try:
        items = ast.literal_eval(json_str)
        if isinstance(items, list):
            return [item["name"] for item in items if isinstance(item, dict) and "name" in item]
    except Exception:
        pass
    return []


def parse_top_cast(json_str: Any, top_n: int = 3) -> List[str]:
    """Extracts top N actors/actresses from stringified JSON cast array."""
    if not isinstance(json_str, str) or not json_str.strip():
        return []
    try:
        items = ast.literal_eval(json_str)
        if isinstance(items, list):
            return [
                item["name"]
                for idx, item in enumerate(items)
                if isinstance(item, dict) and "name" in item and idx < top_n
            ]
    except Exception:
        pass
    return []


def parse_director(json_str: Any) -> List[str]:
    """Extracts Director name(s) from stringified JSON crew array."""
    if not isinstance(json_str, str) or not json_str.strip():
        return []
    try:
        items = ast.literal_eval(json_str)
        if isinstance(items, list):
            return [
                item["name"]
                for item in items
                if isinstance(item, dict) and item.get("job") == "Director" and "name" in item
            ]
    except Exception:
        pass
    return []


def collapse_entities(entity_list: List[str]) -> List[str]:
    """
    Removes whitespace from entity names (e.g., 'Sam Worthington' -> 'SamWorthington').
    Prevents tokenizers from mistakenly separating multi-word entities.
    """
    if not isinstance(entity_list, list):
        return []
    return [str(e).replace(" ", "") for e in entity_list if e]


def stem_text(text: str) -> str:
    """
    Applies PorterStemmer to space-separated text tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    stemmed_words = [_stemmer.stem(word) for word in text.split()]
    return " ".join(stemmed_words)


def enrich_dataset_metadata(
    movies_df: pd.DataFrame,
    raw_movies_path: str,
    raw_credits_path: str
) -> pd.DataFrame:
    """
    Safely enriches the existing 4,799-row movies dataframe with original
    rich metadata (genres, cast, director, release_year, overview) without
    modifying row ordering or breaking existing columns (id, title, tags).
    """
    try:
        raw_m = pd.read_csv(raw_movies_path)
        raw_c = pd.read_csv(raw_credits_path)
        merged = pd.concat([raw_m, raw_c.rename(columns={"title": "Title"})], axis=1)
        merged = merged[["genres", "id", "keywords", "overview", "release_date", "title", "cast", "crew"]]
        merged.dropna(inplace=True)

        if len(merged) == len(movies_df) and (merged["id"].values == movies_df["id"].values).all():
            enriched = movies_df.copy()
            enriched["release_year"] = (
                pd.to_datetime(merged["release_date"], errors="coerce")
                .dt.year.fillna(0)
                .astype(int)
                .values
            )
            enriched["genres"] = merged["genres"].apply(parse_json_names).values
            enriched["cast"] = merged["cast"].apply(lambda x: parse_top_cast(x, 3)).values
            enriched["director"] = merged["crew"].apply(parse_director).values
            enriched["overview"] = merged["overview"].values
            return enriched
    except Exception as exc:
        logger.warning("Could not enrich dataframe from raw CSVs: %s", exc)

    return movies_df
