"""
TMDB API Client Service
Handles fetching movie details, posters, backdrops, ratings, credits, and trailers from The Movie Database (TMDB).
Includes robust retry logic, connection handling, timeouts, and fallbacks.
"""

import logging
import time
from typing import Any, Dict, List, Optional
import requests

from recommender.config import get_tmdb_access_token

logger = logging.getLogger(__name__)

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p"

POSTER_SIZES = {"w92", "w154", "w185", "w342", "w500", "w780", "original"}
BACKDROP_SIZES = {"w300", "w780", "w1280", "original"}

REQUEST_TIMEOUT_SECONDS = 7.0
MAX_RETRIES = 3


def is_v3_api_key(token: str) -> bool:
    """Checks if token is a standard 32-character hexadecimal TMDB v3 API Key."""
    return bool(token and len(token) == 32 and all(c in "0123456789abcdefABCDEF" for c in token))


def get_tmdb_auth_params_and_headers(token: Optional[str] = None):
    """
    Constructs headers and query parameters based on whether the token
    is a v4 Bearer Read Access Token or a v3 API Key.
    """
    auth_token = token or get_tmdb_access_token()
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
        "Connection": "close",
    }
    params: Dict[str, str] = {}

    if auth_token:
        if is_v3_api_key(auth_token):
            params["api_key"] = auth_token
        else:
            headers["Authorization"] = f"Bearer {auth_token}"

    return headers, params


def get_poster_url(poster_path: Optional[str], size: str = "w500") -> Optional[str]:
    """
    Constructs a fully-qualified TMDB poster URL.
    Returns None if poster_path is empty or missing.
    """
    if not poster_path or not isinstance(poster_path, str) or not poster_path.strip():
        return None
    selected_size = size if size in POSTER_SIZES else "w500"
    path = poster_path.strip().lstrip("/")
    return f"{TMDB_IMAGE_BASE_URL}/{selected_size}/{path}"


def get_backdrop_url(backdrop_path: Optional[str], size: str = "w1280") -> Optional[str]:
    """
    Constructs a fully-qualified TMDB backdrop URL.
    Returns None if backdrop_path is empty or missing.
    """
    if not backdrop_path or not isinstance(backdrop_path, str) or not backdrop_path.strip():
        return None
    selected_size = size if size in BACKDROP_SIZES else "w1280"
    path = backdrop_path.strip().lstrip("/")
    return f"{TMDB_IMAGE_BASE_URL}/{selected_size}/{path}"


def extract_trailer_url(videos_data: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Extracts the most relevant YouTube trailer or teaser URL from TMDB videos data.
    """
    if not videos_data or not isinstance(videos_data, dict):
        return None

    results = videos_data.get("results", [])
    if not isinstance(results, list) or not results:
        return None

    # Priority 1: Official Trailer on YouTube
    for video in results:
        if (
            video.get("site") == "YouTube"
            and video.get("type") == "Trailer"
            and video.get("official") is True
            and video.get("key")
        ):
            return f"https://www.youtube.com/watch?v={video['key']}"

    # Priority 2: Any Trailer on YouTube
    for video in results:
        if (
            video.get("site") == "YouTube"
            and video.get("type") == "Trailer"
            and video.get("key")
        ):
            return f"https://www.youtube.com/watch?v={video['key']}"

    # Priority 3: Teaser on YouTube
    for video in results:
        if (
            video.get("site") == "YouTube"
            and video.get("type") == "Teaser"
            and video.get("key")
        ):
            return f"https://www.youtube.com/watch?v={video['key']}"

    # Priority 4: Clip or Featurette on YouTube
    for video in results:
        if (
            video.get("site") == "YouTube"
            and video.get("key")
        ):
            return f"https://www.youtube.com/watch?v={video['key']}"

    return None


def get_movie_details(
    movie_id: int,
    token: Optional[str] = None,
    append_to_response: str = "credits,videos"
) -> Optional[Dict[str, Any]]:
    """
    Fetches comprehensive movie details from TMDB with appended credits and videos.
    Includes an automatic retry mechanism with exponential backoff to handle
    transient Windows socket resets and rate limits.
    """
    auth_token = token or get_tmdb_access_token()
    if not auth_token or not movie_id:
        return None

    headers, base_params = get_tmdb_auth_params_and_headers(auth_token)
    url = f"{TMDB_BASE_URL}/movie/{movie_id}"
    params = dict(base_params)
    params["language"] = "en-US"
    if append_to_response:
        params["append_to_response"] = append_to_response

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS
            )
            if response.status_code == 200:
                data = response.json()
                # If appended videos didn't return any YouTube video, try fetching raw videos list without language filter
                if not extract_trailer_url(data.get("videos")):
                    raw_videos = get_movie_videos(movie_id, auth_token)
                    if raw_videos and "results" in raw_videos:
                        data["videos"] = raw_videos
                return data
            elif response.status_code == 404:
                logger.warning("Movie ID %s not found on TMDB (404).", movie_id)
                return None
            elif response.status_code in (429, 500, 502, 503, 504):
                time.sleep(0.3 * (attempt + 1))
                continue
            else:
                logger.warning("TMDB request returned status %s for movie %s", response.status_code, movie_id)
                return None
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(0.35 * (attempt + 1))
                continue
            logger.warning("TMDB connection error for movie ID %s after %d retries: %s", movie_id, MAX_RETRIES, exc)
            return None
        except Exception as exc:
            logger.warning("Unexpected error fetching TMDB movie %s: %s", movie_id, exc)
            return None

    return None


def get_movie_credits(movie_id: int, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetches standalone cast & crew credits from TMDB with retry."""
    auth_token = token or get_tmdb_access_token()
    if not auth_token or not movie_id:
        return None

    headers, params = get_tmdb_auth_params_and_headers(auth_token)
    url = f"{TMDB_BASE_URL}/movie/{movie_id}/credits"
    params["language"] = "en-US"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(0.3 * (attempt + 1))
                continue
    return None


def get_movie_videos(movie_id: int, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetches standalone videos/trailers from TMDB without strict language filter with retry."""
    auth_token = token or get_tmdb_access_token()
    if not auth_token or not movie_id:
        return None

    headers, params = get_tmdb_auth_params_and_headers(auth_token)
    url = f"{TMDB_BASE_URL}/movie/{movie_id}/videos"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(0.3 * (attempt + 1))
                continue
    return None
