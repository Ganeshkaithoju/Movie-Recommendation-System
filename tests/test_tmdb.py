"""
Tests for TMDB API integration, fallbacks, image URL builders, and trailer parsing.
All external network calls are mocked; no real API requests are made.
"""

from unittest.mock import MagicMock, patch
import requests
from recommender.tmdb import (
    get_poster_url,
    get_backdrop_url,
    extract_trailer_url,
    get_movie_details,
    get_movie_credits,
    get_movie_videos,
)


def test_get_poster_url_valid_and_invalid():
    assert get_poster_url("/sample.jpg", size="w500") == "https://image.tmdb.org/t/p/w500/sample.jpg"
    assert get_poster_url("sample.jpg", size="w780") == "https://image.tmdb.org/t/p/w780/sample.jpg"
    # Fallback size if invalid
    assert get_poster_url("/sample.jpg", size="invalid_size") == "https://image.tmdb.org/t/p/w500/sample.jpg"
    # None on missing or empty
    assert get_poster_url(None) is None
    assert get_poster_url("") is None
    assert get_poster_url("   ") is None


def test_get_backdrop_url_valid_and_invalid():
    assert get_backdrop_url("/bg.jpg", size="w1280") == "https://image.tmdb.org/t/p/w1280/bg.jpg"
    assert get_backdrop_url(None) is None
    assert get_backdrop_url("") is None


def test_extract_trailer_url_prefers_official_trailer():
    videos = {
        "results": [
            {"site": "YouTube", "type": "Teaser", "key": "teaser123", "official": False},
            {"site": "YouTube", "type": "Trailer", "key": "official_trailer456", "official": True},
            {"site": "Vimeo", "type": "Trailer", "key": "vimeo789", "official": True},
        ]
    }
    trailer = extract_trailer_url(videos)
    assert trailer == "https://www.youtube.com/watch?v=official_trailer456"


def test_extract_trailer_url_handles_empty_or_malformed():
    assert extract_trailer_url(None) is None
    assert extract_trailer_url({}) is None
    assert extract_trailer_url({"results": []}) is None


@patch("requests.get")
def test_get_movie_details_success(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": 19995,
        "title": "Avatar",
        "vote_average": 7.5,
        "poster_path": "/avatar.jpg",
    }
    mock_get.return_value = mock_resp

    details = get_movie_details(19995, token="fake_token")
    assert details is not None
    assert details["id"] == 19995
    assert details["title"] == "Avatar"


@patch("requests.get")
def test_get_movie_details_404_handled_gracefully(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_get.return_value = mock_resp

    details = get_movie_details(99999999, token="fake_token")
    assert details is None


@patch("requests.get")
def test_get_movie_details_timeout_handled_gracefully(mock_get):
    mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

    details = get_movie_details(19995, token="fake_token")
    assert details is None


@patch("requests.get")
def test_get_movie_details_network_error_handled_gracefully(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError("Network down")

    details = get_movie_details(19995, token="fake_token")
    assert details is None


def test_get_movie_details_no_token():
    with patch("recommender.tmdb.get_tmdb_access_token", return_value=None):
        details = get_movie_details(19995, token=None)
        assert details is None
