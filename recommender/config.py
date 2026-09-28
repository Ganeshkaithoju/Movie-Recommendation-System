"""
Configuration management for the Movie Recommendation System.
Safely retrieves configuration and secrets without exposing them.
"""

import os
from typing import Optional


def get_tmdb_access_token() -> Optional[str]:
    """
    Safely retrieves the TMDB Read Access Token from:
    1. Streamlit secrets (st.secrets["TMDB_ACCESS_TOKEN"])
    2. Environment variables (os.environ["TMDB_ACCESS_TOKEN"])
    3. Local .env file (if present)

    Returns:
        Optional[str]: The token string if found, otherwise None.
    """
    token = None

    # 1. Check Streamlit secrets first (if running in Streamlit)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "TMDB_ACCESS_TOKEN" in st.secrets:
            token = str(st.secrets["TMDB_ACCESS_TOKEN"]).strip()
    except Exception:
        pass

    # 2. Check environment variable
    if not token:
        token = os.environ.get("TMDB_ACCESS_TOKEN", "").strip() or None

    # 3. Check local .env file manually if not in env
    if not token and os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        if key.strip() == "TMDB_ACCESS_TOKEN":
                            token = val.strip().strip("\"'")
                            break
        except Exception:
            pass

    return token if token else None


def is_tmdb_configured() -> bool:
    """Returns True if a valid TMDB token is detected."""
    token = get_tmdb_access_token()
    return bool(token and len(token) > 10)
