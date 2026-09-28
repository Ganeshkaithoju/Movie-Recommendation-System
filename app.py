"""
CineMatch - Movie Discovery and Recommendation Platform
A content-first, cinematic movie recommendation system using NLP feature extraction
and live TMDB metadata enrichment.
"""

import io
import os
import pickle
import joblib
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
import importlib

import recommender.config
import recommender.tmdb
import recommender.engine

importlib.reload(recommender.config)
importlib.reload(recommender.tmdb)
importlib.reload(recommender.engine)

from recommender.config import is_tmdb_configured
from recommender.engine import recommend, search_movies
from recommender.tmdb import (
    get_movie_details,
    get_poster_url,
    get_backdrop_url,
    extract_trailer_url,
)

# Page Configuration (Zero emojis, clean title)
st.set_page_config(
    page_title="CineMatch - Movie Discovery",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Cinematic Editorial Styling
CUSTOM_CSS = """
<style>
/* Global Layout and Typography */
body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #E2E5E9;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1360px;
}

/* Header */
.app-header {
    background-color: #15181F;
    border: 1px solid #252A36;
    border-radius: 4px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
}
.app-header-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #F0F2F5;
    letter-spacing: 0.5px;
    margin: 0;
    text-transform: uppercase;
}
.app-header-subtitle {
    color: #8C93A3;
    font-size: 0.95rem;
    margin-top: 0.35rem;
    margin-bottom: 0;
}

/* Hero Section (Selected Movie) */
.hero-container {
    background-color: #15181F;
    border: 1px solid #262B37;
    border-radius: 4px;
    padding: 1.75rem;
    margin-bottom: 2.5rem;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.2;
    margin-bottom: 0.6rem;
    letter-spacing: -0.3px;
}
.hero-meta-row {
    margin-bottom: 1rem;
}
.meta-chip {
    display: inline-block;
    background-color: #1E232E;
    color: #C5CBD6;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.25rem 0.65rem;
    border: 1px solid #2E3545;
    border-radius: 3px;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}
.meta-chip-accent {
    display: inline-block;
    background-color: #271E20;
    color: #E86B6D;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.25rem 0.65rem;
    border: 1px solid #4A272B;
    border-radius: 3px;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}
.hero-overview {
    color: #C0C6D2;
    font-size: 0.98rem;
    line-height: 1.65;
    margin-top: 0.75rem;
    margin-bottom: 1.25rem;
}
.hero-credit {
    font-size: 0.88rem;
    color: #8C93A3;
    margin-bottom: 0.3rem;
}
.hero-credit-label {
    color: #C5CBD6;
    font-weight: 600;
}

/* Recommendation Cards & Alignment Styling */
.rec-card-info-box {
    min-height: 56px;
    margin-top: 0.35rem;
    margin-bottom: 0.35rem;
}
.rec-card-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: #EDF0F5;
    line-height: 1.25;
    margin-bottom: 0.25rem;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.rec-card-meta {
    font-size: 0.78rem;
    color: #8C93A3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.match-tag {
    background-color: #1B212B;
    color: #A4B1C5;
    border: 1px solid #2B3342;
    border-radius: 2px;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.1rem 0.4rem;
}

/* Legal and Attribution Footer */
.legal-footer {
    border-top: 1px solid #222632;
    padding-top: 2rem;
    margin-top: 3.5rem;
    font-size: 0.82rem;
    color: #717786;
    line-height: 1.6;
}
.legal-footer a {
    color: #9AA2B2;
    text-decoration: none;
}
.legal-footer a:hover {
    text-decoration: underline;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Fallback Poster Generator with Movie Name
@st.cache_data
def get_movie_title_poster_bytes(title: str, year: int = None) -> bytes:
    """
    Generates a dark cinematic poster image featuring the movie's title
    when no official TMDB poster artwork is available.
    """
    w, h = 400, 600
    img = Image.new("RGB", (w, h), color="#15181F")
    draw = ImageDraw.Draw(img)

    # Clean borders
    draw.rectangle([6, 6, w - 6, h - 6], outline="#282D3B", width=2)
    draw.rectangle([14, 14, w - 14, h - 14], outline="#1D222D", width=1)

    # Header label
    draw.text((w // 2, 45), "CATALOG SELECTION", fill="#5A6172", anchor="mm")

    # Wrap title into readable uppercase lines
    words = str(title).upper().split()
    lines, curr = [], []
    for word in words:
        curr.append(word)
        if len(" ".join(curr)) > 14:
            if len(curr) > 1:
                curr.pop()
                lines.append(" ".join(curr))
                curr = [word]
            else:
                lines.append(" ".join(curr))
                curr = []
    if curr:
        lines.append(" ".join(curr))

    total_lines = min(len(lines), 6)
    start_y = (h // 2) - (total_lines * 18)
    for i, line in enumerate(lines[:6]):
        draw.text((w // 2, start_y + (i * 36)), line, fill="#ECEFF4", anchor="mm")

    if year and year > 0:
        draw.text((w // 2, h - 45), f"YEAR {year}", fill="#757D8E", anchor="mm")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Resource & Data Caching
@st.cache_resource(show_spinner="Loading movie collection...")
def load_data_and_similarity():
    """Loads precomputed movie records and cosine similarity matrix once."""
    data_path = "data/movies.pickle" if os.path.exists("data/movies.pickle") else "movies.pickle"
    sim_path = "data/similarity.joblib" if os.path.exists("data/similarity.joblib") else "similarity.joblib"

    if not os.path.exists(data_path) or not os.path.exists(sim_path):
        st.error(f"Required artifact files missing: {data_path} or {sim_path}")
        st.stop()

    with open(data_path, "rb") as f:
        movies_df = pickle.load(f)

    similarity_matrix = joblib.load(sim_path)
    return movies_df, similarity_matrix


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_enriched_tmdb_data(movie_id: int):
    """Caches TMDB API responses for 24 hours per movie ID."""
    return get_movie_details(movie_id, append_to_response="credits,videos")


# Load Model and Metadata
movies_df, similarity_matrix = load_data_and_similarity()
tmdb_active = is_tmdb_configured()

# State Management: Selected Movie Index
if "selected_movie_index" not in st.session_state:
    st.session_state.selected_movie_index = 0


def select_movie(index: int):
    st.session_state.selected_movie_index = index


# Sidebar: Navigation, Catalog Search, and Recommendation Settings
with st.sidebar:
    st.markdown("### Movie Search")
    st.markdown("Browse 4,799 films from the TMDb collection.")

    search_query = st.text_input(
        "Search by Title",
        value="",
        placeholder="Type a movie title (e.g. Interstellar, Avatar)...",
        label_visibility="collapsed",
    )

    if search_query.strip():
        search_results = search_movies(search_query, movies_df, limit=20)
        if search_results:
            options_labels = [
                f"{item['title']} ({item['year']})" if item.get("year") else item["title"]
                for item in search_results
            ]
            selected_label = st.selectbox("Matching Titles", options_labels, index=0)
            chosen_match = search_results[options_labels.index(selected_label)]
            if st.button("Select Movie", width="stretch", type="primary"):
                select_movie(chosen_match["index"])
                st.rerun()
        else:
            st.info("No matching titles found in the local index.")
    else:
        # Quick browse dropdown
        chosen_browse = st.selectbox(
            "Browse Catalog Titles",
            range(min(500, len(movies_df))),
            format_func=lambda i: (
                f"{movies_df.iloc[i]['title'].title() if str(movies_df.iloc[i]['title']).islower() else movies_df.iloc[i]['title']}"
                + (f" ({int(movies_df.iloc[i]['release_year'])})" if "release_year" in movies_df.iloc[i] and movies_df.iloc[i]['release_year'] else "")
            ),
            index=st.session_state.selected_movie_index if st.session_state.selected_movie_index < 500 else 0,
        )
        if chosen_browse != st.session_state.selected_movie_index:
            if st.button("Load Selection", width="stretch"):
                select_movie(chosen_browse)
                st.rerun()

    st.markdown("---")
    st.markdown("### Recommendation Settings")

    rec_count = st.selectbox(
        "Number of Recommendations",
        options=[5, 10, 15, 20],
        index=1,
    )

    selected_year = st.selectbox(
        "Select Year",
        options=["All Years", "2010 and Newer", "2000 - 2009", "1990 - 1999", "Before 1990"],
        index=0,
    )

    selected_type = st.selectbox(
        "Select Type (Sort)",
        options=["Top Similar", "Early First", "Top Rated", "Latest First"],
        index=0,
    )

    st.markdown("---")
    # Clean Factual Status
    if tmdb_active:
        st.caption("TMDB Service: Connected (Live Posters and Trailers Active)")
    else:
        st.caption("TMDB Service: Local Mode (Built-in Metadata Active)")


# Application Header
st.markdown(
    """
    <div class="app-header">
        <h1 class="app-header-title">CineMatch Movie Discovery</h1>
        <p class="app-header-subtitle">Content-based movie recommendation powered by NLP metadata vectors and TMDB catalog data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Selected Movie Data Resolution
current_index = st.session_state.selected_movie_index
local_row = movies_df.iloc[current_index]
movie_id = int(local_row.get("id", 0))

# Retrieve TMDB data if token is configured
tmdb_data = fetch_enriched_tmdb_data(movie_id) if tmdb_active else None

raw_local_title = str(local_row.get("title", ""))
display_title = (
    tmdb_data.get("title")
    if (tmdb_data and tmdb_data.get("title"))
    else (raw_local_title.title() if raw_local_title.islower() else raw_local_title)
)

poster_url = None
if tmdb_data and tmdb_data.get("poster_path"):
    poster_url = get_poster_url(tmdb_data["poster_path"], size="w500")

vote_avg = (
    tmdb_data.get("vote_average")
    if (tmdb_data and tmdb_data.get("vote_average"))
    else local_row.get("vote_average", 0.0)
)
runtime = tmdb_data.get("runtime") if tmdb_data else None
release_year = (
    tmdb_data.get("release_date", "")[:4]
    if (tmdb_data and tmdb_data.get("release_date"))
    else (
        str(int(local_row["release_year"]))
        if ("release_year" in local_row and local_row["release_year"])
        else ""
    )
)

# Extract Genres
genres_list = []
if tmdb_data and tmdb_data.get("genres"):
    genres_list = [g["name"] for g in tmdb_data["genres"] if isinstance(g, dict) and "name" in g]
elif "genres" in local_row and isinstance(local_row["genres"], list):
    genres_list = local_row["genres"]

# Extract Director and Cast
director_names = []
cast_names = []
if tmdb_data and tmdb_data.get("credits"):
    crew = tmdb_data["credits"].get("crew", [])
    director_names = [c["name"] for c in crew if c.get("job") == "Director"]
    cast_list = tmdb_data["credits"].get("cast", [])
    cast_names = [c["name"] for c in cast_list[:4] if "name" in c]
else:
    if "director" in local_row and isinstance(local_row["director"], list):
        director_names = local_row["director"]
    if "cast" in local_row and isinstance(local_row["cast"], list):
        cast_names = local_row["cast"]

# Extract Synopsis
overview = (
    tmdb_data.get("overview")
    if (tmdb_data and tmdb_data.get("overview"))
    else local_row.get("overview", "No synopsis currently available for this title.")
)

# Extract Trailer
trailer_url = extract_trailer_url(tmdb_data.get("videos")) if tmdb_data else None


# Selected Movie Section (Hero)
st.markdown("### Selected Movie")

with st.container():
    col_poster, col_details = st.columns([1, 2.5], gap="large")

    with col_poster:
        if poster_url:
            st.image(poster_url, width="stretch")
        else:
            # Show movie name in poster card when artwork is unavailable
            year_int = int(release_year) if release_year and release_year.isdigit() else None
            hero_poster_bytes = get_movie_title_poster_bytes(display_title, year_int)
            st.image(hero_poster_bytes, width="stretch")

        if trailer_url:
            st.link_button("Watch Official Trailer", trailer_url, width="stretch")
        else:
            st.button("Trailer Unavailable", key="hero_no_tr", disabled=True, width="stretch")

    with col_details:
        st.markdown(f'<div class="hero-title">{display_title}</div>', unsafe_allow_html=True)

        # Metadata Row
        meta_html = ""
        if vote_avg and float(vote_avg) > 0:
            meta_html += f'<span class="meta-chip-accent">Rating {float(vote_avg):.1f} / 10</span>'
        if release_year:
            meta_html += f'<span class="meta-chip">Year {release_year}</span>'
        if runtime and runtime > 0:
            meta_html += f'<span class="meta-chip">Runtime {runtime} min</span>'
        if meta_html:
            st.markdown(f'<div class="hero-meta-row">{meta_html}</div>', unsafe_allow_html=True)

        # Genre tags
        if genres_list:
            genre_html = "".join([f'<span class="meta-chip">{g}</span>' for g in genres_list[:6]])
            st.markdown(f'<div class="hero-meta-row">{genre_html}</div>', unsafe_allow_html=True)

        # Overview synopsis
        st.markdown(f'<div class="hero-overview">{overview}</div>', unsafe_allow_html=True)

        # Credits
        if director_names:
            st.markdown(
                f'<div class="hero-credit"><span class="hero-credit-label">Director:</span> {", ".join(director_names[:2])}</div>',
                unsafe_allow_html=True,
            )
        if cast_names:
            st.markdown(
                f'<div class="hero-credit"><span class="hero-credit-label">Cast:</span> {", ".join(cast_names[:4])}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        btn_col, _ = st.columns([1.5, 2])
        with btn_col:
            st.button("Find Similar Movies", type="primary", width="stretch")


# Recommendations Section
st.markdown("---")
subhead_meta = f"Top {rec_count}"
if selected_year != "All Years":
    subhead_meta += f" | {selected_year}"
if selected_type != "Top Similar":
    subhead_meta += f" | {selected_type}"

st.markdown(f"### Recommended Movies ({subhead_meta})")

# Execute recommendation engine
try:
    raw_recommendations = recommend(
        movie_index=current_index,
        similarity_matrix=similarity_matrix,
        movies_df=movies_df,
        n=rec_count,
        year_filter=selected_year,
        sort_type=selected_type,
    )
except TypeError:
    raw_recommendations = recommend(
        movie_index=current_index,
        similarity_matrix=similarity_matrix,
        movies_df=movies_df,
        n=rec_count,
    )

if not raw_recommendations:
    st.info(f"No titles match the year filter '{selected_year}'. Please set 'Select Year' to 'All Years' to view similar titles.")
else:
    cols_per_row = 5

    # Process in batches of 5 to guarantee identical horizontal alignment across columns
    for row_start in range(0, len(raw_recommendations), cols_per_row):
        row_items = raw_recommendations[row_start : row_start + cols_per_row]
        batch_size = len(row_items)

        # Pre-enrich TMDB data for all movies in this batch
        batch_enriched = []
        for rec in row_items:
            rec_id = rec["id"]
            rec_tmdb = fetch_enriched_tmdb_data(rec_id) if tmdb_active else None

            rec_poster_url = None
            rec_rating_str = ""
            rec_year_str = ""
            rec_trailer_url = None

            if rec_tmdb:
                if rec_tmdb.get("poster_path"):
                    rec_poster_url = get_poster_url(rec_tmdb["poster_path"], size="w342")
                if rec_tmdb.get("vote_average") and float(rec_tmdb["vote_average"]) > 0:
                    rec_rating_str = f"Rating {float(rec_tmdb['vote_average']):.1f}"
                if rec_tmdb.get("release_date"):
                    rec_year_str = rec_tmdb["release_date"][:4]
                rec_trailer_url = extract_trailer_url(rec_tmdb.get("videos"))
            else:
                if rec.get("vote_average") and float(rec["vote_average"]) > 0:
                    rec_rating_str = f"Rating {float(rec['vote_average']):.1f}"
                if rec.get("release_year"):
                    rec_year_str = str(int(rec["release_year"]))

            batch_enriched.append({
                "rec": rec,
                "poster_url": rec_poster_url,
                "rating_str": rec_rating_str,
                "year_str": rec_year_str,
                "trailer_url": rec_trailer_url,
            })

        # 1. Posters Row (All 5 posters perfectly aligned side-by-side)
        cols_posters = st.columns(cols_per_row, gap="small")
        for i, item in enumerate(batch_enriched):
            with cols_posters[i]:
                rec = item["rec"]
                poster_url = item["poster_url"]
                if poster_url:
                    st.image(poster_url, width="stretch")
                else:
                    # When poster is unavailable, show the movie name in the movie card as a poster
                    year_val = int(rec["release_year"]) if rec.get("release_year") else None
                    title_poster_bytes = get_movie_title_poster_bytes(rec["title"], year_val)
                    st.image(title_poster_bytes, width="stretch")

        # 2. Movie Title & Metadata Row (Aligned with uniform min-height box)
        cols_info = st.columns(cols_per_row, gap="small")
        for i, item in enumerate(batch_enriched):
            with cols_info[i]:
                rec = item["rec"]
                meta_line = []
                if item["rating_str"]:
                    meta_line.append(item["rating_str"])
                if item["year_str"]:
                    meta_line.append(item["year_str"])
                meta_line.append(f"<span class='match-tag'>Match {rec['similarity_pct']}%</span>")
                meta_str = " | ".join(meta_line)

                st.markdown(
                    f"""
                    <div class="rec-card-info-box">
                        <div class="rec-card-title">{rec['title']}</div>
                        <div class="rec-card-meta">{meta_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # 3. Trailers Row (Side-by-side: Watch Trailer link or Trailer Unavailable button)
        cols_trailers = st.columns(cols_per_row, gap="small")
        for i, item in enumerate(batch_enriched):
            with cols_trailers[i]:
                rec = item["rec"]
                trailer_url = item["trailer_url"]
                if trailer_url:
                    st.link_button("Watch Trailer", trailer_url, width="stretch")
                else:
                    st.button(
                        "Trailer Unavailable",
                        key=f"no_tr_{rec['index']}",
                        disabled=True,
                        width="stretch",
                    )

        # 4. Recommendation Factors Row (Side-by-side: All expanders horizontally aligned)
        cols_factors = st.columns(cols_per_row, gap="small")
        for i, item in enumerate(batch_enriched):
            with cols_factors[i]:
                rec = item["rec"]
                with st.expander("Match Factors"):
                    if rec.get("reasons"):
                        for reason in rec["reasons"]:
                            st.caption(f"- {reason}")
                    else:
                        st.caption("Content similarity based on plot and genre vectors.")

        # 5. Explore Movie Row (Side-by-side: All Explore buttons perfectly aligned in same line)
        cols_explore = st.columns(cols_per_row, gap="small")
        for i, item in enumerate(batch_enriched):
            with cols_explore[i]:
                rec = item["rec"]
                if st.button("Explore Movie", key=f"rec_btn_{rec['index']}", width="stretch"):
                    select_movie(rec["index"])
                    st.rerun()

        # Spacing between batches of 5
        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)


# Legal and Attribution Footer
st.markdown("---")
st.markdown("### Information and Policies")

tab_about, tab_terms, tab_privacy = st.tabs(["About and Attribution", "Terms of Service", "Privacy Policy"])

with tab_about:
    st.markdown(
        """
        **CineMatch Movie Recommendation System**
        
        This application provides content-based movie recommendations by evaluating semantic and categorical similarity across plot summaries, genres, keywords, cast members, and directors.
        
        - **Data Attribution:** Movie metadata, posters, and video trailers are provided courtesy of [The Movie Database (TMDB)](https://www.themoviedb.org/).
        - **Notice:** This product uses the TMDB API but is not endorsed or certified by TMDB.
        - **Pipeline:** Preprocessing utilizes NLTK Porter Stemming, Scikit-learn CountVectorizer representation, and Cosine Similarity metric calculation.
        """
    )

with tab_terms:
    st.markdown(
        """
        **Terms of Service**
        
        1. **Acceptance of Terms:** By using this application, you agree to these Terms of Service. This application is intended for personal, non-commercial informational and educational exploration.
        2. **Intellectual Property:** All movie titles, descriptions, imagery, posters, and trademarks displayed within the application remain the property of their respective copyright holders and studio distributors. Metadata is obtained via the TMDB API under their standard API terms.
        3. **No Warranty:** The recommendation system and its content are provided on an "as is" and "as available" basis without warranties of any kind regarding accuracy, uninterrupted availability, or completeness of third-party metadata.
        4. **Production Notice:** This template text is provided for educational and portfolio demonstration. Before production or commercial deployment, these terms should be reviewed by legal counsel.
        """
    )

with tab_privacy:
    st.markdown(
        """
        **Privacy Policy**
        
        1. **Data Collection:** This application does not collect, record, or sell any personally identifiable information (PII). No user accounts or personal profiles are created.
        2. **Search Queries:** Title searches and recommendation filtering run locally in your active session memory. Search inputs are not stored in any external database or analytics repository.
        3. **External API Requests:** When TMDB integration is active, movie identifiers are transmitted directly to TMDB servers via encrypted HTTPS to fetch posters, details, and video links. Refer to [TMDB Privacy Policy](https://www.themoviedb.org/privacy-policy) for details on their data handling.
        4. **Cookies and Tracking:** This application does not utilize tracking cookies, advertising trackers, or third-party behavioral analytics.
        """
    )
