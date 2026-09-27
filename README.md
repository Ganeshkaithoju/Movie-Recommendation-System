# Movie Recommendation System

An NLP-based Movie Recommendation System that recommends movies based on their similarity to a selected movie.

## About

This project uses a movie dataset available on **Kaggle**. The system processes movie information such as genres, keywords, overview, cast, and director to create meaningful movie features.

Natural Language Processing (NLP) techniques are used to transform the movie information into a format that can be compared for similarity.

## Features

- Movie-based recommendations
- NLP-based text preprocessing
- Feature extraction from movie metadata
- Cosine similarity for finding similar movies
- Simple recommendation interface using Streamlit

## Dataset

The project uses a movie dataset obtained from **Kaggle** containing information about thousands of movies, including:

- Genres
- Keywords
- Overview
- Cast
- Director
- Release information

## Technologies Used

- Python
- Pandas
- NumPy
- NLP
- Scikit-learn
- Cosine Similarity
- Streamlit

## Project Structure

```text
Movie Recommendation/
│
├── app.py
├── movies.pickle
├── similarity.joblib
├── requirements.txt
└── README.md
