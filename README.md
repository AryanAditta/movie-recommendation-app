# Movie Recommendation App

A Python-based movie recommendation project that explores multiple recommendation approaches through a simple API and web interface.

The deployed recommendation service uses **TF-IDF vectorization and cosine similarity** on movie plot descriptions to identify movies with similar content. The repository also contains an experimental recommendation script that explores **content-based and collaborative-filtering approaches** using movie metadata and user-rating data.

## Features

- Content-based movie recommendations using TF-IDF and cosine similarity
- FastAPI backend for serving recommendations
- Streamlit-based user interface
- Movie title matching and recommendation lookup
- Experimental collaborative-filtering and neural-network implementation
- Uses movie metadata and rating datasets for recommendation experiments

## Tech Stack

- Python
- Pandas
- Scikit-learn
- FastAPI
- Streamlit
- TensorFlow / Keras
- NumPy
- FuzzyWuzzy

## Project Structure

- `app.py` — FastAPI-based content recommendation service
- `streamlit_ui.py` — Streamlit frontend
- `Movie Recommendation System 2.py` — experimental hybrid/content-collaborative recommendation implementation
- `movies.csv` — movie metadata
- `ratings.csv` — user-rating data
- `requirements.txt` — Python dependencies

## Purpose

This project was developed as a practical exploration of recommendation systems, similarity-based retrieval, API development, and basic machine-learning-based personalization.
