# Movie Recommendation App

**Python recommendation-system project with a FastAPI backend, Streamlit interface and multiple recommendation experiments.**

The main service uses **TF-IDF vectorization and cosine similarity** over movie plot descriptions to recommend content-similar titles. The repository also includes an experimental script exploring metadata-based and collaborative-filtering approaches.

## Main Features

- Content-based recommendations using TF-IDF and cosine similarity.
- FastAPI endpoint for recommendation requests.
- Streamlit user interface.
- Experimental collaborative-filtering and neural-network workflow.
- Movie metadata and user-rating datasets for experimentation.

## Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `FastAPI` · `Streamlit` · `TensorFlow/Keras` · `NumPy`

## Repository Structure

- [`app.py`](app.py) — FastAPI content-recommendation service.
- [`streamlit_ui.py`](streamlit_ui.py) — Streamlit frontend.
- [`Movie Recommendation System 2.py`](Movie%20Recommendation%20System%202.py) — experimental recommendation workflow.
- [`movies.csv`](movies.csv) — movie metadata used by the experimental workflow.
- [`ratings.csv`](ratings.csv) — user-rating data used by the experimental workflow.
- [`requirements.txt`](requirements.txt) — Python dependencies.
- [`.env.example`](.env.example) — example environment-variable configuration.

## Setup

```bash
git clone https://github.com/AryanAditta/movie-recommendation-app.git
cd movie-recommendation-app
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

The FastAPI service fetches a small curated movie set from OMDb the first time it runs. Set your own OMDb API key through an environment variable:

```bash
export OMDB_API_KEY="your_key_here"
```

On Windows PowerShell:

```powershell
$env:OMDB_API_KEY="your_key_here"
```

Then start the API:

```bash
uvicorn app:app --reload
```

The recommendation endpoint is available at:

```text
/recommend/?title=Inception&n=5
```

To run the Streamlit interface:

```bash
streamlit run streamlit_ui.py
```

## Project Purpose

This project was developed as a practical exploration of recommendation systems, similarity-based retrieval, API development and basic ML-based personalization. It is retained as part of my broader software and machine-learning portfolio rather than my current primary research direction.

[GitHub Profile](https://github.com/AryanAditta) · [EDA Academic Profile](https://aryanaditta.github.io/eda-academic/)
