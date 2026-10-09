from fastapi import FastAPI
import pandas as pd
import requests
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

API_KEY = os.getenv("OMDB_API_KEY")
MOVIE_FILE = "movies_dataset.csv"  # Local cache generated from OMDb data

app = FastAPI()

# Function to fetch movie details from OMDb API
def get_movie_data(title):
    if not API_KEY:
        raise RuntimeError(
            "OMDB_API_KEY is not set. Export your OMDb API key before fetching movie data."
        )

    url = f"https://www.omdbapi.com/?t={title}&apikey={API_KEY}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    if data.get("Response") == "True":
        return {
            "title": data["Title"],
            "genre": data["Genre"],
            "plot": data["Plot"]
        }
    return None

# Function to fetch multiple movies (only run once)
def fetch_movies():
    movie_titles = [
        "Inception", "The Matrix", "Interstellar", "The Dark Knight", "Titanic", "Avatar",
        "Pulp Fiction", "Forrest Gump", "Fight Club", "The Shawshank Redemption", "Gladiator",
        "The Godfather", "The Lord of the Rings", "The Avengers", "Iron Man", "Spider-Man",
        "Doctor Strange", "The Social Network", "Joker", "Deadpool", "Black Panther", "Thor",
        "Captain America", "Guardians of the Galaxy", "Harry Potter", "The Revenant",
        "The Wolf of Wall Street", "Mad Max: Fury Road", "Gravity", "Django Unchained",
        "The Irishman", "Bohemian Rhapsody", "The Grand Budapest Hotel", "La La Land",
        "Shutter Island", "The Departed", "Whiplash", "A Beautiful Mind", "The Prestige",
        "Schindler's List", "Goodfellas", "No Country for Old Men", "Casino Royale",
        "The Green Mile", "Saving Private Ryan", "Inglourious Basterds", "Logan",
        "The Lion King", "Zodiac", "Blade Runner 2049", "Inside Out", "Coco", "Up",
        "Toy Story", "Ratatouille", "Wall-E", "The Incredibles", "Finding Nemo"
    ]

    movies_data = []
    for title in movie_titles:
        movie = get_movie_data(title)
        if movie:
            movies_data.append(movie)
        time.sleep(1)  # Avoid hitting API rate limits

    movies_df = pd.DataFrame(movies_data)
    movies_df.to_csv(MOVIE_FILE, index=False)
    return movies_df

# Load dataset or fetch if not available
if os.path.exists(MOVIE_FILE):
    print("Loading movies from CSV...")
    movies_df = pd.read_csv(MOVIE_FILE)
else:
    print("Fetching movies from API...")
    movies_df = fetch_movies()

movies_df["plot"] = movies_df["plot"].fillna("")

print("Computing similarity matrix...")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["plot"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend_movie(title, n=5):
    idx = movies_df[movies_df["title"] == title].index
    if len(idx) == 0:
        return {"error": "Movie not found in dataset."}

    idx = idx[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n + 1]
    recommended_movies = [movies_df.iloc[i[0]]["title"] for i in scores]

    return {"recommended_movies": recommended_movies}


@app.get("/recommend/")
def get_recommendations(title: str, n: int = 5):
    return recommend_movie(title, n)


@app.get("/")
def root():
    return {
        "message": "Welcome to the Movie Recommendation API! Use /recommend/?title=MovieName&n=5 to get recommendations."
    }
