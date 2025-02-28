import pandas as pd
import numpy as np
from fuzzywuzzy import process
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Flatten, Dense, Input, Dot
from sklearn.preprocessing import LabelEncoder



# 🔹 Load Movie Metadata (Content-Based)
movies = pd.read_csv("movies.csv")  # Contains movieId, title, genres
ratings = pd.read_csv("ratings.csv")  # Contains userId, movieId, rating

# 🔹 Ensure genres are not NaN
movies['genres'] = movies['genres'].fillna('')

# 🔹 Normalize movieId and userId so they start from 0
movies["movieId"] = movies["movieId"].astype("category").cat.codes
ratings["movieId"] = ratings["movieId"].astype("category").cat.codes
ratings["userId"] = ratings["userId"].astype("category").cat.codes

# 🔹 Get the number of unique users and movies
num_users = ratings["userId"].nunique()
num_movies = movies["movieId"].nunique()

print(f"Total Users: {num_users}, Total Movies: {num_movies}")

# 🔹 Convert movie genres into numerical encoding
encoder = LabelEncoder()
movies["genres_encoded"] = encoder.fit_transform(movies["genres"])

# 🔹 Define Content-Based Filtering Model
model_content = Sequential([
    Embedding(input_dim=num_movies, output_dim=50, input_length=1),
    Flatten(),
    Dense(50, activation="relu"),
    Dense(1, activation="linear")
])

model_content.compile(loss="mse", optimizer="adam")

# 🔹 Define Collaborative Filtering Model
user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))

user_embedding = Embedding(num_users, 50)(user_input)
movie_embedding = Embedding(num_movies, 50)(movie_input)

dot_product = Dot(axes=1)([user_embedding, movie_embedding])
output = Dense(1, activation="linear")(dot_product)

model_collab = Model([user_input, movie_input], output)
model_collab.compile(optimizer="adam", loss="mse")

print(model_collab.summary())

# 🔹 Train Content-Based Model
X_train = movies[["genres_encoded"]].values
y_train = np.random.rand(len(movies))  # Fake labels for training

model_content.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)

# 🔹 Train Collaborative Filtering Model
X_train_users = ratings["userId"].values
X_train_movies = ratings["movieId"].values
y_train_ratings = ratings["rating"].values

model_collab.fit([X_train_users, X_train_movies], y_train_ratings, epochs=5, batch_size=64, verbose=1)

# 🔹 Fuzzy Matching to Find Closest Movie
def find_closest_movie(title):
    title = title.strip().lower()
    movies["title"] = movies["title"].str.strip().str.lower()

    choices = movies["title"].tolist()
    best_match = process.extractOne(title, choices)

    return best_match[0] if best_match and best_match[1] > 80 else None  # 80% match threshold

# 🔹 Hybrid Recommendation System
def hybrid_recommend(user_id, movie_title):
    movie_title = find_closest_movie(movie_title)
    
    if movie_title is None:
        return {"error": "Movie not found in dataset."}

    matched_movies = movies[movies["title"] == movie_title]
    if matched_movies.empty:
        return {"error": "Movie ID not found."}

    movie_id = matched_movies.iloc[0]["movieId"]

    if movie_id < 0 or movie_id >= num_movies:
        return {"error": f"Movie ID {movie_id} is out of range."}

    try:
        content_score = model_content.predict(np.array([[movie_id]]))[0][0]
        collab_score = model_collab.predict([np.array([user_id]), np.array([movie_id])])[0][0]
        final_score = 0.5 * content_score + 0.5 * collab_score
    except:
        final_score = "N/A"  # Handle cases where prediction fails

    return {"title": movie_title, "predicted_score": final_score}


# 🔹 Example Usage
print(hybrid_recommend(user_id=1, movie_title="Inception"))

import joblib

# Save the trained models
joblib.dump(model_content, "content_model.pkl")
joblib.dump(model_collab, "collab_model.pkl")

print("Models saved successfully!")
