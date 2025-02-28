import streamlit as st
import requests

st.title("🎬 Hybrid Movie Recommendation System")

FASTAPI_URL = "https://movie-recommendation-app-bii8.onrender.com"  # Corrected API URL

movie_name = st.text_input("Enter a movie name:")
user_id = st.number_input("Enter User ID (optional):", min_value=1, step=1, value=1)

if st.button("Get Recommendations"):
    try:
        # Dynamically adjust query params
        if user_id:
            response = requests.get(f"{FASTAPI_URL}/recommend/?user_id={user_id}&title={movie_name}")
        else:
            response = requests.get(f"{FASTAPI_URL}/recommend/?title={movie_name}")

        response.raise_for_status()  # Raise error for bad status codes (e.g., 404, 500)

        result = response.json()

        if "error" in result:
            st.error(f"❌ Error: {result['error']}")  # Show API error message
        elif "recommended_movies" in result:
            st.write(f"🎥 **Recommended Movies for {movie_name}:**")
            for movie in result["recommended_movies"]:
                st.write(f"- {movie}")  # Display as bullet points
        else:
            st.warning("⚠️ Unexpected response format. Check API response.")

    except requests.exceptions.RequestException as e:
        st.error(f"❌ API request failed: {e}")
