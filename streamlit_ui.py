import streamlit as st
import requests

st.title("🎬 Hybrid Movie Recommendation System")

movie_name = st.text_input("Enter a movie name:")
user_id = st.number_input("Enter User ID:", min_value=1, step=1)

if st.button("Get Recommendations"):
    FASTAPI_URL = "https://your-fastapi-app.onrender.com"  # Replace with your Render URL
    response = requests.get(f"{FASTAPI_URL}/recommend/?user_id={user_id}&title={movie_name}")

    
    if response.status_code == 200:
        result = response.json()
        
        if "error" in result:
            st.write("❌ Error:", result["error"])  # Show error message
        elif "recommended_movies" in result:
            st.write(f"🎥 **Recommended Movies for {movie_name}:**")
            for movie in result["recommended_movies"]:
                st.write(f"- {movie}")  # Display as bullet points
        else:
            st.write("❌ Unexpected response format.")
    else:
        st.write("❌ Failed to fetch recommendations.")
