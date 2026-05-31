from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import os

app = FastAPI(title="Movie Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data on startup
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(DATA_DIR, 'similarity_data.pkl'), 'rb') as f:
    all_properties = pickle.load(f)

with open(os.path.join(DATA_DIR, 'movie_list.pkl'), 'rb') as f:
    movie_list = pickle.load(f)

def calculate_distance(id1, id2):
    try:
        pop_dis = abs(all_properties.loc[id1, 'popularity'] - all_properties.loc[id2, 'popularity'])
        genre_dis = cosine(all_properties.loc[id1, 'Action':'Western'], all_properties.loc[id2, 'Action':'Western'])
        
        # Handle potential NaN in cosine distance (if no genres marked)
        if np.isnan(genre_dis):
            genre_dis = 1.0
            
        return pop_dis + genre_dis
    except Exception:
        return 10.0

def get_neighbours(movie_id, k=5):
    distances = []
    for mid in all_properties.index:
        if mid != movie_id:
            dist = calculate_distance(movie_id, mid)
            distances.append((mid, dist))
    
    distances.sort(key=lambda x: x[1])
    return distances[:k]

class RecommendRequest(BaseModel):
    movie_title: str
    num_recommendations: int = 5

@app.get("/")
def read_root():
    return {"message": "Movie Recommender API is live!"}

@app.get("/movies")
def list_movies():
    # Return a list of movies for the autocomplete search
    return movie_list['movie title'].tolist()

@app.post("/recommend")
def recommend_movies(request: RecommendRequest):
    # Find movie ID by title (case insensitive)
    matched_movies = movie_list[movie_list['movie title'].str.lower() == request.movie_title.lower()]
    
    if matched_movies.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    movie_id = matched_movies.iloc[0]['movie id']
    
    neighbour_ids = get_neighbours(movie_id, k=request.num_recommendations)
    
    recommendations = []
    for mid, dist in neighbour_ids:
        movie_info = movie_list[movie_list['movie id'] == mid].iloc[0]
        # Also get properties like rating
        avg_rating = all_properties.loc[mid, 'mean']
        recommendations.append({
            "id": int(mid),
            "title": str(movie_info['movie title']),
            "average_rating": float(avg_rating)
        })
        
    return {
        "input_movie": str(matched_movies.iloc[0]['movie title']),
        "recommendations": recommendations
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
