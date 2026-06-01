from fastapi import FastAPI, HTTPException
import os
import httpx
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import dataset

app = FastAPI(title="Movie Recommender API", version="1.0.0")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    movie_title: str
    num_recommendations: Optional[int] = 6

class MovieInfo(BaseModel):
    id: int
    title: str
    genres: List[str]
    average_rating: float

class RecommendResponse(BaseModel):
    recommendations: List[MovieInfo]

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Movie Recommender Engine"}

@app.get("/movies", response_model=List[str])
def list_movies():
    return dataset.get_all_movies()

@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest):
    recs = dataset.get_recommendations(payload.movie_title, payload.num_recommendations)
    if recs is None:
        raise HTTPException(status_code=404, detail="Movie not found in the dataset index")
    return {"recommendations": recs}


# Simple OMDb proxy to keep API key server-side. Frontend should call `/omdb?t=Title`.
OMDB_URL = "https://www.omdbapi.com/"

@app.get("/omdb")
async def omdb_proxy(t: str):
    key = os.environ.get("OMDB_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="OMDB_KEY not configured")
    try:
        params = {"t": t, "apikey": key}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(OMDB_URL, params=params)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail="OMDb fetch failed")
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

# Mount static files folder to serve the frontend on http://localhost:8000/
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
