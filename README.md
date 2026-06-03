# 🎬 Movie Recommender System

Live Demo: [Movie Recommender →](https://harrsh-here-movie-recommender.up.railway.app)


## Overview
A content-based movie recommendation engine built using the MovieLens 100k dataset. It identifies and recommends movies similar to a user's selection by analyzing genres and popularity metrics.

## How It Works
The system uses a **Content-Based Filtering** approach with a **K-Nearest Neighbors (KNN)** algorithm. It calculates similarity between movies based on:
- **Genre Similarity**: Computed using Cosine Distance between multi-hot encoded genre vectors.
- **Popularity Distance**: Computed as the absolute difference between normalized popularity scores (derived from the number of ratings).

The final distance is a weighted sum of these factors, allowing the model to recommend movies that are both topically similar and have comparable popularity.

## Tech Stack
- **Languages**: Python 3.x
- **Libraries**: Pandas, NumPy, Scikit-learn, SciPy
- **API Framework**: FastAPI
- **Data Source**: MovieLens 100k

## Dataset
This project uses the **MovieLens 100k** dataset provided by GroupLens Research. The data includes 100,000 ratings from 943 users on 1,682 movies.

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/harrsh-here/movie-recommender.git
   ```
2. Navigate to the project directory:
   ```bash
   cd movie-recommender
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the API:
   ```bash
   python app.py
   ```
   The API will be available at `http://localhost:8000`.

## Screenshots
<img width="1875" height="862" alt="image" src="https://github.com/user-attachments/assets/46a3a891-419d-4752-b587-c50580de99bc" />
<img width="1897" height="971" alt="image" src="https://github.com/user-attachments/assets/cb195853-2536-4999-878d-cb72fe2b8238" />

