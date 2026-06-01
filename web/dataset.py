import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# List of popular movies representing a subset of MovieLens-100k
MOVIES = [
    {"id": 1, "title": "Toy Story (1995)", "genres": ["Animation", "Children", "Comedy"]},
    {"id": 2, "title": "GoldenEye (1995)", "genres": ["Action", "Adventure", "Thriller"]},
    {"id": 3, "title": "Get Shorty (1995)", "genres": ["Action", "Comedy", "Drama"]},
    {"id": 4, "title": "Twelve Monkeys (1995)", "genres": ["Drama", "Sci-Fi"]},
    {"id": 5, "title": "Babe (1995)", "genres": ["Children", "Comedy", "Drama"]},
    {"id": 6, "title": "Dead Man Walking (1995)", "genres": ["Drama"]},
    {"id": 7, "title": "Seven (Se7en) (1995)", "genres": ["Action", "Thriller"]},
    {"id": 8, "title": "Usual Suspects, The (1995)", "genres": ["Crime", "Mystery", "Thriller"]},
    {"id": 9, "title": "Toy Story 2 (1999)", "genres": ["Animation", "Children", "Comedy"]},
    {"id": 10, "title": "Braveheart (1995)", "genres": ["Action", "Drama", "War"]},
    {"id": 11, "title": "Apollo 13 (1995)", "genres": ["Drama", "IMAX"]},
    {"id": 12, "title": "Star Wars (1977)", "genres": ["Action", "Adventure", "Sci-Fi"]},
    {"id": 13, "title": "Pulp Fiction (1994)", "genres": ["Comedy", "Crime", "Drama", "Thriller"]},
    {"id": 14, "title": "Shawshank Redemption, The (1994)", "genres": ["Crime", "Drama"]},
    {"id": 15, "title": "Forrest Gump (1994)", "genres": ["Comedy", "Drama", "Romance", "War"]},
    {"id": 16, "title": "Lion King, The (1994)", "genres": ["Adventure", "Animation", "Children", "Drama", "Musical"]},
    {"id": 17, "title": "Jurassic Park (1993)", "genres": ["Action", "Adventure", "Sci-Fi", "Thriller"]},
    {"id": 18, "title": "Blade Runner (1982)", "genres": ["Action", "Sci-Fi", "Thriller"]},
    {"id": 19, "title": "Alien (1979)", "genres": ["Action", "Horror", "Sci-Fi"]},
    {"id": 20, "title": "Terminator 2: Judgment Day (1991)", "genres": ["Action", "Sci-Fi"]},
    {"id": 21, "title": "Silence of the Lambs, The (1991)", "genres": ["Crime", "Horror", "Thriller"]},
    {"id": 22, "title": "Fargo (1996)", "genres": ["Comedy", "Crime", "Drama", "Thriller"]},
    {"id": 23, "title": "Godfather, The (1972)", "genres": ["Crime", "Drama"]},
    {"id": 24, "title": "Empire Strikes Back, The (1980)", "genres": ["Action", "Adventure", "Sci-Fi"]},
    {"id": 25, "title": "Raiders of the Lost Ark (1981)", "genres": ["Action", "Adventure"]},
    {"id": 26, "title": "Return of the Jedi (1983)", "genres": ["Action", "Adventure", "Sci-Fi"]},
    {"id": 27, "title": "Independence Day (ID4) (1996)", "genres": ["Action", "Adventure", "Sci-Fi"]},
    {"id": 28, "title": "Copycat (1995)", "genres": ["Crime", "Drama", "Thriller"]},
    {"id": 29, "title": "Schindler's List (1993)", "genres": ["Drama", "War"]},
    {"id": 30, "title": "Matrix, The (1999)", "genres": ["Action", "Sci-Fi", "Thriller"]}
]

# Generate mock ratings to simulate collaborative behavior:
# e.g., Animation lovers rate Toy Story, Toy Story 2, Lion King high
# Sci-Fi lovers rate Star Wars, Empire, Return of the Jedi, Blade Runner, Alien, Matrix high
np.random.seed(42)
num_users = 50
ratings_data = []

for user_id in range(1, num_users + 1):
    profile_type = user_id % 3  # 0: Sci-Fi fan, 1: Animation fan, 2: Drama/Thriller fan
    
    for movie in MOVIES:
        m_id = movie["id"]
        genres = movie["genres"]
        
        # Base rating
        rating = 0
        should_rate = np.random.rand() < 0.6  # 60% chance of rating a movie
        
        if should_rate:
            if profile_type == 0:  # Sci-Fi lover
                if any(g in ["Sci-Fi", "Action", "Adventure"] for g in genres):
                    rating = np.random.choice([4, 5], p=[0.3, 0.7])
                else:
                    rating = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            elif profile_type == 1:  # Animation/Kids lover
                if any(g in ["Animation", "Children", "Comedy"] for g in genres):
                    rating = np.random.choice([4, 5], p=[0.4, 0.6])
                else:
                    rating = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            else:  # Drama/Crime/Thriller lover
                if any(g in ["Drama", "Crime", "Thriller", "Horror"] for g in genres):
                    rating = np.random.choice([4, 5], p=[0.3, 0.7])
                else:
                    rating = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            
            ratings_data.append({"userId": user_id, "movieId": m_id, "rating": rating})

df_ratings = pd.DataFrame(ratings_data)
df_movies = pd.DataFrame(MOVIES)

# Calculate average rating for each movie
avg_ratings = df_ratings.groupby("movieId")["rating"].mean().to_dict()
for movie in MOVIES:
    movie["average_rating"] = round(avg_ratings.get(movie["id"], 3.5), 2)

df_movies = pd.DataFrame(MOVIES)

# Build rating pivot matrix (movies x users) for Item-based collaborative filtering
pivot_matrix = df_ratings.pivot(index="movieId", columns="userId", values="rating").fillna(0)
movie_sparse = csr_matrix(pivot_matrix.values)

# Fit KNN model with Cosine Similarity
model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
model_knn.fit(movie_sparse)

def get_recommendations(movie_title: str, num_recommendations: int = 6):
    try:
        # Find matching movie ID using literal string matching to avoid regex issues
        matches = df_movies[df_movies["title"].str.lower().str.contains(movie_title.lower(), regex=False)]
        if matches.empty:
            return None
        
        movie_idx = matches.index[0]
        movie_id = df_movies.iloc[movie_idx]["id"]
        
        # Get position in the pivot table (movieId index)
        pivot_idx = pivot_matrix.index.get_loc(movie_id)
        
        # Query KNN
        distances, indices = model_knn.kneighbors(
            pivot_matrix.iloc[pivot_idx, :].values.reshape(1, -1), 
            n_neighbors=num_recommendations + 1
        )
        
        recommendations = []
        for i in range(1, len(distances.flatten())):
            neighbor_pivot_idx = indices.flatten()[i]
            neighbor_movie_id = pivot_matrix.index[neighbor_pivot_idx]
            
            # Find movie details
            row = df_movies[df_movies["id"] == neighbor_movie_id].iloc[0]
            recommendations.append({
                "id": int(row["id"]),
                "title": row["title"],
                "genres": row["genres"],
                "average_rating": float(row["average_rating"])
            })
            
        return recommendations
    except Exception as e:
        print("Recommendation error:", e)
        return None

def get_all_movies():
    return list(df_movies["title"].values)
