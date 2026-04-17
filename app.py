import torch
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import torch.nn.functional as F

# Import your classes from the local file
from enode import TwoTowerModel, UserTower, MovieTower, VocabBuilder

app = FastAPI(title="DeepCut Pi-Engine")

# 1. Define the Input Schema
# Expected format: {"username": "risham", "ratings": [(101, 4.5), (202, 5.0)]}
class Rating(BaseModel):
    movie_id: int
    rating: float

class RecRequest(BaseModel):
    username: str
    ratings: List[Tuple[int, float]]

# Global assets
model = None
vocab = None
movie_db = None
movie_embeddings = None
device = torch.device("cpu")

@app.on_event("startup")
def load_assets():
    global model, vocab, movie_db, movie_embeddings
    print("🚀 Initializing DeepCut on Pi 4...")
    
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    
    movie_db = pd.read_parquet("movies.parquet")
    
    # Initialize Model (Matching your 114-dim architecture)
    user_tower = UserTower(num_users=vocab.num_users, num_languages=vocab.num_languages)
    movie_tower = MovieTower(
        num_movies=vocab.num_movies, 
        num_languages=vocab.num_languages,
        num_content_types=vocab.num_content_types,
        num_genres=vocab.num_genres
    )
    model = TwoTowerModel(user_tower, movie_tower).to(device)
    model.load_state_dict(torch.load("best_two_tower.pt", map_location=device))
    model.eval()

    # Pre-compute all movie embeddings once to save Pi's CPU cycles during requests
    print("📦 Pre-computing movie library embeddings...")
    # (Assuming you have a helper in enode to get all movie features)
    # movie_embeddings = engine.compute_all_movie_vectors() 
    
    torch.set_num_threads(4)
    print("✅ Pi is ready for requests.")

@app.post("/recommend")
async def get_recommendations(data: RecRequest):
    try:
        # 1. Filter out movies not in our vocabulary
        valid_ratings = []
        known_movie_ids = set(vocab.movie_encoder.classes_)
        
        for m_id, rating in data.ratings:
            if m_id in known_movie_ids:
                valid_ratings.append((m_id, rating))
        
        if not valid_ratings:
            return {"status": "error", "message": "No known movies found in input."}

        # 2. Aggregate User Profile
        # We simulate the User Tower input by averaging the embeddings 
        # of the movies the user liked (weighted by rating)
        with torch.no_grad():
            # This is a simplified "Average Taste" approach for inference
            # In a production Two-Tower, you'd pass user-specific stats
            # Here we just find movies similar to their high-rated ones
            user_vector = torch.randn(1, 128) # Placeholder for the 128-dim embedding
            
            # 3. Calculate Scores (Cosine Similarity)
            # scores = torch.matmul(user_vector, movie_embeddings.T)
            # top_k_indices = torch.topk(scores, k=10).indices
            
            # Mocking the return for testing
            recommendations = [
                {"title": "The Matrix", "year": 1999, "score": 0.98},
                {"title": "Inception", "year": 2010, "score": 0.95}
            ]

        return {
            "username": data.username,
            "recommendations": recommendations,
            "processed_count": len(valid_ratings)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)