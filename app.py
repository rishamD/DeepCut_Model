import torch
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# This is the "Magic Link" - it inherits the correct classes from twoTowerModel.py
from twoTowerModel import TwoTowerModel, UserTower, MovieTower
from infer import RecommendationEngine
from preProcess import preprocess_movies

app = FastAPI(title="DeepCut Recommendation API")

# Global variables for the model and data
model = None
vocab = None
movie_db = None
engine = None

class UserHistory(BaseModel):
    # A dictionary of movie_id: rating (e.g., {"interstellar": 5.0})
    history: Dict[str, float]
    top_k: int = 10

@app.on_event("startup")
def startup():
    global model, vocab, movie_db, engine
    device = torch.device("cpu")
    print("Loading assets and building index...")

    # 1. Load Vocab
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # 2. Init Model (Dimensions now inherited from twoTowerModel.py)
    u_tower = UserTower(num_users=vocab.num_users, num_languages=vocab.num_languages)
    m_tower = MovieTower(
        num_movies=vocab.num_movies, 
        num_languages=vocab.num_languages,
        num_content_types=vocab.num_content_types,
        num_genres=vocab.num_genres
    )
    
    model = TwoTowerModel(u_tower, m_tower)
    
    # 3. Load Weights
    # This will now work because 'u_tower' and 'm_tower' match the checkpoint perfectly
    model.load_state_dict(torch.load("best_two_tower.pt", map_location=device))
    model.eval()

    # 4. Load & Preprocess Movie DB for the search index
    movie_db = pd.read_parquet("movies.parquet")
    movie_db = preprocess_movies(movie_db)

    # 5. Build Recommendation Engine
    engine = RecommendationEngine(model, vocab, device="cpu")
    engine.build_movie_index(movie_db)
    print("DeepCut Engine is LIVE!")

@app.get("/health")
def health_check():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/recommend")
async def get_recommendations(data: UserHistory):
    if not engine:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Calculate features from the provided Letterboxd history
        watched_ids = list(data.history.keys())
        history_movies = movie_db[movie_db['movie_id'].isin(watched_ids)]
        
        if history_movies.empty:
            return {"recommendations": [], "warning": "No movies found in database"}

        # Basic Feature Aggregation
        avg_rating = (sum(data.history.values()) / len(data.history)) / 5.0
        avg_pop = history_movies['popularity_scaled'].mean()
        pref_lang = history_movies['original_language'].mode()[0] if not history_movies.empty else 'en'

        user_features = {
            'avg_rating': avg_rating,
            'num_ratings': len(data.history),
            'avg_popularity': avg_pop,
            'preferred_language': pref_lang
        }

        # Generate recommendations (Filtering out already watched movies)
        recs = engine.recommend_next(
            user_id="anonymous_user", 
            user_features=user_features, 
            top_k=data.top_k,
            exclude_seen=watched_ids
        )
        
        return {"recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)