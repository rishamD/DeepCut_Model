import torch
import pickle
import pandas as pd
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

from twoTowerModel import TwoTowerModel, UserTower, MovieTower
from infer import RecommendationEngine
from preProcess import preprocess_movies

app = FastAPI(title="DeepCut Recommendation API")

# Global variables
model = None
vocab = None
movie_db = None
engine = None

class UserHistory(BaseModel):
    history: Dict[str, float]
    top_k: int = 10

@app.on_event("startup")
def startup():
    global model, vocab, movie_db, engine
    device = torch.device("cpu")
    
    print("🚀 Startup: Loading Pre-computed Assets...")

    # 1. Load Vocab
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # 2. Init Model
    u_tower = UserTower(num_users=vocab.num_users, num_languages=vocab.num_languages)
    m_tower = MovieTower(
        num_movies=vocab.num_movies, 
        num_languages=vocab.num_languages,
        num_content_types=vocab.num_content_types,
        num_genres=vocab.num_genres
    )
    model = TwoTowerModel(u_tower, m_tower)
    model.load_state_dict(torch.load("best_two_tower.pt", map_location=device))
    model.eval()

    # 3. Load Movie DB (for metadata)
    movie_db = pd.read_parquet("movies.parquet")
    movie_db = preprocess_movies(movie_db)

    # 4. Setup Engine
    engine = RecommendationEngine(model, vocab, device="cpu")
    
    # --- THE INSTANT LOAD LOGIC ---
    try:
        print("Loading FAISS index from disk...")
        engine.index = faiss.read_index("movie_index.bin")
        with open("movie_ids.pkl", "rb") as f:
            engine.movie_id_map = pickle.load(f)
        print("✅ Index loaded instantly!")
    except Exception as e:
        print(f"⚠️ Pre-computed index not found, falling back to encoding: {e}")
        engine.build_movie_index(movie_db)
    # ------------------------------

    print("DeepCut Engine is LIVE!")

@app.post("/recommend")
async def get_recommendations(data: UserHistory):
    if not engine:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # 1. FILTER: Ignore movies that aren't in our training vocabulary
        # This prevents KeyErrors inside the User Tower
        known_history = {
            m_id: rating for m_id, rating in data.history.items() 
            if m_id in vocab.movie_to_idx
        }
        
        # Identify what we dropped for the warning message
        ignored_movies = list(set(data.history.keys()) - set(known_history.keys()))
        
        if not known_history:
            return {
                "recommendations": [], 
                "warning": "None of the movies provided exist in the model's vocabulary.",
                "ignored": ignored_movies
            }

        # 2. Sync with Movie Database for feature extraction
        watched_ids = list(known_history.keys())
        history_movies = movie_db[movie_db['movie_id'].isin(watched_ids)]
        
        # 3. Feature Aggregation (using only known movies)
        user_features = {
            'avg_rating': (sum(known_history.values()) / len(known_history)) / 5.0,
            'num_ratings': len(known_history),
            'avg_popularity': history_movies['popularity_scaled'].mean() if not history_movies.empty else 0.5,
            'preferred_language': history_movies['original_language'].mode()[0] if not history_movies.empty else 'en'
        }

        # 4. Generate Recommendations
        recs = engine.recommend(
            user_id="anonymous_user", 
            user_features=user_features, 
            top_k=data.top_k,
            exclude_seen=watched_ids
        )
        
        response = {"recommendations": recs}
        if ignored_movies:
            response["warning"] = f"Ignored {len(ignored_movies)} unknown movies."
            response["ignored_ids"] = ignored_movies
            
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))