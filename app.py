import torch
import pickle
import pandas as pd
import faiss
import sqlite3
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime

# Import your local project modules
from twoTowerModel import TwoTowerModel, UserTower, MovieTower
from infer import RecommendationEngine
from preProcess import preprocess_movies

app = FastAPI(title="DeepCut Recommendation API")

# --- DATA MODELS ---
class MovieReview(BaseModel):
    slug: str
    title: str
    rating: float

class UserHistory(BaseModel):
    username: str
    reviews: List[MovieReview]
    top_k: int = 10

# --- GLOBAL STATE ---
model = None
vocab = None
movie_db = None
engine = None
DB_PATH = "deepcut_extras.db"

# --- DATABASE LOGIC ---
def init_db():
    """Checks if the database exists; if not, creates the schema."""
    db_exists = os.path.exists(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if not db_exists:
        print(f"📁 Database not found. Creating new DB at {DB_PATH}...")
    
    # 1. Unknown movies table (stores slug/title for movies not in Parquet)
    cursor.execute('''CREATE TABLE IF NOT EXISTS unknown_movies 
                     (movie_id TEXT PRIMARY KEY, title TEXT, slug TEXT, added_at DATETIME)''')
    
    # 2. Reviews table (stores movie_id, user_id, rating_val)
    cursor.execute('''CREATE TABLE IF NOT EXISTS reviews 
                     (movie_id TEXT, user_id TEXT, rating_val INTEGER, timestamp DATETIME)''')
    
    conn.commit()
    conn.close()

# --- STARTUP EVENT ---
@app.on_event("startup")
def startup():
    global model, vocab, movie_db, engine
    device = torch.device("cpu")
    
    print("🚀 Startup: Initializing DeepCut Resources...")
    
    # Check/Create Database
    init_db()

    # 1. Load Vocabulary
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # 2. Initialize Two-Tower Model
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

    # 3. Load Parquet Metadata into high-speed Dictionary
    print("📦 Loading movie metadata from Parquet...")
    raw_df = pd.read_parquet("movies.parquet")
    # Store in dict for O(1) lookups during result enrichment
    movie_db = raw_df.set_index('movie_id').to_dict('index')
    
    # Preprocess for the FAISS index building (if needed)
    processed_df = preprocess_movies(raw_df)

    # 4. Setup Recommendation Engine
    engine = RecommendationEngine(model, vocab, device="cpu")
    
    try:
        print("🔍 Loading FAISS index from disk...")
        engine.index = faiss.read_index("movie_index.bin")
        with open("movie_ids.pkl", "rb") as f:
            engine.movie_id_map = pickle.load(f)
        print("✅ Index loaded successfully!")
    except Exception as e:
        print(f"⚠️ Pre-computed index error: {e}. Building index from Parquet...")
        engine.build_movie_index(processed_df)

    print("🔥 DeepCut Recommendation Engine is LIVE on Raspberry Pi!")

# --- RECOMMENDATION ENDPOINT ---
@app.post("/recommend")
async def get_recommendations(data: UserHistory):
    if not engine:
        raise HTTPException(status_code=503, detail="Model initializing")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now = datetime.now()

        # history_for_model will store {movie_id: rating} for valid vocab matches
        history_for_model = {}
        watched_ids = []

        # 1. PROCESS INCOMING REVIEWS
        for review in data.reviews:
            m_id = review.slug  # Slug is treated as our movie_id
            watched_ids.append(m_id)
            
            # Save review to SQLite
            cursor.execute(
                "INSERT INTO reviews (movie_id, user_id, rating_val, timestamp) VALUES (?, ?, ?, ?)",
                (m_id, data.username, int(review.rating), now)
            )
            
            # Capture metadata for movies not in our master Parquet
            if m_id not in movie_db:
                cursor.execute(
                    "INSERT OR IGNORE INTO unknown_movies (movie_id, title, slug, added_at) VALUES (?, ?, ?, ?)",
                    (m_id, review.title, review.slug, now)
                )
            
            # Filter for model input (must exist in training vocabulary)
            if m_id in vocab.movie_encoder.classes_:
                history_for_model[m_id] = review.rating

        conn.commit()
        conn.close()

        # 2. ABORT IF NO USABLE DATA FOR INFERENCE
        if not history_for_model:
            return {
                "recommendations": [], 
                "warning": "None of the provided movies are in the model's training vocabulary."
            }

        # 3. FEATURE EXTRACTION FOR USER TOWER
        # Retrieve metadata for only the known movies to calculate taste vector
        history_metadata = [movie_db[m_id] for m_id in history_for_model.keys() if m_id in movie_db]
        
        if not history_metadata:
            # Safe defaults if no metadata is available for history
            user_features = {'avg_rating': 0.5, 'num_ratings': 0, 'avg_popularity': 0.5, 'preferred_language': 'en'}
        else:
            avg_pop = sum(m.get('popularity_scaled', 0.5) for m in history_metadata) / len(history_metadata)
            langs = [m.get('original_language', 'en') for m in history_metadata]
            pref_lang = max(set(langs), key=langs.count)
            
            user_features = {
                'avg_rating': (sum(history_for_model.values()) / len(history_for_model)) / 5.0,
                'num_ratings': len(history_for_model),
                'avg_popularity': avg_pop,
                'preferred_language': pref_lang
            }

        # 4. RUN INFERENCE
        recs = engine.recommend(
            user_id=data.username, 
            user_features=user_features, 
            top_k=data.top_k,
            exclude_seen=watched_ids
        )
        
        # 5. ENRICH RESPONSE WITH PARQUET METADATA
        enriched_recs = []
        for r in recs:
            rec_id = r['movie_id']
            meta = movie_db.get(rec_id, {})
            enriched_recs.append({
                "movie_id": rec_id,
                "score": float(r['score']),
                "title": meta.get("movie_title", "Unknown"),
                "image_url": meta.get("image_url", ""),
                "year": meta.get("year_released", None)
            })

        return {"recommendations": enriched_recs}

    except Exception as e:
        print(f"❌ Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run with standard Pi settings
    uvicorn.run(app, host="0.0.0.0", port=8000)