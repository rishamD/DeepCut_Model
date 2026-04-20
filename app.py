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
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now = datetime.now()

        # 1. FETCH HISTORICAL REVIEWS FROM DB
        cursor.execute("SELECT movie_id, rating_val FROM reviews WHERE user_id = ?", (data.username,))
        rows = cursor.fetchall()
        
        # history_for_model will store ALL reviews (old from DB + new from Request)
        # We use a dict so new reviews can overwrite old ones if the ID is the same
        all_history = {row[0]: float(row[1]) for row in rows}

        # 2. PROCESS NEW INCOMING REVIEWS
        for review in data.reviews:
            m_id = review.slug
            
            # Save the new review to the database
            cursor.execute(
                "INSERT INTO reviews (movie_id, user_id, rating_val, timestamp) VALUES (?, ?, ?, ?)",
                (m_id, data.username, int(review.rating), now)
            )
            
            # Capture metadata for unknown movies
            if m_id not in movie_db:
                cursor.execute(
                    "INSERT OR IGNORE INTO unknown_movies (movie_id, title, slug, added_at) VALUES (?, ?, ?, ?)",
                    (m_id, review.title, review.slug, now)
                )
            
            # Add/Update in our local history dictionary
            all_history[m_id] = review.rating

        conn.commit()
        conn.close()

        # 3. FILTER FOR MODEL (Must be in training vocab)
        known_history = {
            m_id: rating for m_id, rating in all_history.items() 
            if m_id in vocab.movie_encoder.classes_
        }
        
        if not known_history:
            return {"recommendations": [], "warning": "User has no history in model vocabulary."}

        # 4. FEATURE EXTRACTION (Based on FULL merged history)
        watched_ids = list(all_history.keys())
        history_metadata = [movie_db[m_id] for m_id in known_history.keys() if m_id in movie_db]
        
        if not history_metadata:
            user_features = {'avg_rating': 0.5, 'num_ratings': 0, 'avg_popularity': 0.5, 'preferred_language': 'en'}
        else:
            avg_pop = sum(m.get('popularity_scaled', 0.5) for m in history_metadata) / len(history_metadata)
            langs = [m.get('original_language', 'en') for m in history_metadata]
            pref_lang = max(set(langs), key=langs.count)
            
            user_features = {
                'avg_rating': (sum(known_history.values()) / len(known_history)) / 5.0,
                'num_ratings': len(known_history),
                'avg_popularity': avg_pop,
                'preferred_language': pref_lang
            }

        # 5. GENERATE RECS (Model now sees all 12 movies)
        recs = engine.recommend(
            user_id=data.username, 
            user_features=user_features, 
            top_k=data.top_k,
            exclude_seen=watched_ids # Filters out all 12 movies
        )
        
        # 6. ENRICH RESULTS
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
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run with standard Pi settings
    uvicorn.run(app, host="0.0.0.0", port=8000)