import torch
import pickle
import pandas as pd
import faiss
import sqlite3
import os
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime

from twoTowerModel import TwoTowerModel, UserTower, MovieTower
from infer import RecommendationEngine
from preProcess import preprocess_movies
from enode import VocabBuilder

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
vocab: VocabBuilder = None
movie_db: dict = None
engine: RecommendationEngine = None
user_scaler = None
DB_PATH = "deepcut_extras.db"
device = None

# --- DATABASE ---
def init_db():
    db_exists = os.path.exists(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if not db_exists:
        print(f"📁 Database not found. Creating new DB at {DB_PATH}...")

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS unknown_movies
           (movie_id TEXT PRIMARY KEY, title TEXT, slug TEXT,
            added_at DATETIME)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS reviews
           (movie_id TEXT, user_id TEXT, rating_val INTEGER,
            timestamp DATETIME)"""
    )
    conn.commit()
    conn.close()

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vocab, movie_db, engine, user_scaler, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Startup: Initializing DeepCut on {device}...")

    init_db()

    # 1. Load VocabBuilder
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    print(
        f"   Vocab: {vocab.num_users} users | "
        f"{vocab.num_movies} movies | "
        f"{vocab.num_languages} languages | "
        f"{vocab.num_genres} genres"
    )

    # 2. Load scalers
    user_scaler = joblib.load("user_scaler.pkl")
    print("✅ User scaler loaded.")

    # 3. Build model
    u_tower = UserTower(
        num_users=vocab.num_users,
        num_languages=vocab.num_languages,
    )
    m_tower = MovieTower(
        num_movies=vocab.num_movies,
        num_languages=vocab.num_languages,
        num_genres=vocab.num_genres,
    )
    model = TwoTowerModel(u_tower, m_tower)
    model.load_state_dict(
        torch.load("best_two_tower.pt", map_location=device)
    )
    model = model.to(device)
    model.eval()
    print("✅ Model loaded.")

    # 4. Load movie metadata
    print("📦 Loading movie metadata from Parquet...")
    raw_df = pd.read_parquet("movies.parquet")
    movie_db = raw_df.set_index("movie_id").to_dict("index")
    processed_df = preprocess_movies(raw_df, fit=False)

    # 5. Setup engine + FAISS index
    engine = RecommendationEngine(model, vocab, device=str(device))

    try:
        print("🔍 Loading FAISS index from disk...")
        engine.index = faiss.read_index("movie_index.bin")
        with open("movie_ids.pkl", "rb") as f:
            engine.movie_id_map = pickle.load(f)
        print(
            f"✅ FAISS index loaded: "
            f"{engine.index.ntotal} vectors, "
            f"{len(engine.movie_id_map)} movie IDs."
        )
    except Exception as e:
        print(f"⚠️  Pre-computed index unavailable ({e}). Building now...")
        engine.build_movie_index(processed_df)
        faiss.write_index(engine.index, "movie_index.bin")
        with open("movie_ids.pkl", "wb") as f:
            pickle.dump(engine.movie_id_map, f)
        print("✅ FAISS index built and saved.")

    print("🔥 DeepCut Recommendation Engine is LIVE!")

    yield

    print("🛑 Shutting down DeepCut...")


app = FastAPI(title="DeepCut Recommendation API", lifespan=lifespan)


# --- HELPERS ---
def _build_user_features(
    known_history: dict[str, float],
    history_metadata: list[dict],
) -> dict:
    """
    Reconstructs user features and scales them using the same
    MinMaxScaler fitted in build_user_features() at training time.

    user_continuous order must match training: 
        [avg_rating, num_ratings, avg_popularity]
    """
    if not history_metadata:
        # Scale a neutral zero row for cold-start users
        scaled = user_scaler.transform([[0.0, 0.0, 0.5]])[0]
        return {
            "avg_rating": float(scaled[0]),
            "num_ratings": float(scaled[1]),
            "avg_popularity": float(scaled[2]),
            "preferred_language": "unknown",
        }

    # Raw avg_rating — pass unscaled, scaler handles it
    avg_rating = sum(known_history.values()) / len(known_history)

    # num_ratings — raw count, scaler handles it
    num_ratings = float(len(known_history))

    # avg_popularity — movie_db stores raw popularity (not scaled)
    # so this matches what build_user_features() used at training time
    avg_pop = sum(
        m.get("popularity", 0.0) for m in history_metadata
    ) / len(history_metadata)

    scaled = user_scaler.transform([[avg_rating, num_ratings, avg_pop]])[0]

    langs = [
        m.get("original_language", "unknown") for m in history_metadata
    ]
    pref_lang = max(set(langs), key=langs.count)

    return {
        "avg_rating": float(scaled[0]),
        "num_ratings": float(scaled[1]),
        "avg_popularity": float(scaled[2]),
        "preferred_language": pref_lang,
    }


# --- RECOMMENDATION ENDPOINT ---
@app.post("/recommend")
async def get_recommendations(data: UserHistory):
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        now = datetime.now()

        # 1. Pull stored reviews for this user
        cursor.execute(
            "SELECT movie_id, rating_val FROM reviews WHERE user_id = ?",
            (data.username,),
        )
        all_history: dict[str, float] = {
            row[0]: float(row[1]) for row in cursor.fetchall()
        }

        # 2. Persist incoming reviews; log unknown movies
        for review in data.reviews:
            m_id = review.slug

            cursor.execute(
                "INSERT INTO reviews "
                "(movie_id, user_id, rating_val, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (m_id, data.username, int(review.rating), now),
            )

            if m_id not in movie_db:
                cursor.execute(
                    "INSERT OR IGNORE INTO unknown_movies "
                    "(movie_id, title, slug, added_at) VALUES (?, ?, ?, ?)",
                    (m_id, review.title, review.slug, now),
                )

            all_history[m_id] = review.rating

        conn.commit()

        # 3. Filter to movies the model knows
        known_movie_ids = set(vocab.movie_encoder.classes_)
        known_history = {
            m_id: rating
            for m_id, rating in all_history.items()
            if m_id in known_movie_ids
        }

        if not known_history:
            return {
                "recommendations": [],
                "warning": "User has no watch history in model vocabulary.",
            }

        # 4. Build user features using raw popularity to match training
        history_metadata = [
            movie_db[m_id] for m_id in known_history if m_id in movie_db
        ]
        user_features = _build_user_features(known_history, history_metadata)

        # 5. Generate recommendations
        watched_ids = list(all_history.keys())
        recs = engine.recommend(
            user_id=data.username,
            user_features=user_features,
            top_k=data.top_k,
            exclude_seen=watched_ids,
        )

        # 6. Enrich with metadata
        enriched_recs = [
            {
                "movie_id": r["movie_id"],
                "score": float(r["score"]),
                "title": movie_db.get(r["movie_id"], {}).get(
                    "movie_title", "Unknown"
                ),
                "image_url": movie_db.get(r["movie_id"], {}).get(
                    "image_url", ""
                ),
                "year": movie_db.get(r["movie_id"], {}).get(
                    "year_released", None
                ),
            }
            for r in recs
        ]

        return {"recommendations": enriched_recs}

    except Exception as e:
        print(f"❌ Error in /recommend: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)