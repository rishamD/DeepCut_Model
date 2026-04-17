import torch
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import torch.nn.functional as F

# Directly importing from your enode.py
from enode import VocabBuilder

# --- ARCHITECTURE RE-DEFINITION ---
# These must match your training code exactly
import torch.nn as nn

class UserTower(nn.Module):
    def __init__(self, num_users, num_languages):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, 64)
        self.lang_emb = nn.Embedding(num_languages, 32)
        self.network = nn.Sequential(nn.Linear(64 + 32 + 2, 128), nn.ReLU(), nn.Linear(128, 128))
    def forward(self, user_ids, lang_ids, stats):
        u = self.user_emb(user_ids)
        l = self.lang_emb(lang_ids)
        x = torch.cat([u, l, stats], dim=1)
        return self.network(x)

class MovieTower(nn.Module):
    def __init__(self, num_movies, num_languages, num_content_types, num_genres):
        super().__init__()
        self.movie_emb = nn.Embedding(num_movies, 64)
        self.lang_emb = nn.Embedding(num_languages, 32)
        self.type_emb = nn.Embedding(num_content_types, 16)
        # Input: 64 (ID) + 32 (Lang) + 16 (Type) + num_genres (Multi-hot) + 3 (Stats)
        input_dim = 64 + 32 + 16 + num_genres + 3
        self.network = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128))
    def forward(self, movie_ids, lang_ids, type_ids, genres, stats):
        m = self.movie_emb(movie_ids)
        l = self.lang_emb(lang_ids)
        t = self.type_emb(type_ids)
        x = torch.cat([m, l, t, genres, stats], dim=1)
        return self.network(x)

class TwoTowerModel(nn.Module):
    def __init__(self, user_tower, movie_tower):
        super().__init__()
        self.user_tower = user_tower
        self.movie_tower = movie_tower

# --- API IMPLEMENTATION ---

app = FastAPI()

class RecRequest(BaseModel):
    username: str
    ratings: List[Tuple[int, float]] # [(movie_id, rating), ...]

# Global state
model = None
vocab = None
movie_db = None
movie_embs = None
device = torch.device("cpu")

@app.on_event("startup")
def startup():
    global model, vocab, movie_db, movie_embs
    print("Loading assets...")
    
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    
    movie_db = pd.read_parquet("movies.parquet")
    
    # Init Model
    u_tower = UserTower(vocab.num_users, vocab.num_languages)
    m_tower = MovieTower(vocab.num_movies, vocab.num_languages, vocab.num_content_types, vocab.num_genres)
    model = TwoTowerModel(u_tower, m_tower)
    model.load_state_dict(torch.load("best_two_tower.pt", map_location=device))
    model.eval()

    print("Pre-computing movie library embeddings...")
    with torch.no_grad():
        # Encode all movies in the DB for fast lookup
        m_ids = torch.tensor(vocab.movie_encoder.transform(movie_db["movie_id"]))
        l_ids = torch.tensor(vocab.safe_encode(vocab.language_encoder, movie_db["original_language"].fillna("unknown")))
        t_ids = torch.tensor(vocab.safe_encode(vocab.content_type_encoder, movie_db["content_type"].fillna("unknown")))
        
        # Build Genre Multi-hot Matrix
        g_list = [vocab.encode_genres_multihot(g) for g in movie_db["genres"]]
        genres_tensor = torch.tensor(np.stack(g_list))
        
        # Stats (Scaled Popularity, Runtime, Year)
        stats = torch.tensor(movie_db[["popularity_scaled", "runtime_scaled", "year_scaled"]].values.astype(np.float32))
        
        movie_embs = model.movie_tower(m_ids, l_ids, t_ids, genres_tensor, stats)
        # Normalize for Cosine Similarity
        movie_embs = F.normalize(movie_embs, p=2, dim=1)

@app.post("/recommend")
async def recommend(req: RecRequest):
    # 1. Filter known movies
    known_ids = set(vocab.movie_encoder.classes_)
    valid_data = [(m, r) for m, r in req.ratings if m in known_ids]
    
    if not valid_data:
        return {"error": "No recognizable movies in input."}

    # 2. Build Mock User Vector from their liked movies
    # We aggregate the embeddings of their top-rated movies to represent their 'taste'
    with torch.no_grad():
        # Get indices of the movies they provided
        movie_indices = [np.where(movie_db["movie_id"] == m_id)[0][0] for m_id, r in valid_data if r >= 4.0]
        
        if not movie_indices: # Fallback if they didn't rate anything highly
            movie_indices = [np.where(movie_db["movie_id"] == m_id)[0][0] for m_id, r in valid_data]

        user_taste = movie_embs[movie_indices].mean(dim=0, keepdim=True)
        user_taste = F.normalize(user_taste, p=2, dim=1)

        # 3. Similarity Search
        scores = torch.matmul(user_taste, movie_embs.T).squeeze(0)
        
        # 4. Filter out movies they've already seen
        seen_ids = [m_id for m_id, r in req.ratings]
        scores[movie_db["movie_id"].isin(seen_ids)] = -1.0
        
        # 5. Get Top 10
        top_v, top_i = torch.topk(scores, k=10)
        
        results = []
        for i, score in zip(top_i.tolist(), top_v.tolist()):
            row = movie_db.iloc[i]
            results.append({
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "score": round(score, 4)
            })

    return {"username": req.username, "recommendations": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)