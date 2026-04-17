import torch
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from typing import List, Dict
from twoTowerModel import TwoTowerModel
from enode import VocabBuilder

class RecommendationEngine:
    def __init__(self, model: TwoTowerModel, vocab: VocabBuilder, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.device = device
        self.index = None
        self.movie_id_map: List[str] = []

    def build_movie_index(self, movies_df: pd.DataFrame):
        all_embeddings = []
        self.movie_id_map = movies_df["movie_id"].tolist()
        
        batch_size = 512
        total_movies = len(movies_df)
        
        for start in tqdm(range(0, total_movies, batch_size), desc="Encoding Movies"):
            chunk = movies_df.iloc[start : start + batch_size]
            
            movie_ids = self.vocab.safe_encode(self.vocab.movie_encoder, chunk["movie_id"])
            movie_language = self.vocab.safe_encode(self.vocab.language_encoder, chunk["original_language"].fillna("unknown"))
            movie_content_type = self.vocab.safe_encode(self.vocab.content_type_encoder, chunk["content_type"].fillna("unknown"))
            movie_genres = np.stack(chunk["genres"].apply(self.vocab.encode_genres_multihot).values)

            batch = {
                "movie_id": torch.tensor(movie_ids, dtype=torch.long).to(self.device),
                "movie_continuous": torch.tensor(
                    chunk[["popularity_scaled", "runtime_scaled", "year_scaled"]].values, 
                    dtype=torch.float32
                ).to(self.device),
                "movie_language": torch.tensor(movie_language, dtype=torch.long).to(self.device),
                "movie_content_type": torch.tensor(movie_content_type, dtype=torch.long).to(self.device),
                "movie_genres": torch.tensor(movie_genres, dtype=torch.float32).to(self.device),
            }

            with torch.no_grad():
                chunk_emb = self.model.get_movie_embedding(batch).cpu().numpy()
                all_embeddings.append(chunk_emb)

        all_embeddings = np.vstack(all_embeddings)
        self.index = faiss.IndexFlatIP(all_embeddings.shape[1])
        self.index.add(all_embeddings)

    def recommend(self, user_id: str, user_features: Dict, top_k: int = 10, exclude_seen: List[str] = None) -> List[Dict]:
        """Generate recommendations while optionally filtering out specific movie IDs."""
        uid = self.vocab.safe_encode(self.vocab.user_encoder, [user_id])[0]
        lang = self.vocab.safe_encode(self.vocab.language_encoder, [user_features.get("preferred_language", "unknown")])[0]

        batch = {
            "user_id": torch.tensor([uid], dtype=torch.long).to(self.device),
            "user_continuous": torch.tensor([[
                user_features["avg_rating"], 
                user_features["num_ratings"], 
                user_features["avg_popularity"]
            ]], dtype=torch.float32).to(self.device),
            "user_lang": torch.tensor([lang], dtype=torch.long).to(self.device),
        }

        with torch.no_grad():
            user_emb = self.model.get_user_embedding(batch).cpu().numpy()

        # Fetch slightly more than top_k to account for filtered items
        fetch_k = top_k + len(exclude_seen or [])
        scores, indices = self.index.search(user_emb, fetch_k)
        
        seen_set = set(exclude_seen or [])
        results = []
        
        for score, idx in zip(scores[0], indices[0]):
            movie_id = self.movie_id_map[idx]
            if movie_id not in seen_set:
                results.append({"movie_id": movie_id, "score": float(score)})
            
            if len(results) >= top_k:
                break
                
        return results