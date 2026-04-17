import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# This is the line that fixes your specific error:
from enode import VocabBuilder

class MovieRatingDataset(Dataset):
    def __init__(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        vocab: VocabBuilder,
    ):
        self.vocab = vocab

        # Merge data to ensure all features are available per rating
        merged = ratings_df.merge(movies_df, on="movie_id", how="left")
        merged = merged.merge(user_features_df, on="user_id", how="left")

        # Encode IDs
        self.user_ids = vocab.user_encoder.transform(merged["user_id"])
        self.movie_ids = vocab.movie_encoder.transform(merged["movie_id"])
        
        # Labels (Normalized ratings)
        self.labels = merged["rating_norm"].values.astype(np.float32)

        # ── User features ──────────────────────────────────────────────────
        self.user_avg_rating = merged["avg_rating"].fillna(0).values.astype(np.float32)
        self.user_num_ratings = merged["num_ratings"].fillna(0).values.astype(np.float32)
        self.user_avg_popularity = merged["avg_popularity"].fillna(0).values.astype(np.float32)
        self.user_lang = vocab.language_encoder.transform(
            merged["preferred_language"].fillna("unknown")
        )

        # ── Movie features ─────────────────────────────────────────────────
        self.movie_popularity = merged["popularity_scaled"].fillna(0).values.astype(np.float32)
        self.movie_runtime = merged["runtime_scaled"].fillna(0).values.astype(np.float32)
        self.movie_year = merged["year_scaled"].fillna(0).values.astype(np.float32)
        
        self.movie_language = vocab.language_encoder.transform(
            merged["original_language"].fillna("unknown")
        )
        self.movie_content_type = vocab.content_type_encoder.transform(
            merged["content_type"].fillna("unknown")
        )
        
        # Multi-hot encode genres
        self.movie_genres = np.stack(
            merged["genres"].apply(vocab.encode_genres_multihot).values
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            # IDs
            "user_id": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "movie_id": torch.tensor(self.movie_ids[idx], dtype=torch.long),
            
            # User continuous & categorical
            "user_continuous": torch.tensor(
                [
                    self.user_avg_rating[idx],
                    self.user_num_ratings[idx],
                    self.user_avg_popularity[idx],
                ],
                dtype=torch.float32,
            ),
            "user_lang": torch.tensor(self.user_lang[idx], dtype=torch.long),
            
            # Movie continuous & categorical
            "movie_continuous": torch.tensor(
                [
                    self.movie_popularity[idx],
                    self.movie_runtime[idx],
                    self.movie_year[idx],
                ],
                dtype=torch.float32,
            ),
            "movie_language": torch.tensor(self.movie_language[idx], dtype=torch.long),
            "movie_content_type": torch.tensor(self.movie_content_type[idx], dtype=torch.long),
            "movie_genres": torch.tensor(self.movie_genres[idx], dtype=torch.float32),
            
            # Label
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }