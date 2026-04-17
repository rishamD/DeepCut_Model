import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Any
import numpy as np

class VocabBuilder:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.language_encoder = LabelEncoder()
        self.genre_vocab: Dict[str, int] = {}
        self.content_type_encoder = LabelEncoder()

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        # Fit exactly on the data present during training
        self.user_encoder.fit(ratings_df["user_id"])
        self.movie_encoder.fit(movies_df["movie_id"])
        
        self.language_encoder.fit(movies_df["original_language"].fillna("unknown"))
        self.content_type_encoder.fit(movies_df["content_type"].fillna("unknown"))

        # Genre multi-hot vocab
        all_genres = [
            g
            for genres in movies_df["genres"]
            for g in (genres if isinstance(genres, list) else [])
        ]
        unique_genres = sorted(set(all_genres))
        self.genre_vocab = {g: i for i, g in enumerate(unique_genres)}
        return self

    def safe_encode(self, encoder: LabelEncoder, values: Any) -> np.ndarray:
        """Maps unseen labels to index 0 to maintain model compatibility."""
        known_classes = set(encoder.classes_)
        default_idx = 0 
        
        return np.array([
            encoder.transform([x])[0] if x in known_classes else default_idx 
            for x in values
        ])

    def encode_genres_multihot(self, genres: List[str]) -> np.ndarray:
        vec = np.zeros(len(self.genre_vocab), dtype=np.float32)
        if not isinstance(genres, (list, np.ndarray)):
            return vec
        for g in genres:
            if g in self.genre_vocab:
                vec[self.genre_vocab[g]] = 1.0
        return vec

    @property
    def num_users(self): return len(self.user_encoder.classes_)
    @property
    def num_movies(self): return len(self.movie_encoder.classes_)
    @property
    def num_languages(self): return len(self.language_encoder.classes_)
    @property
    def num_genres(self): return len(self.genre_vocab)
    @property
    def num_content_types(self): return len(self.content_type_encoder.classes_)