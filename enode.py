import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Any, Optional
import numpy as np


class VocabBuilder:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.language_encoder = LabelEncoder()
        self.genre_vocab: Dict[str, int] = {}

    def fit(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        user_features_df: Optional[pd.DataFrame] = None,
    ):
        self.user_encoder.fit(ratings_df["user_id"])
        self.movie_encoder.fit(movies_df["movie_id"])

        movie_langs = movies_df["original_language"].fillna("unknown")
        if user_features_df is not None:
            user_langs = user_features_df["preferred_language"].fillna("unknown")
            all_langs = pd.concat([movie_langs, user_langs]).unique()
        else:
            all_langs = movie_langs.unique()
        self.language_encoder.fit(all_langs)

        all_genres = [
            g
            for genres in movies_df["genres"]
            for g in (genres if isinstance(genres, list) else [])
        ]
        self.genre_vocab = {
            g: i for i, g in enumerate(sorted(set(all_genres)))
        }
        return self

    def safe_encode(self, encoder: LabelEncoder, values: Any) -> np.ndarray:
        """Maps unseen labels to index 0. Vectorized — no Python loop."""
        class_to_idx = {c: i for i, c in enumerate(encoder.classes_)}
        if isinstance(values, pd.Series):
            return values.map(class_to_idx).fillna(0).values.astype(np.int64)
        return np.array(
            [class_to_idx.get(v, 0) for v in values], dtype=np.int64
        )

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