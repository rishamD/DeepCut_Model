import torch
import torch.nn as nn
from typing import Dict
class UserTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_languages: int,
        embedding_dim: int = 64,
        output_dim: int = 128,
    ):
        super().__init__()

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.lang_embedding = nn.Embedding(num_languages + 1, 16)

        # Continuous features: avg_rating, num_ratings, avg_popularity = 3
        continuous_dim = 3

        input_dim = embedding_dim + 16 + continuous_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, user_id, user_continuous, user_lang):
        u_emb = self.user_embedding(user_id)           # (B, 64)
        l_emb = self.lang_embedding(user_lang)         # (B, 16)
        x = torch.cat([u_emb, l_emb, user_continuous], dim=-1)
        return self.network(x)                         # (B, 128)


class MovieTower(nn.Module):
    def __init__(
        self,
        num_movies: int,
        num_languages: int,
        num_content_types: int,
        num_genres: int,
        embedding_dim: int = 64,
        output_dim: int = 128,
    ):
        super().__init__()

        # Embeddings
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.lang_embedding = nn.Embedding(num_languages + 1, 16)
        self.content_type_embedding = nn.Embedding(num_content_types + 1, 8)

        # Continuous: popularity, runtime, year = 3
        # Genre multi-hot: num_genres
        input_dim = embedding_dim + 16 + 8 + 3 + num_genres

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, movie_id, movie_continuous, movie_language, movie_content_type, movie_genres):
        m_emb = self.movie_embedding(movie_id)                      # (B, 64)
        l_emb = self.lang_embedding(movie_language)                 # (B, 16)
        ct_emb = self.content_type_embedding(movie_content_type)    # (B, 8)

        x = torch.cat(
            [m_emb, l_emb, ct_emb, movie_continuous, movie_genres], dim=-1
        )
        return self.network(x)                                      # (B, 128)


class TwoTowerModel(nn.Module):
    def __init__(self, user_tower: UserTower, movie_tower: MovieTower):
        super().__init__()
        self.user_tower = user_tower
        self.movie_tower = movie_tower

    def forward(self, batch: Dict) -> torch.Tensor:
        user_emb = self.user_tower(
            batch["user_id"],
            batch["user_continuous"],
            batch["user_lang"],
        )
        movie_emb = self.movie_tower(
            batch["movie_id"],
            batch["movie_continuous"],
            batch["movie_language"],
            batch["movie_content_type"],
            batch["movie_genres"],
        )

        # L2 normalize → cosine similarity via dot product
        user_emb = nn.functional.normalize(user_emb, dim=-1)
        movie_emb = nn.functional.normalize(movie_emb, dim=-1)

        scores = (user_emb * movie_emb).sum(dim=-1)  # (B,)
        return scores

    def get_user_embedding(self, batch: Dict) -> torch.Tensor:
        return nn.functional.normalize(
            self.user_tower(
                batch["user_id"],
                batch["user_continuous"],
                batch["user_lang"],
            ),
            dim=-1,
        )

    def get_movie_embedding(self, batch: Dict) -> torch.Tensor:
        return nn.functional.normalize(
            self.movie_tower(
                batch["movie_id"],
                batch["movie_continuous"],
                batch["movie_language"],
                batch["movie_content_type"],
                batch["movie_genres"],
            ),
            dim=-1,
        )
