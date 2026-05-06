import torch
import torch.nn as nn
from typing import Dict


class UserTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_languages: int,
        embedding_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.lang_embedding = nn.Embedding(num_languages + 1, 32)

        continuous_dim = 3
        input_dim = embedding_dim + 32 + continuous_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.lang_embedding.weight, std=0.01)

    def forward(self, user_id, user_continuous, user_lang):
        u_emb = self.user_embedding(user_id)   # (B, 128)
        l_emb = self.lang_embedding(user_lang) # (B, 32)
        x = torch.cat([u_emb, l_emb, user_continuous], dim=-1)
        return self.network(x)                 # (B, 256)


class MovieTower(nn.Module):
    def __init__(
        self,
        num_movies: int,
        num_languages: int,
        num_genres: int,
        embedding_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()

        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.lang_embedding = nn.Embedding(num_languages + 1, 32)

        input_dim = embedding_dim + 32 + 3 + num_genres

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        nn.init.normal_(self.lang_embedding.weight, std=0.01)

    def forward(self, movie_id, movie_continuous, movie_language, movie_genres):
        m_emb = self.movie_embedding(movie_id)      # (B, 128)
        l_emb = self.lang_embedding(movie_language) # (B, 32)
        x = torch.cat([m_emb, l_emb, movie_continuous, movie_genres], dim=-1)
        return self.network(x)                      # (B, 256)


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
            batch["movie_genres"],
        )

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
                batch["movie_genres"],
            ),
            dim=-1,
        )