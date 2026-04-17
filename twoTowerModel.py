import torch
import torch.nn as nn
from typing import Dict

class UserTower(nn.Module):
    def __init__(self, num_users: int, num_languages: int, embedding_dim: int = 64, output_dim: int = 128):
        super().__init__()
        # Matches state_dict: 'user_embedding' and 'lang_embedding'
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # 16 + 1 (for unknown/padding) = 17. 
        # Total Input: 64 (user) + 16 (lang) + 3 (stats) = 83
        self.lang_embedding = nn.Embedding(num_languages + 1, 16)

        input_dim = embedding_dim + 16 + 3

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), # Matches [256, 83]
            nn.BatchNorm1d(256),       # Found in your error log keys
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, user_id, user_continuous, user_lang):
        u_emb = self.user_embedding(user_id)
        l_emb = self.lang_embedding(user_lang)
        x = torch.cat([u_emb, l_emb, user_continuous], dim=-1)
        return self.network(x)

class MovieTower(nn.Module):
    def __init__(self, num_movies: int, num_languages: int, num_content_types: int, num_genres: int, embedding_dim: int = 64, output_dim: int = 128):
        super().__init__()
        # Matches state_dict: 'movie_embedding', 'lang_embedding', 'content_type_embedding'
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.lang_embedding = nn.Embedding(num_languages + 1, 16)
        self.content_type_embedding = nn.Embedding(num_content_types + 1, 8)

        # Total Input: 64 (movie) + 16 (lang) + 8 (type) + 3 (cont) + num_genres (23) = 114
        input_dim = embedding_dim + 16 + 8 + 3 + num_genres

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), # Matches [256, 114]
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
        m_emb = self.movie_embedding(movie_id)
        l_emb = self.lang_embedding(movie_language)
        ct_emb = self.content_type_embedding(movie_content_type)
        x = torch.cat([m_emb, l_emb, ct_emb, movie_continuous, movie_genres], dim=-1)
        return self.network(x)

class TwoTowerModel(nn.Module):
    def __init__(self, user_tower: UserTower, movie_tower: MovieTower):
        super().__init__()
        self.user_tower = user_tower
        self.movie_tower = movie_tower

    def forward(self, batch: Dict) -> torch.Tensor:
        user_emb = self.get_user_embedding(batch)
        movie_emb = self.get_movie_embedding(batch)
        return (user_emb * movie_emb).sum(dim=-1)

    def get_user_embedding(self, batch: Dict) -> torch.Tensor:
        res = self.user_tower(batch["user_id"], batch["user_continuous"], batch["user_lang"])
        return nn.functional.normalize(res, dim=-1)

    def get_movie_embedding(self, batch: Dict) -> torch.Tensor:
        res = self.movie_tower(batch["movie_id"], batch["movie_continuous"], batch["movie_language"], batch["movie_content_type"], batch["movie_genres"])
        return nn.functional.normalize(res, dim=-1)