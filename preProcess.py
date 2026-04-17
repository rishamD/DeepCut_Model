import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import ast

# ─── 1.1 Load & Merge Data ───────────────────────────────────────────────────

def load_and_merge(movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
    df = ratings_df.merge(movies_df, on="movie_id", how="left")
    return df


# ─── 1.2 Feature Engineering ─────────────────────────────────────────────────

def preprocess_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
    df = movies_df.copy()

    # Improved genre parsing for Parquet/NumPy arrays
    def parse_genres(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except:
                return []
        # If it's already a list or numpy array, return it as a list
        if isinstance(x, (list, np.ndarray)):
            return list(x)
        return []

    df["genres"] = df["genres"].apply(parse_genres)

    # ... rest of your scaling logic ...
    scaler = MinMaxScaler()
    # Ensure numerical columns don't have NaNs before scaling
    df["popularity"] = df["popularity"].fillna(0)
    df["runtime"] = df["runtime"].fillna(df["runtime"].median())
    
    df["popularity_scaled"] = scaler.fit_transform(df[["popularity"]])
    df["runtime_scaled"] = scaler.fit_transform(df[["runtime"]])

    # Extract year
    df["year"] = df["year_released"].fillna(
        pd.to_datetime(df["release_date"], errors="coerce").dt.year
    ).fillna(df["year_released"].median() if "year_released" in df else 2000)
    
    df["year_scaled"] = scaler.fit_transform(df[["year"]])

    return df


def preprocess_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    df = ratings_df.copy()

    # Normalize rating to [0, 1]
    df["rating_norm"] = df["rating_val"] / df["rating_val"].max()

    # Binary implicit feedback (watched = 1)
    df["implicit"] = 1

    return df


def build_user_features(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate user-level features from rating history."""
    merged = ratings_df.merge(movies_df, on="movie_id", how="left")

    user_features = merged.groupby("user_id").agg(
        avg_rating=("rating_val", "mean"),
        num_ratings=("rating_val", "count"),
        avg_popularity=("popularity", "mean"),
        preferred_language=("original_language", lambda x: x.mode()[0]),
    ).reset_index()

    # Normalize
    scaler = MinMaxScaler()
    user_features[["avg_rating", "num_ratings", "avg_popularity"]] = (
        scaler.fit_transform(
            user_features[["avg_rating", "num_ratings", "avg_popularity"]]
        )
    )

    return user_features
