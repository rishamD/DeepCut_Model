import os
import torch
import pickle
import pandas as pd
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from preProcess import preprocess_movies, preprocess_ratings, build_user_features
from enode import VocabBuilder
from twoTowerModel import TwoTowerModel, UserTower, MovieTower
from trainingLoop import train


class MovieRatingDataset(Dataset):
    def __init__(self, ratings_df, movies_df, user_features_df, vocab):
        assert user_features_df["user_id"].is_unique, (
            "user_features_df has duplicate user_ids — merge will explode row count."
        )
        if not movies_df["movie_id"].is_unique:
            dupes = movies_df["movie_id"].duplicated().sum()
            tqdm.write(
                f"   ⚠️  movies_df has {dupes:,} duplicate movie_ids — "
                "should have been fixed in preprocess_movies. Deduplicating now."
            )
            movies_df = (
                movies_df.sort_values("popularity", ascending=False)
                .drop_duplicates("movie_id", keep="first")
            )

        tqdm.write("   Merging dataframes...")
        t = time.time()
        movie_cols = [
            "movie_id",
            "popularity_scaled",
            "runtime_scaled",
            "year_scaled",
            "original_language",
            "genres",
        ]
        merged = ratings_df.merge(movies_df[movie_cols], on="movie_id", how="left")
        merged = merged.merge(user_features_df, on="user_id", how="left")

        assert len(merged) == len(ratings_df), (
            f"Merge changed row count: {len(ratings_df):,} → {len(merged):,}. "
            "Check for duplicate movie_ids or user_ids in lookup tables."
        )
        tqdm.write(f"   Merges done: {time.time()-t:.1f}s | Shape: {merged.shape}")

        tqdm.write("   Encoding IDs...")
        t = time.time()
        user_map = {v: i for i, v in enumerate(vocab.user_encoder.classes_)}
        movie_map = {v: i for i, v in enumerate(vocab.movie_encoder.classes_)}
        self.user_ids = torch.from_numpy(
            merged["user_id"].map(user_map).values.astype(np.int32)
        ).long()
        self.movie_ids = torch.from_numpy(
            merged["movie_id"].map(movie_map).values.astype(np.int32)
        ).long()
        self.labels = torch.from_numpy(
            merged["rating_norm"].values.astype(np.float32)
        ).float()
        tqdm.write(f"   IDs encoded: {time.time()-t:.1f}s")

        tqdm.write("   Encoding features...")
        t = time.time()

        self.user_continuous = torch.from_numpy(
            np.stack(
                [
                    merged["avg_rating"].fillna(0).values.astype(np.float32),
                    merged["num_ratings"].fillna(0).values.astype(np.float32),
                    merged["avg_popularity"].fillna(0).values.astype(np.float32),
                ],
                axis=1,
            )
        ).float()

        self.user_lang = torch.from_numpy(
            vocab.safe_encode(
                vocab.language_encoder,
                merged["preferred_language"].fillna("unknown"),
            ).astype(np.int32)
        ).long()

        self.movie_continuous = torch.from_numpy(
            np.stack(
                [
                    merged["popularity_scaled"].fillna(0).values.astype(np.float32),
                    merged["runtime_scaled"].fillna(0).values.astype(np.float32),
                    merged["year_scaled"].fillna(0).values.astype(np.float32),
                ],
                axis=1,
            )
        ).float()

        self.movie_language = torch.from_numpy(
            vocab.safe_encode(
                vocab.language_encoder,
                merged["original_language"].fillna("unknown"),
            ).astype(np.int32)
        ).long()

        tqdm.write(f"   Features encoded: {time.time()-t:.1f}s")

        tqdm.write("   Building genre matrix...")
        t = time.time()
        num_genres = len(vocab.genre_vocab)

        unique_movies = (
            movies_df[["movie_id", "genres"]]
            .drop_duplicates("movie_id")
            .reset_index(drop=True)
        )
        genre_matrix = np.zeros((len(unique_movies), num_genres), dtype=np.float32)

        for i, genres in enumerate(unique_movies["genres"]):
            if isinstance(genres, list):
                for g in genres:
                    if g in vocab.genre_vocab:
                        genre_matrix[i, vocab.genre_vocab[g]] = 1.0

        tqdm.write(f"   Genre matrix built: {time.time()-t:.1f}s")

        tqdm.write("   Mapping genres to ratings...")
        t = time.time()
        movie_id_to_idx = {mid: i for i, mid in enumerate(unique_movies["movie_id"])}
        rating_movie_indices = (
            merged["movie_id"].map(movie_id_to_idx).fillna(0).astype(int).values
        )
        self.movie_genres = torch.from_numpy(
            genre_matrix[rating_movie_indices]
        ).float()
        tqdm.write(f"   Genre mapping done: {time.time()-t:.1f}s")
        tqdm.write("   ✅ Dataset ready")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "movie_id": self.movie_ids[idx],
            "user_continuous": self.user_continuous[idx],
            "user_lang": self.user_lang[idx],
            "movie_continuous": self.movie_continuous[idx],
            "movie_language": self.movie_language[idx],
            "movie_genres": self.movie_genres[idx],
            "label": self.labels[idx],
        }


def main():
    BATCH_SIZE = 65536
    EPOCHS = 20
    LR = 1e-3
    VAL_SPLIT = 0.1
    NUM_WORKERS = 0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_CACHE = "dataset_cache.pkl"

    print(f"🖥️  Training on: {DEVICE}")

    overall = tqdm(total=7, desc="Overall Progress", position=0)

    # ── 1. Load Raw Data ──────────────────────────────────────────────────
    overall.set_description("📦 Loading data")
    t = time.time()
    movies_df = pd.read_parquet("movies.parquet")
    ratings_df = pd.read_parquet("ratings.parquet")
    tqdm.write(
        f"   ✅ Movies: {len(movies_df):,} | Ratings: {len(ratings_df):,} | "
        f"{time.time()-t:.1f}s"
    )
    overall.update(1)

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    overall.set_description("⚙️  Preprocessing")
    with tqdm(total=3, desc="   Preprocessing", position=1, leave=False) as pbar:
        t = time.time()
        movies_df = preprocess_movies(movies_df, fit=True)
        tqdm.write(f"   preprocess_movies: {time.time()-t:.1f}s")
        pbar.update(1)

        t = time.time()
        ratings_df = preprocess_ratings(ratings_df)
        tqdm.write(f"   preprocess_ratings: {time.time()-t:.1f}s")
        pbar.update(1)

        t = time.time()
        user_features_df = build_user_features(ratings_df, movies_df)
        tqdm.write(f"   build_user_features: {time.time()-t:.1f}s")
        pbar.update(1)

    tqdm.write("   ✅ Preprocessing done, scalers.pkl saved")
    overall.update(1)

    # ── 3. Build & Save Vocab ─────────────────────────────────────────────
    overall.set_description("📖 Building vocab")
    with tqdm(total=2, desc="   Vocab", position=1, leave=False) as pbar:
        t = time.time()
        vocab = VocabBuilder()
        vocab.fit(movies_df, ratings_df, user_features_df)
        pbar.update(1)
        with open("vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
        pbar.update(1)
    tqdm.write(f"   ✅ vocab.pkl saved | {time.time()-t:.1f}s")
    tqdm.write(
        f"   Users: {vocab.num_users:,} | Movies: {vocab.num_movies:,} | "
        f"Genres: {vocab.num_genres} | Languages: {vocab.num_languages}"
    )
    overall.update(1)

    # ── 4. Filter ratings to only known movies & users ────────────────────
    overall.set_description("🔍 Filtering ratings")
    t = time.time()
    known_movies = set(vocab.movie_encoder.classes_)
    known_users = set(vocab.user_encoder.classes_)
    ratings_df = ratings_df[
        ratings_df["movie_id"].isin(known_movies)
        & ratings_df["user_id"].isin(known_users)
    ].reset_index(drop=True)
    tqdm.write(
        f"   ✅ Ratings after filter: {len(ratings_df):,} | {time.time()-t:.1f}s"
    )
    overall.update(1)

    # ── 5. Build Dataset & Loaders ────────────────────────────────────────
    overall.set_description("🗂️  Building dataset")
    with tqdm(total=3, desc="   Dataset", position=1, leave=False) as pbar:
        t = time.time()

        if os.path.exists(DATASET_CACHE):
            tqdm.write(f"   💾 Loading cached dataset from {DATASET_CACHE}...")
            with open(DATASET_CACHE, "rb") as f:
                dataset = pickle.load(f)
            tqdm.write(f"   ✅ Dataset loaded from cache: {time.time()-t:.1f}s")
        else:
            dataset = MovieRatingDataset(
                ratings_df, movies_df, user_features_df, vocab
            )
            tqdm.write(f"   💾 Saving dataset to {DATASET_CACHE}...")
            with open(DATASET_CACHE, "wb") as f:
                pickle.dump(dataset, f)
            tqdm.write(f"   ✅ Dataset cached: {time.time()-t:.1f}s")

        pbar.update(1)

        val_size = int(len(dataset) * VAL_SPLIT)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        pbar.update(1)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )
        pbar.update(1)

    tqdm.write(f"   ✅ Train: {train_size:,} | Val: {val_size:,}")
    overall.update(1)

    # ── 6. Build Model ────────────────────────────────────────────────────
    overall.set_description("🏗️  Building model")
    t = time.time()
    user_tower = UserTower(
        num_users=vocab.num_users,
        num_languages=vocab.num_languages,
    )
    movie_tower = MovieTower(
        num_movies=vocab.num_movies,
        num_languages=vocab.num_languages,
        num_genres=vocab.num_genres,
    )
    model = TwoTowerModel(user_tower, movie_tower)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tqdm.write(f"   ✅ Trainable parameters: {total_params:,} | {time.time()-t:.1f}s")
    overall.update(1)

    # ── 7. Train ──────────────────────────────────────────────────────────
    overall.set_description("🚀 Training")
    tqdm.write("\n🚀 Starting training loop...\n")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        device=DEVICE,
    )
    overall.update(1)
    overall.set_description("✅ Done")
    overall.close()

    print("\n✅ Training complete! Saved: best_two_tower.pt, vocab.pkl, scalers.pkl")


if __name__ == "__main__":
    main()