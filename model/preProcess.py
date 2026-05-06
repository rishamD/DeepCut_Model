import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


def preprocess_movies(movies_df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    df = movies_df.copy()

    def parse_genres(x):
        if isinstance(x, str):
            import re
            matches = re.findall(r"'([^']+)'", x)
            if matches:
                return [g.strip() for g in matches]
            return [g.strip() for g in x.split(",") if g.strip()]
        if isinstance(x, np.ndarray):
            return [str(g).strip() for g in x.tolist()]
        if isinstance(x, list):
            return [str(g).strip() for g in x]
        return []

    df["genres"] = df["genres"].apply(parse_genres)

    unique_genres = df["genres"].explode().nunique()
    print(f"   ✅ Unique genres after parse: {unique_genres}")
    if unique_genres > 100:
        raise ValueError(
            f"Genre parsing failed — {unique_genres} unique values suggests "
            "lists are not being exploded. Check raw data format."
        )

    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df["runtime"] = df["runtime"].fillna(df["runtime"].median())
    df["year_released"] = pd.to_numeric(df["year_released"], errors="coerce")

    df["year"] = df["year_released"].fillna(
        pd.to_datetime(df["release_date"], errors="coerce").dt.year
    ).fillna(2000)

    df["original_language"] = (
        df["original_language"].fillna("unknown").str.strip().str.lower()
    )

    if fit:
        pop_scaler = MinMaxScaler()
        run_scaler = MinMaxScaler()
        year_scaler = MinMaxScaler()

        df["popularity_scaled"] = pop_scaler.fit_transform(df[["popularity"]])
        df["runtime_scaled"] = run_scaler.fit_transform(df[["runtime"]])
        df["year_scaled"] = year_scaler.fit_transform(df[["year"]])

        joblib.dump(
            {"popularity": pop_scaler, "runtime": run_scaler, "year": year_scaler},
            "scalers.pkl",
        )
        print("✅ Scalers saved to scalers.pkl")
    else:
        scalers = joblib.load("scalers.pkl")
        df["popularity_scaled"] = scalers["popularity"].transform(df[["popularity"]])
        df["runtime_scaled"] = scalers["runtime"].transform(df[["runtime"]])
        df["year_scaled"] = scalers["year"].transform(df[["year"]])

    before = len(df)
    df = df.sort_values("popularity", ascending=False).drop_duplicates(
        "movie_id", keep="first"
    )
    after = len(df)
    if before != after:
        print(f"   ⚠️  Dropped {before - after:,} duplicate movie_ids")

    return df


def preprocess_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    df = ratings_df.copy()
    max_rating = df["rating_val"].max()
    if max_rating == 0:
        raise ValueError("All ratings are 0 — check your ratings data.")
    df["rating_norm"] = df["rating_val"] / max_rating
    df["implicit"] = 1
    return df


def build_user_features(
    ratings_df: pd.DataFrame, movies_df: pd.DataFrame
) -> pd.DataFrame:
    movies_slim = (
        movies_df[["movie_id", "popularity", "original_language"]]
        .sort_values("popularity", ascending=False)
        .drop_duplicates("movie_id", keep="first")
    )

    merged = ratings_df.merge(movies_slim, on="movie_id", how="left")

    assert len(merged) == len(ratings_df), (
        f"build_user_features merge duplicated rows: "
        f"{len(ratings_df):,} → {len(merged):,}. "
        "movies_df has duplicate movie_ids."
    )

    user_features = (
        merged.groupby("user_id")
        .agg(
            avg_rating=("rating_val", "mean"),
            num_ratings=("rating_val", "count"),
            avg_popularity=("popularity", "mean"),
            preferred_language=(
                "original_language",
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown",
            ),
        )
        .reset_index()
    )

    assert user_features["user_id"].is_unique, (
        "build_user_features produced duplicate user_ids after groupby."
    )

    scaler = MinMaxScaler()
    user_features[["avg_rating", "num_ratings", "avg_popularity"]] = (
        scaler.fit_transform(
            user_features[["avg_rating", "num_ratings", "avg_popularity"]]
        )
    )

    joblib.dump(scaler, "user_scaler.pkl")
    print("✅ User scaler saved to user_scaler.pkl")

    return user_features