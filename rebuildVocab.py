import pandas as pd
import numpy as np
import pickle
from enode import VocabBuilder
from preProcess import preprocess_movies

def rebuild_vocab():
    print("--- Vocab Rebuild Tool ---")
    
    # 1. Load the raw parquet files
    try:
        print("Loading data files...")
        movies_df = pd.read_parquet("movies.parquet")
        ratings_df = pd.read_parquet("ratings.parquet")
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        return

    # 2. Apply the official preprocessing
    # This converts 'genres' from strings/NaNs into Python lists 
    # and handles language/content_type fills.
    print("Preprocessing movies...")
    movies_df = preprocess_movies(movies_df)

    # 3. Fit the VocabBuilder using the clean data
    print("Fitting encoders...")
    vocab = VocabBuilder()
    vocab.fit(movies_df, ratings_df)

    # 4. Dimension Verification Math (Target: 114)
    # Based on MovieTower in twoTowerModel.py:
    # input_dim = Movie_Emb(64) + Lang_Emb(16) + Type_Emb(8) + Cont(3) + num_genres
    
    movie_emb_dim = 64
    lang_emb_dim = 16
    type_emb_dim = 8
    continuous_dim = 3
    num_genres = vocab.num_genres  # Should be 23

    calculated_input_dim = (
        movie_emb_dim + 
        lang_emb_dim + 
        type_emb_dim + 
        continuous_dim + 
        num_genres
    )

    print("\n" + "="*30)
    print("DIMENSION REPORT")
    print("="*30)
    print(f"Unique Languages:     {vocab.num_languages}")
    print(f"Unique Content Types: {vocab.num_content_types}")
    print(f"Unique Genres:        {num_genres}")
    print(f"---")
    print(f"MovieTower Input Dim: {calculated_input_dim}")
    print("="*30)

    # 5. Save the file if dimensions match the model weights
    if calculated_input_dim == 114:
        with open("vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
        print("\n✅ SUCCESS: 'vocab.pkl' rebuilt and verified for 114-dim model.")
    else:
        print(f"\n❌ ERROR: Calculated dimension {calculated_input_dim} != 114.")
        print("Possible cause: The parquet data has changed since the model was trained.")

if __name__ == "__main__":
    rebuild_vocab()