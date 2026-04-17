import torch
import pandas as pd
import pickle
import numpy as np
from twoTowerModel import TwoTowerModel, UserTower, MovieTower
from infer import RecommendationEngine
from preProcess import preprocess_movies

def recommend_from_history(engine, movies_df, history_dict, top_k=10):
    """
    Calculates a virtual user profile based on a list of watched movies.
    history_dict: {'movie-id': rating_out_of_5}
    """
    watched_ids = list(history_dict.keys())
    
    # Filter DB for movies in the user's history
    history_movies = movies_df[movies_df['movie_id'].isin(watched_ids)].copy()
    
    if history_movies.empty:
        print("❌ Error: None of the movies in your history were found in the database.")
        return []

    # 1. Calculate Aggregated Features
    # Average rating (normalized 0-1)
    avg_rating = (sum(history_dict.values()) / len(history_dict)) / 5.0
    
    # Average popularity of movies they like
    avg_pop = history_movies['popularity_scaled'].mean()
    
    # Preferred language (the one that appears most in their history)
    pref_lang = history_movies['original_language'].mode()[0] if not history_movies.empty else 'en'

    user_features = {
        'avg_rating': avg_rating,
        'num_ratings': len(history_dict),
        'avg_popularity': avg_pop,
        'preferred_language': pref_lang
    }

    print(f"\n--- Virtual User Profile ---")
    print(f"Items in History: {len(history_dict)}")
    print(f"Top Language:    {pref_lang}")
    print(f"Avg Popularity:  {avg_pop:.4f}")
    print(f"----------------------------")

    # 2. Get recommendations, excluding what they've already seen
    return engine.recommend(
        user_id="unknown_new_user", 
        user_features=user_features, 
        top_k=top_k,
        exclude_seen=watched_ids 
    )

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load requirements
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # Initialize Model
    user_tower = UserTower(num_users=vocab.num_users, num_languages=vocab.num_languages)
    movie_tower = MovieTower(
        num_movies=vocab.num_movies, 
        num_languages=vocab.num_languages,
        num_content_types=vocab.num_content_types,
        num_genres=vocab.num_genres
    )
    model = TwoTowerModel(user_tower, movie_tower).to(device)
    model.load_state_dict(torch.load("best_two_tower.pt", map_location=device, weights_only=True))
    model.eval()

    # Load and Preprocess Movie Data
    movies_df = pd.read_parquet("movies.parquet")
    movies_df = preprocess_movies(movies_df)

    # Setup Engine
    engine = RecommendationEngine(model, vocab, device=device)
    print("Building movie index...")
    engine.build_movie_index(movies_df)

    # ---------------------------------------------------------
    # TEST CASE: Define a custom history for a brand new user
    # ---------------------------------------------------------
    # Example: A user who loves Sci-Fi and Animation
    custom_history = {
        'spider-man-into-the-spider-verse': 5.0,
        'interstellar': 5.0,
        'wall-e': 4.5,
        'the-matrix': 4.0,
        'spirited-away': 5.0
    }

    print(f"\nGenerating recommendations based on {len(custom_history)} movies...")
    recommendations = recommend_from_history(engine, movies_df, custom_history, top_k=5)

    print("\n🚀 Top Recommendations for you:")
    for i, rec in enumerate(recommendations, 1):
        # Find the title for display
        movie_row = movies_df[movies_df['movie_id'] == rec['movie_id']]
        title = movie_row['title'].iloc[0] if not movie_row.empty else "Unknown Title"
        print(f"{i}. {title} (Score: {rec['score']:.4f})")

if __name__ == "__main__":
    main()