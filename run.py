import torch
import pandas as pd
import pickle
from twoTowerModel import TwoTowerModel, UserTower, MovieTower
from infer import RecommendationEngine
from preProcess import preprocess_movies

def test_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Vocab
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # 2. Re-init Model Architecture (Must match your 114 dim)
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

    # 3. Load and PREPROCESS Data
    movies_df = pd.read_parquet("movies.parquet")
    movies_df = preprocess_movies(movies_df)

    # 4. Engine
    engine = RecommendationEngine(model, vocab, device=device)
    engine.build_movie_index(movies_df)

    # 5. Recommendation
    test_user_features = {
        'avg_rating': 0.8,
        'num_ratings': 20,
        'avg_popularity': 10.0,
        'preferred_language': 'en'
    }
    
    print("\nGenerating recommendations...")
    recs = engine.recommend(user_id="any-id-works-now", user_features=test_user_features, top_k=5)
    
    for i, r in enumerate(recs, 1):
        title = movies_df[movies_df['movie_id'] == r['movie_id']]['title'].iloc[0]
        print(f"{i}. {title} (Score: {r['score']:.4f})")

if __name__ == "__main__":
    test_model()