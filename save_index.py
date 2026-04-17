import torch
import pandas as pd
import pickle
import faiss
from twoTowerModel import TwoTowerModel, UserTower, MovieTower
from infer import RecommendationEngine
from preProcess import preprocess_movies

def export_index():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Vocab
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # Init Model (Ensure this matches your updated architecture)
    u_tower = UserTower(num_users=vocab.num_users, num_languages=vocab.num_languages)
    m_tower = MovieTower(
        num_movies=vocab.num_movies, 
        num_languages=vocab.num_languages,
        num_content_types=vocab.num_content_types,
        num_genres=vocab.num_genres
    )
    model = TwoTowerModel(u_tower, m_tower).to(device)
    model.load_state_dict(torch.load("best_two_tower.pt", map_location=device))
    model.eval()

    # Load Data
    movies_df = pd.read_parquet("movies.parquet")
    movies_df = preprocess_movies(movies_df)

    # Build Index
    engine = RecommendationEngine(model, vocab, device=device)
    engine.build_movie_index(movies_df)

    # SAVE THE WORK
    # 1. Save the FAISS vector index
    faiss.write_index(engine.index, "movie_index.bin")
    # 2. Save the movie ID mapping (so the index knows which vector is which movie)
    with open("movie_ids.pkl", "wb") as f:
        pickle.dump(engine.movie_id_map, f)
    
    print("✅ Exported movie_index.bin and movie_ids.pkl")

if __name__ == "__main__":
    export_index()