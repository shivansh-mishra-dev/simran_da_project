import ast
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

print("=" * 60)
print("  TMDB Movie Recommendation System")
print("  Cosine Similarity | Content-Based | Collaborative | SVD")
print("=" * 60)


# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

movies  = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

print(f"\n[1] Movies loaded  : {movies.shape[0]:,} rows x {movies.shape[1]} cols")
print(f"    Credits loaded : {credits.shape[0]:,} rows x {credits.shape[1]} cols")


# --------------------------------------------------
# 2. Merge on movie id
# --------------------------------------------------

credits.rename(columns={"movie_id": "id"}, inplace=True)

df = movies.merge(credits, on="id", suffixes=("", "_c"))

df.drop(columns=["title_c"], inplace=True)

print(f"\n[2] Merged shape   : {df.shape[0]:,} rows x {df.shape[1]} cols")


# --------------------------------------------------
# 3. Feature Extraction helpers
# --------------------------------------------------

def extract_names(text, field="name", limit=3):
    """Parse JSON-like list string and return up to `limit` name values."""
    try:
        items = ast.literal_eval(text)
        # Replace spaces inside names so they stay as single tokens
        return [item[field].replace(" ", "") for item in items[:limit]]
    except Exception:
        return []


def get_director(crew_text):
    """Return director name from crew JSON string."""
    try:
        for member in ast.literal_eval(crew_text):
            if member.get("job") == "Director":
                return member["name"].replace(" ", "")
    except Exception:
        pass
    return ""


print("\n[3] Extracting genres, keywords, cast, director ...")

df["genres_clean"]   = df["genres"].apply(extract_names)
df["keywords_clean"] = df["keywords"].apply(extract_names)
df["cast_clean"]     = df["cast"].apply(extract_names)
df["director"]       = df["crew"].apply(get_director)


# --------------------------------------------------
# 4. Build content "soup" for each movie
#    (genres + keywords + cast + director — all as tokens)
# --------------------------------------------------

def build_soup(row):
    return " ".join(
        row["genres_clean"]
        + row["keywords_clean"]
        + row["cast_clean"]
        + [row["director"]]
    )


df["soup"] = df.apply(build_soup, axis=1)

# Drop rows with empty soup
df = df[df["soup"].str.strip() != ""].reset_index(drop=True)

print(f"    Soup sample (Avatar): {df[df['title']=='Avatar']['soup'].values[0][:80]} ...")


# --------------------------------------------------
# 5. Content-Based Filtering
#    TF-IDF on the soup -> cosine similarity matrix
# --------------------------------------------------

print("\n[5] Building TF-IDF matrix for Content-Based Filtering ...")

# Pipeline: TF-IDF vectorisation (no fit on test data needed here —
#           we fit on the whole catalogue)
tfidf_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000))
])

tfidf_matrix = tfidf_pipeline.fit_transform(df["soup"])

print(f"    TF-IDF matrix shape : {tfidf_matrix.shape}")


# --------------------------------------------------
# 6. Cosine Similarity — Content-Based
# --------------------------------------------------

print("\n[6] Computing Cosine Similarity (content-based) ...")

cosine_sim_content = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(f"    Cosine similarity matrix shape : {cosine_sim_content.shape}")


# --------------------------------------------------
# 7. Content-Based Recommendation function
# --------------------------------------------------

# Build title -> index lookup
title_to_idx = pd.Series(df.index, index=df["title"]).drop_duplicates()


def recommend_content_based(title, top_n=10):
    """
    Return top-N movie recommendations for a given title
    using content-based cosine similarity.
    """
    if title not in title_to_idx:
        return f"Movie '{title}' not found in dataset."

    idx        = title_to_idx[title]
    sim_scores = list(enumerate(cosine_sim_content[idx]))

    # Sort by similarity score descending, skip the movie itself (score=1.0)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: top_n + 1]

    movie_indices = [i[0] for i in sim_scores]
    scores        = [round(i[1], 4) for i in sim_scores]

    result = df["title"].iloc[movie_indices].reset_index(drop=True)
    return pd.DataFrame({"Recommended Movie": result, "Cosine Score": scores})


print("\n[7] Content-Based Recommendations for 'Avatar':")
print(recommend_content_based("Avatar").to_string(index=False))

print("\n    Content-Based Recommendations for 'The Dark Knight':")
print(recommend_content_based("The Dark Knight").to_string(index=False))


# --------------------------------------------------
# 8. Collaborative Filtering — Simulate User-Item Matrix
#    (Real user data is unavailable; we simulate 200 users
#     whose ratings are vote_average + small personal noise)
# --------------------------------------------------

print("\n[8] Simulating User-Item Rating Matrix for Collaborative Filtering ...")

np.random.seed(42)

NUM_USERS  = 200
NUM_MOVIES = 300   # use top-300 popular movies to keep matrix manageable

# Select top popular movies
top_movies = (df.nlargest(NUM_MOVIES, "vote_count")
                .reset_index(drop=True))

# Simulate: each user rates ~40% of movies; others are 0 (not rated)
ratings = np.zeros((NUM_USERS, NUM_MOVIES))
for u in range(NUM_USERS):
    # Randomly pick ~40% of movies this user has rated
    rated_mask = np.random.rand(NUM_MOVIES) < 0.4
    base       = top_movies["vote_average"].values          # 0–10 base score
    noise      = np.random.normal(0, 0.8, NUM_MOVIES)      # personal preference
    user_score = np.clip(base + noise, 0, 10)
    ratings[u, rated_mask] = user_score[rated_mask]

print(f"    Rating matrix shape : {ratings.shape}  (users x movies)")
print(f"    Sparsity            : {(ratings == 0).mean():.1%} entries are unrated")


# --------------------------------------------------
# 9. User-User Cosine Similarity (Collaborative Filtering)
# --------------------------------------------------

print("\n[9] Computing User-User Cosine Similarity ...")

# Normalise each user's rating row so cosine similarity focuses on
# preferences rather than absolute rating magnitudes
scaler = MinMaxScaler()
ratings_norm = scaler.fit_transform(ratings)

user_sim_matrix = cosine_similarity(ratings_norm)   # shape: (200, 200)

print(f"    User similarity matrix shape : {user_sim_matrix.shape}")


# --------------------------------------------------
# 10. Collaborative Filtering — predict ratings for a user
# --------------------------------------------------

def collaborative_recommend(user_id, top_n=10, n_similar_users=20):
    """
    Predict unwatched movie ratings for `user_id` using weighted
    average ratings from the top-N most similar users.
    Returns a DataFrame of top recommended movies.
    """
    # Similarities for this user with all other users
    sim_scores = user_sim_matrix[user_id]

    # Find top similar users (exclude the user themselves)
    similar_users = np.argsort(sim_scores)[::-1][1: n_similar_users + 1]

    # Movies this user has NOT yet rated
    unrated_mask   = ratings[user_id] == 0
    unrated_indices = np.where(unrated_mask)[0]

    predicted_scores = []
    for movie_idx in unrated_indices:
        # Weighted average: similarity * rating from similar users who rated it
        weights  = sim_scores[similar_users]
        r        = ratings[similar_users, movie_idx]
        rated    = r > 0   # only include users who actually rated this movie

        if rated.sum() == 0:
            predicted_scores.append((movie_idx, 0.0))
        else:
            w_sum    = weights[rated].sum()
            pred     = np.dot(weights[rated], r[rated]) / (w_sum + 1e-9)
            predicted_scores.append((movie_idx, round(pred, 4)))

    # Sort by predicted score
    predicted_scores.sort(key=lambda x: x[1], reverse=True)
    top_recs = predicted_scores[:top_n]

    movie_names = [top_movies["title"].iloc[i] for i, _ in top_recs]
    pred_vals   = [s for _, s in top_recs]

    return pd.DataFrame({
        "Recommended Movie" : movie_names,
        "Predicted Rating"  : pred_vals
    })


print("\n[10] Collaborative Filtering Recommendations for User 0:")
print(collaborative_recommend(0).to_string(index=False))

print("\n     Collaborative Filtering Recommendations for User 42:")
print(collaborative_recommend(42).to_string(index=False))


# --------------------------------------------------
# 11. Matrix Operations — SVD (Singular Value Decomposition)
#     Decompose the rating matrix to find latent factors
#     (used internally by many real recommender systems, e.g. Netflix)
# --------------------------------------------------

print("\n[11] Matrix Operations — SVD on rating matrix ...")

N_COMPONENTS = 20   # latent factors

svd_pipeline = Pipeline([
    ("svd", TruncatedSVD(n_components=N_COMPONENTS, random_state=42))
])

# U matrix: users in latent space  (200 x 20)
U = svd_pipeline.fit_transform(ratings_norm)

# Sigma (singular values)
sigma = svd_pipeline.named_steps["svd"].singular_values_

# V^T matrix: movies in latent space  (20 x 300)
Vt = svd_pipeline.named_steps["svd"].components_

print(f"    U  matrix (users  x latent) : {U.shape}")
print(f"    S  singular values           : {sigma.shape}  — top 5: {sigma[:5].round(2)}")
print(f"    Vt matrix (latent x movies) : {Vt.shape}")

# Reconstruct rating matrix from latent factors
ratings_reconstructed = np.dot(U, np.dot(np.diag(sigma), Vt))

# Reconstruction error
reconstruction_error = np.linalg.norm(ratings_norm - ratings_reconstructed, "fro")
explained_variance   = svd_pipeline.named_steps["svd"].explained_variance_ratio_.sum()

print(f"\n    Reconstruction Frobenius error : {reconstruction_error:.4f}")
print(f"    Variance explained by {N_COMPONENTS} factors : {explained_variance:.2%}")


# --------------------------------------------------
# 12. SVD-Based Recommendation
#     Movies recommended via dot-product in latent space
# --------------------------------------------------

def svd_recommend(user_id, top_n=10):
    """
    Use latent factor vectors (from SVD) to predict ratings for
    all movies and recommend top-N unrated ones.
    """
    predicted = ratings_reconstructed[user_id]

    # Only recommend movies not yet rated by this user
    unrated = (ratings[user_id] == 0)
    predicted_unrated = predicted.copy()
    predicted_unrated[~unrated] = -np.inf   # mask rated movies

    top_indices = np.argsort(predicted_unrated)[::-1][:top_n]
    movie_names = [top_movies["title"].iloc[i] for i in top_indices]
    pred_vals   = [round(predicted_unrated[i], 4) for i in top_indices]

    return pd.DataFrame({
        "Recommended Movie" : movie_names,
        "SVD Predicted Score": pred_vals
    })


print("\n[12] SVD-Based Recommendations for User 0:")
print(svd_recommend(0).to_string(index=False))


# --------------------------------------------------
# 13. Item-Item Cosine Similarity (using SVD latent space)
#     Compare movies in latent factor space — often more
#     accurate than raw feature cosine similarity.
# --------------------------------------------------

print("\n[13] Item-Item Cosine Similarity in SVD latent space ...")

# Movie vectors are rows of Vt^T = columns of Vt => shape (300, 20)
movie_latent_vectors = Vt.T   # (300 movies, 20 latent factors)

item_sim_matrix = cosine_similarity(movie_latent_vectors)  # (300, 300)

print(f"    Item similarity matrix shape : {item_sim_matrix.shape}")

item_title_to_idx = pd.Series(
    top_movies.index, index=top_movies["title"]
).drop_duplicates()


def item_item_recommend(title, top_n=10):
    """
    Recommend movies similar to `title` using item-item cosine
    similarity computed in the SVD latent space.
    """
    if title not in item_title_to_idx:
        return f"'{title}' not in top-{NUM_MOVIES} movies."

    idx        = item_title_to_idx[title]
    sim_scores = list(enumerate(item_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: top_n + 1]

    movie_names = [top_movies["title"].iloc[i] for i, _ in sim_scores]
    scores      = [round(s, 4) for _, s in sim_scores]

    return pd.DataFrame({"Recommended Movie": movie_names, "Latent Cosine Score": scores})


print("\n     Item-Item Recommendations for 'Avatar':")
print(item_item_recommend("Avatar").to_string(index=False))


# --------------------------------------------------
# 14. Summary Report
# --------------------------------------------------

print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"  Dataset             : {len(df):,} movies")
print(f"  TF-IDF vocab size   : {tfidf_pipeline.named_steps['tfidf'].vocabulary_.__len__():,} terms")
print(f"  Content sim matrix  : {cosine_sim_content.shape[0]:,} x {cosine_sim_content.shape[1]:,}")
print(f"  Simulated users     : {NUM_USERS}")
print(f"  Rating matrix       : {ratings.shape[0]} x {ratings.shape[1]}")
print(f"  SVD latent factors  : {N_COMPONENTS}")
print(f"  Variance explained  : {explained_variance:.2%}")
print(f"  SVD recon. error    : {reconstruction_error:.4f}")
print()
print("  Algorithms Used:")
print("    [1] TF-IDF Vectorisation          -> Content feature extraction")
print("    [2] Cosine Similarity (content)   -> Content-Based Filtering")
print("    [3] Cosine Similarity (user-user) -> Collaborative Filtering")
print("    [4] Truncated SVD                 -> Matrix Factorisation")
print("    [5] Cosine Similarity (item-item) -> Latent-Space Item Filtering")
print()
print("[DONE]")

