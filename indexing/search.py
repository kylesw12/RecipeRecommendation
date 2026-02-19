import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import indexer
import preprocess

def normalize_query(q: str) -> str:
    q = q.lower()
    q = re.sub(r"[^a-z\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def search(query: str, df: pd.DataFrame, inverted_index: dict, vectorizer: TfidfVectorizer, tfidf_matrix, k: int = 10):

    q_norm = normalize_query(query)
    if not q_norm:
        return []

    query_tokens = q_norm.split()

    # ingredient filter
    candidates = set()
    for t in query_tokens:
        if t in inverted_index:
            candidates |= set(inverted_index[t])

    # Map RecipeId -> row index for fast subsetting
    # (Build once and reuse if you want performance)
    rid_to_row = {rid: i for i, rid in enumerate(df["RecipeId"].to_numpy())}

    # If we found ingredient candidates, restrict ranking to them
    if candidates:
        candidate_rows = [rid_to_row[rid] for rid in candidates if rid in rid_to_row]
        if not candidate_rows:
            candidate_rows = list(range(len(df)))
        sub_matrix = tfidf_matrix[candidate_rows]
    else:
        candidate_rows = None
        sub_matrix = tfidf_matrix

    # --- TF-IDF ranking ---
    q_vec = vectorizer.transform([q_norm])
    sims = cosine_similarity(q_vec, sub_matrix).ravel()

    ingredient_scores = []

    for row_idx in (candidate_rows if candidate_rows is not None else range(len(df))):
        recipe_ingredients = set(df.iloc[row_idx]["parsed_ingredients"])
        overlap = len(recipe_ingredients.intersection(query_tokens))
        ingredient_scores.append(overlap)

    ingredient_scores = np.array(ingredient_scores, dtype=float)

    # Normalize ingredient score
    if ingredient_scores.max() > 0:
        ingredient_scores /= ingredient_scores.max()

    # Hybrid final scores
    text_weight = 0.7
    ingredient_weight = 0.3
    
    final_scores = (
        text_weight * sims + ingredient_weight * ingredient_scores
    )


    # top-k indices within the subset
    top_local = np.argsort(-final_scores)[:k]

    results = []
    for j in top_local:
        row_i = candidate_rows[j]
        results.append({
            "RecipeId": int(df.iloc[row_i]["RecipeId"]),
            "Name": str(df.iloc[row_i]["Name"]),
            "text_score": float(sims[j]),
            "ingredient_score": float(ingredient_scores[j]),
            "final_score": float(final_scores[j])
        })
    return results

if __name__ == "__main__":
    q = input("Enter your query: ")
    df = preprocess.load_recipes()
    df = preprocess.preprocess_recipes(df)

    # # Ingredient inverted index
    # inverted_index = indexer.build_inverted_index(df)
    # indexer.save_index(inverted_index)
    # #tf-idf indexer
    # vectorizer, tfidf_matrix = indexer.build_tfidf_index(df)
    # indexer.save_tfidf_index(vectorizer, tfidf_matrix)

    if not os.path.exists("data/inverted_index.pkl") or not os.path.exists("data/tfidf_index.pkl"):
        print("Building indices...")
        indexer.build_and_save_index()
        vectorizer, tfidf_matrix = indexer.build_tfidf_index(df)
        indexer.save_tfidf_index(vectorizer, tfidf_matrix)

    inverted_index = indexer.load_index()
    vectorizer, tfidf_matrix = indexer.load_tfidf_index()
    
    results = search(q, df, inverted_index, vectorizer, tfidf_matrix, k=10)

    print("\nTOP RESULTS:\n")
    for r in results:
        print(
            f"{r['final_score']:.4f} | "
            f"{r['RecipeId']} | "
            f"{r['Name']} "
            f"(text={r['text_score']:.3f}, "
            f"ing={r['ingredient_score']:.3f})"
        )