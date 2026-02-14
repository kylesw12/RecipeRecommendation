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

    # top-k indices within the subset
    top_local = np.argsort(-sims)[:k]

    results = []
    for j in top_local:
        score = float(sims[j])
        row_i = candidate_rows[j] if candidate_rows is not None else int(j)
        rid = int(df.iloc[row_i]["RecipeId"])
        name = str(df.iloc[row_i]["Name"])
        results.append({"RecipeId": rid, "Name": name, "score": score})

    return results

if __name__ == "__main__":
    q = input()
    df = preprocess.load_recipes()
    df = preprocess.preprocess_recipes(df)

    # Ingredient inverted index
    inverted_index = indexer.build_inverted_index(df)
    indexer.save_index(inverted_index)
    #tf-idf indexer
    vectorizer, tfidf_matrix = indexer.build_tfidf_index(df)
    indexer.save_tfidf_index(vectorizer, tfidf_matrix)
    
    results = search(q, df, inverted_index, vectorizer, tfidf_matrix, k=10)

    for r in results:
        print(r["score"], r["RecipeId"], r["Name"])