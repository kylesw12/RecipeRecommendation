from collections import defaultdict
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import load_recipes, preprocess_recipes

def build_inverted_index(df):
    inverted_index = defaultdict(set)

    for _, row in df.iterrows():
        recipe_id = row["RecipeId"]
        for ingredient in row["parsed_ingredients"]:
            if ingredient:
                inverted_index[ingredient].add(recipe_id)

    return inverted_index

def build_text_corpus(df):
    corpus = []
    for _, row in df.iterrows():
        text = " ".join([
            str(row['Name']),
            str(row['Description']),
            str(row['RecipeCategory']),
            " ".join(row['parsed_keywords'])
        ])
        corpus.append(text)
    return corpus

def build_tfidf_index(df):
    corpus = build_text_corpus(df)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

def save_tfidf_index(vectorizer, tfidf_matrix, path="data/tfidf_index.pkl"):
    with open(path, "wb") as f:
        pickle.dump((vectorizer, tfidf_matrix), f)

def load_tfidf_index(path="data/tfidf_index.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_index(index, path="data/inverted_index.pkl"):
    with open(path, "wb") as f:
        pickle.dump(dict(index), f)

def build_and_save_index():
    df = load_recipes()
    df = preprocess_recipes(df)
    index = build_inverted_index(df)
    save_index(index)

def load_index(path="data/inverted_index.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    df = load_recipes()
    df = preprocess_recipes(df)

    # Ingredient inverted index
    inverted_index = build_inverted_index(df)
    save_index(inverted_index)

    print("garlic" in inverted_index)
    if "garlic" in inverted_index:
        print("Number of recipes with garlic:", len(inverted_index["garlic"]))

    # TF-IDF index
    vectorizer, tfidf_matrix = build_tfidf_index(df)
    save_tfidf_index(vectorizer, tfidf_matrix)
    print("TF-IDF index built and saved!")
