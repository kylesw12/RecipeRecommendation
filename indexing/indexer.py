from collections import defaultdict
import pickle

from preprocess import load_recipes, preprocess_recipes

def build_inverted_index(df):
    inverted_index = defaultdict(set)

    for _, row in df.iterrows():
        recipe_id = row["RecipeId"]
        for ingredient in row["parsed_ingredients"]:
            if ingredient:
                inverted_index[ingredient].add(recipe_id)

    return inverted_index

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
    build_and_save_index()

    index = load_index()

    print("garlic" in index)
    if "garlic" in index:
        print("Number of recipes with garlic:", len(index["garlic"]))
