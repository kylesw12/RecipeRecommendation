import pandas as pd
import ast
import re

# Loads recipe data
def load_recipes(path="data/recipes.csv"):
    df = pd.read_csv(path)
    return df

# Parses the list of ingredients 
# ie. c("blueberries", "granulated sugar") -> ["blueberries", "granulated sugar"]


def parse_ingredients(ingredients_str):
    if pd.isna(ingredients_str):
        return []
    ingredients_str = ingredients_str.strip()
    ingredients_str = ingredients_str[2:-1]

    parts = ingredients_str.split('", "')
    parts = [p.replace('"', '') for p in parts]
    return parts

# Normalizes ingredient names by converting to lowercase and removing non-alphabetic characters
def normalize_ingredient(ingredient):
    ingredient = ingredient.lower()
    ingredient = re.sub(r"[^a-z\s]", "", ingredient)
    return ingredient.strip()

# apply preprocessing to all recipes in the dataframe
def preprocess_recipes(df):
    df["parsed_ingredients"] = df["RecipeIngredientParts"].apply(
        lambda x: [
            normalize_ingredient(i)
            for i in parse_ingredients(x)
        ]
    )
    return df

if __name__ == "__main__":
    df = load_recipes()
    df = preprocess_recipes(df)

    for i in range(5):
        print(df.loc[i, "Name"])
        print(df.loc[i, "parsed_ingredients"])
        print()



