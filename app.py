from flask import Flask, render_template, request
from indexing import preprocess
from indexing import indexer
from indexing.search import search
import os
import ast
import re

app = Flask(__name__)

df = preprocess.load_recipes()
df = preprocess.preprocess_recipes(df)

if (not os.path.exists("data/inverted_index.pkl")) or (not os.path.exists("data/tfidf_vectorizer.pkl")):
    indexer.build_and_save_index()
    vectorizer, tfidf_matrix = indexer.build_tfidf_index(df)
    indexer.save_tfidf_index(vectorizer, tfidf_matrix)

inverted_index = indexer.load_index()
vectorizer, tfidf_matrix = indexer.load_tfidf_index()


@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    query = ""

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            results = search(query, df, inverted_index, vectorizer, tfidf_matrix, k=10)

    return render_template("index.html", results=results, query=query)


@app.route("/recipe/<int:recipe_id>")
def recipe_detail(recipe_id):
    recipe = df[df["RecipeId"] == recipe_id]
    if recipe.empty:
        return "Recipe not found", 404

    recipe = recipe.iloc[0]

    def pick(*keys):
        for k in keys:
            if k in recipe.index:
                v = recipe[k]
                if v is not None and str(v).strip() and str(v).lower() != "nan":
                    return v
        return ""

    def clean_token(t):
        s = str(t).strip()
        s = s.strip().strip('"').strip("'")
        if s.lower() in {"na", "nan", "none", "null"}:
            return ""
        return s

    def parse_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            out = [clean_token(t) for t in x]
            return [t for t in out if t]

        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return []

        if s.startswith("c(") and s.endswith(")"):
            s = "[" + s[2:-1] + "]"

        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                out = [clean_token(t) for t in v]
                return [t for t in out if t]
        except:
            pass

        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]

        parts = [clean_token(t) for t in s.split(",")]
        return [t for t in parts if t]

    def to_steps(x):
        s = str(x).strip() if x is not None else ""
        if s == "" or s.lower() == "nan":
            return []
        lst = parse_list(s)
        if len(lst) > 1:
            return lst
        s = s.replace("\\n", "\n")
        return [p.strip() for p in re.split(r"\n\s*\n|\n|(?<=[.!?])\s+", s) if p.strip()]

    description_raw = pick("Description", "RecipeDescription")
    instructions_raw = pick("RecipeInstructions", "Instructions", "InstructionsText")

    parts = parse_list(pick("RecipeIngredientParts", "Ingredients"))
    qtys = parse_list(pick("RecipeIngredientQuantities", "Quantities"))
    units = parse_list(pick("RecipeIngredientUnits", "Units"))

    ingredients_list = []
    for i, p in enumerate(parts):
        q = qtys[i] if i < len(qtys) else ""
        u = units[i] if i < len(units) else ""
        q = clean_token(q)
        u = clean_token(u)
        p = clean_token(p)

        line = " ".join([q, u, p]).strip()
        if line:
            ingredients_list.append(line)

    return render_template(
        "recipe.html",
        recipe=recipe,
        description_steps=to_steps(description_raw),
        instruction_steps=to_steps(instructions_raw),
        ingredients_list=ingredients_list,
        description_raw=str(description_raw) if description_raw is not None else "",
        instructions_raw=str(instructions_raw) if instructions_raw is not None else ""
    )


if __name__ == "__main__":
    app.run(debug=True)
