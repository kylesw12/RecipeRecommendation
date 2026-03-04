from flask import Flask, render_template, request
from indexing import preprocess
from indexing import indexer
from indexing.search import search
import os

app = Flask(__name__)

# Load data and indices at startup
df = preprocess.load_recipes()
df = preprocess.preprocess_recipes(df)

if not os.path.exists("data/inverted_index.pkl") or \
    not os.path.exists("data/tfidf_vectorizer.pkl"):
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
        query = request.form["query"]
        results = search(query, df, inverted_index, vectorizer, tfidf_matrix, k=10)
    return render_template("index.html", results=results, query=query)

@app.route("/recipe/<int:recipe_id>")
def recipe_detail(recipe_id):
    recipe = df[df["RecipeId"] == recipe_id]

    if recipe.empty:
        return "Recipe not found", 404
    
    recipe = recipe.iloc[0]
    return render_template("recipe.html", recipe=recipe)
if __name__ == "__main__":
    app.run(debug=True)
    
