from flask import Flask, render_template, request, jsonify, session
from indexing import preprocess
from indexing import indexer
from indexing.recommender import recommend, get_similar_recipes, UserProfile
import os
import json
import pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "recipe-search-dev-key-change-in-prod")

df = preprocess.load_recipes()
df = preprocess.preprocess_recipes(df)

INDEX_DIR = "data"
os.makedirs(INDEX_DIR, exist_ok=True)

inv_path = f"{INDEX_DIR}/inverted_index.pkl"
tfidf_path = f"{INDEX_DIR}/tfidf_index.pkl"

if not os.path.exists(inv_path):
    print("Building inverted index...")
    inverted_index = indexer.build_inverted_index(df)
    indexer.save_index(inverted_index, inv_path)
else:
    inverted_index = indexer.load_index(inv_path)

if not os.path.exists(tfidf_path):
    print("Building TF-IDF index...")
    vectorizer, tfidf_matrix = indexer.build_tfidf_index(df)
    indexer.save_tfidf_index(vectorizer, tfidf_matrix, tfidf_path)
else:
    vectorizer, tfidf_matrix = indexer.load_tfidf_index(tfidf_path)

rid_to_row = {rid: i for i, rid in enumerate(df["RecipeId"].to_numpy())}
print(f"Ready! {len(df)} recipes indexed.")


def get_profile() -> UserProfile:
    raw = session.get("profile")
    if raw:
        return UserProfile.from_dict(raw)
    return UserProfile()

def save_profile(profile: UserProfile):
    session["profile"] = profile.to_dict()
    session.modified = True

def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

def safe_int(value, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default

def safe_str(value, default: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return default
        return str(value)
    except Exception:
        return default

def safe_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() == "nan":
            return []
        return [s]
    return []

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    ingredients = data.get("ingredients", [])
    k = int(data.get("k", 20))

    profile = get_profile()

    results = recommend(
        query=query,
        query_ingredients=ingredients,
        df=df,
        inverted_index=inverted_index,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        profile=profile,
        k=k,
    )

    return jsonify({"results": results})


@app.route("/api/recipe/<int:recipe_id>", methods=["GET"])
def api_recipe_detail(recipe_id: int):
    profile = get_profile()
    profile.record_view(recipe_id)
    save_profile(profile)

    if recipe_id not in rid_to_row:
        return jsonify({"error": "Recipe not found"}), 404

    row = df.iloc[rid_to_row[recipe_id]]

    def safe_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, float) and pd.isna(value):
            return []
        if isinstance(value, str):
            s = value.strip()
            if not s or s.lower() == "nan":
                return []
            if s.startswith("[") and s.endswith("]"):
                try:
                    import ast
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    pass
            return [x.strip() for x in s.split(",") if x.strip()]
        return []

    try:
        similar = get_similar_recipes(
            recipe_id,
            df,
            tfidf_matrix,
            rid_to_row,
            k=6,
        )
    except Exception as e:
        print(f"similar-recipes error for {recipe_id}: {e}")
        similar = []

    return jsonify({
        "RecipeId": safe_int(row.get("RecipeId", recipe_id), recipe_id),
        "Name": safe_str(row.get("Name", "")),
        "Description": safe_str(row.get("Description", "")),
        "RecipeCategory": safe_str(row.get("RecipeCategory", "")),
        "AggregatedRating": safe_float(row.get("AggregatedRating", 0)),
        "ReviewCount": safe_int(row.get("ReviewCount", 0)),
        "TotalTime": safe_str(row.get("TotalTime", "")),
        "Calories": safe_float(row.get("Calories", 0)),
        "RecipeServings": safe_str(row.get("RecipeServings", "")),
        "parsed_ingredients": safe_list(row.get("parsed_ingredients", [])),
        "parsed_instructions": safe_list(row.get("parsed_instructions", [])),
        "parsed_keywords": safe_list(row.get("parsed_keywords", [])),
        "is_favorited": recipe_id in profile.favorited,
        "is_made": recipe_id in profile.made,
        "similar": similar,
    })


@app.route("/api/profile/favorite/<int:recipe_id>", methods=["POST"])
def api_toggle_favorite(recipe_id: int):
    profile = get_profile()
    now_favorited = profile.toggle_favorite(recipe_id)
    save_profile(profile)
    return jsonify({"favorited": now_favorited, "recipe_id": recipe_id})


@app.route("/api/profile/made/<int:recipe_id>", methods=["POST"])
def api_toggle_made(recipe_id: int):
    profile = get_profile()
    now_made = profile.toggle_made(recipe_id)
    save_profile(profile)
    return jsonify({"made": now_made, "recipe_id": recipe_id})


@app.route("/api/profile", methods=["GET"])
def api_profile():
    profile = get_profile()
    return jsonify(profile.to_dict())


@app.route("/api/recommendations/for-you", methods=["POST"])
def api_for_you():
    """Purely personalized recommendations based on taste vector + optional ingredients."""
    data = request.get_json(force=True) or {}
    # Accept any ingredients the user currently has in their search box
    context_ingredients = data.get("ingredients", [])
 
    profile = get_profile()
 
    if not profile.get_positive_recipes():
        # Cold start: if they have ingredients, use those; otherwise show popular
        if context_ingredients:
            results = recommend(
                query="",
                query_ingredients=context_ingredients,
                df=df,
                inverted_index=inverted_index,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                profile=profile,
                k=20,
            )
            return jsonify({"results": results, "cold_start": True, "mode": "ingredients"})
 
        popular = df.nlargest(20, "AggregatedRating")
        results = []
        for _, row in popular.iterrows():
            results.append({
                "RecipeId": int(row["RecipeId"]),
                "Name": str(row["Name"]),
                "RecipeCategory": str(row.get("RecipeCategory", "")),
                "AggregatedRating": float(row.get("AggregatedRating", 0) or 0),
                "TotalTime": str(row.get("TotalTime", "")),
                "score": float(row.get("AggregatedRating", 0) or 0),
                "is_favorited": int(row["RecipeId"]) in profile.favorited,
                "is_made": int(row["RecipeId"]) in profile.made,
            })
        return jsonify({"results": results, "cold_start": True, "mode": "popular"})
 
    # Warm start: weight heavily toward taste profile, but blend in any current ingredients
    results = recommend(
        query="",
        query_ingredients=context_ingredients,
        df=df,
        inverted_index=inverted_index,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        profile=profile,
        k=20,
        alpha=0.10,   # small search/ingredient signal
        beta=0.85,    # heavy taste-profile signal
        gamma=0.05,   # small popularity nudge
    )
    return jsonify({"results": results, "cold_start": False, "mode": "personalized"})


if __name__ == "__main__":
    app.run(debug=True)
