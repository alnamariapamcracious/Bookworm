"""
app.py
------
Flask Web Application & Recommendation Server for:
"Book Taste & Recommendation Mining"

Serves the web user interface and REST API endpoints:
- GET  /              : Web Interface
- GET  /api/books     : Searchable 500-Book Catalog
- POST /api/recommend : Mined Apriori Association Rule Recommendations
- GET  /api/stats     : Data Mining Model Metrics & Parameters
"""

import os
import math
import itertools
from collections import defaultdict
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Path definitions
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BOOKS_PATH = os.path.join(PROCESSED_DIR, "selected_books.csv")
RULES_PATH = os.path.join(PROCESSED_DIR, "association_rules.csv")
SUMMARY_PATH = os.path.join(PROCESSED_DIR, "model_summary.txt")

# In-memory indexes
CATALOG = {}                  # book_id (int) -> book dict
BOOKS_LIST = []               # list of book dicts for catalog search
RULES_BY_ANTECEDENT = defaultdict(list) # frozenset(book_ids) -> list of rules
MODEL_STATS = {}              # model metadata and hyperparameter stats


def clean_val(val, default=""):
    """Helper to clean NaN or float values from pandas into JSON-safe types."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val


def load_knowledge_base():
    """Loads selected books and precomputed association rules into fast in-memory indexes."""
    global CATALOG, BOOKS_LIST, RULES_BY_ANTECEDENT, MODEL_STATS

    print("\n" + "=" * 60)
    print("INITIALIZING APRIORI RECOMMENDATION ENGINE...")
    print("=" * 60)

    # 1. Load Catalog
    if not os.path.exists(BOOKS_PATH):
        raise FileNotFoundError(f"Missing catalog file: {BOOKS_PATH}. Run 'prepare_dataset.py' first.")
    
    books_df = pd.read_csv(BOOKS_PATH)
    CATALOG.clear()
    BOOKS_LIST.clear()

    for _, row in books_df.iterrows():
        b_id = int(row["book_id"])
        year_raw = row.get("original_publication_year")
        year = int(year_raw) if (pd.notnull(year_raw) and not math.isnan(float(year_raw))) else None

        book_info = {
            "book_id": b_id,
            "goodreads_book_id": int(row.get("goodreads_book_id", 0)),
            "title": str(row.get("title", f"Book #{b_id}")),
            "authors": str(row.get("authors", "Unknown Author")),
            "original_publication_year": year,
            "average_rating": round(float(row.get("average_rating", 0.0)), 2),
            "ratings_count": int(row.get("ratings_count", 0)),
            "positive_user_count": int(row.get("positive_user_count", 0)),
            "image_url": clean_val(row.get("image_url"), "/static/img/book_placeholder.svg"),
            "small_image_url": clean_val(row.get("small_image_url"), "/static/img/book_placeholder.svg"),
        }
        CATALOG[b_id] = book_info
        BOOKS_LIST.append(book_info)

    # Sort books by positive user count descending
    BOOKS_LIST.sort(key=lambda x: x["positive_user_count"], reverse=True)
    print(f"  - Loaded {len(CATALOG):,} books into catalog lookup.")

    # 2. Load Association Rules Index
    if not os.path.exists(RULES_PATH):
        raise FileNotFoundError(f"Missing rules file: {RULES_PATH}. Run 'mine_apriori.py' first.")

    rules_df = pd.read_csv(RULES_PATH)
    RULES_BY_ANTECEDENT.clear()
    total_rules_loaded = 0

    for _, row in rules_df.iterrows():
        ant_ids_str = str(row["antecedent_ids"])
        ant_ids_list = [int(x.strip()) for x in ant_ids_str.split(",") if x.strip()]
        ant_key = frozenset(ant_ids_list)

        rule_entry = {
            "antecedent_ids": ant_ids_list,
            "antecedent_titles": str(row["antecedent_titles"]),
            "antecedent_len": int(row["antecedent_len"]),
            "consequent_id": int(row["consequent_id"]),
            "consequent_title": str(row["consequent_title"]),
            "consequent_author": str(row.get("consequent_author", "")),
            "support": float(row["support"]),
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
            "leverage": float(row.get("leverage", 0.0)),
            "conviction": float(row.get("conviction", 0.0))
        }
        RULES_BY_ANTECEDENT[ant_key].append(rule_entry)
        total_rules_loaded += 1

    print(f"  - Indexed {total_rules_loaded:,} association rules across {len(RULES_BY_ANTECEDENT):,} unique antecedents.")

    # 3. Model Stats
    MODEL_STATS = {
        "catalog_books": len(CATALOG),
        "total_rules": total_rules_loaded,
        "unique_antecedents": len(RULES_BY_ANTECEDENT),
        "min_support": 0.01,
        "min_confidence": 0.30,
        "min_lift": 1.0,
        "max_itemset_len": 3,
        "transactions_analyzed": 52959
    }
    print("=" * 60)
    print("RECOMMENDATION ENGINE READY.\n")


# Load catalog and rules on startup
load_knowledge_base()


# -------------------------------------------------------------
# HTTP Routes & API Endpoints
# -------------------------------------------------------------
@app.route("/")
def index():
    """Serves the primary web interface."""
    return render_template("index.html")


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Returns dataset overview and Apriori configuration parameters."""
    return jsonify({
        "success": True,
        "stats": MODEL_STATS
    })


@app.route("/api/books", methods=["GET"])
def api_books():
    """
    Searchable catalog endpoint.
    Query parameters:
    - q: search term matching title or author (case-insensitive)
    - limit: maximum books to return (default 60, max 500)
    """
    query = request.args.get("q", "").strip().lower()
    try:
        limit = min(int(request.args.get("limit", 60)), 500)
    except ValueError:
        limit = 60

    if not query:
        results = BOOKS_LIST[:limit]
    else:
        results = [
            b for b in BOOKS_LIST
            if query in b["title"].lower() or query in b["authors"].lower()
        ][:limit]

    return jsonify({
        "success": True,
        "total_matched": len(results),
        "books": results
    })


@app.route("/api/recommend", methods=["GET", "POST"])
def api_recommend():
    """
    Generates book recommendations using precomputed Apriori association rules.
    Accepts:
    - POST JSON: {"book_ids": [1, 2, ...]}
    - GET Query: /api/recommend?ids=1,2,...
    """
    selected_raw = []

    if request.method == "POST":
        if request.is_json:
            data = request.get_json(silent=True) or {}
            selected_raw = data.get("book_ids", [])
        else:
            selected_raw = request.form.getlist("book_ids")
    else:
        ids_param = request.args.get("ids", "").strip()
        if ids_param:
            selected_raw = ids_param.split(",")

    # Validate book IDs
    valid_ids = []
    for item in selected_raw:
        try:
            b_id = int(str(item).strip())
            if b_id in CATALOG and b_id not in valid_ids:
                valid_ids.append(b_id)
        except ValueError:
            continue

    # Return empty response if no valid books provided
    if not valid_ids:
        return jsonify({
            "success": True,
            "selected_count": 0,
            "selected_books": [],
            "recommendation_count": 0,
            "recommendations": [],
            "message": "Please select at least one book from the catalog to receive personalized recommendations."
        })

    selected_set = set(valid_ids)
    selected_books_metadata = [CATALOG[b_id] for b_id in valid_ids]

    # Generate antecedent candidate subsets of lengths 1 and 2
    subsets_to_check = []
    
    # 1-book antecedents (singletons)
    for b_id in valid_ids:
        subsets_to_check.append(frozenset([b_id]))

    # 2-book antecedents (pairs)
    if len(valid_ids) >= 2:
        for pair in itertools.combinations(valid_ids, 2):
            subsets_to_check.append(frozenset(pair))

    # Match rules from knowledge base
    matched_rules_by_consequent = defaultdict(list)

    for subset in subsets_to_check:
        matching_rules = RULES_BY_ANTECEDENT.get(subset, [])
        for rule in matching_rules:
            cons_id = rule["consequent_id"]
            # STRICT REQUIREMENT: Never recommend a book already selected by the user!
            if cons_id in selected_set:
                continue
            matched_rules_by_consequent[cons_id].append(rule)

    # If no rules match, return clear transparent feedback (NO synthetic recommendations)
    if not matched_rules_by_consequent:
        return jsonify({
            "success": True,
            "selected_count": len(valid_ids),
            "selected_books": selected_books_metadata,
            "recommendation_count": 0,
            "recommendations": [],
            "message": (
                "No association rules met the 1.0% minimum support and 30% confidence thresholds "
                "for your selected book combination. In data mining, this indicates your reading taste "
                "is uniquely diverse! Try selecting another popular book to discover matched patterns."
            )
        })

    # Deduplicate and rank recommendations
    ranked_recommendations = []

    for cons_id, rules_list in matched_rules_by_consequent.items():
        if cons_id not in CATALOG:
            continue

        # Sort rules for this consequent to find the single strongest rule
        # Rank by: 1. Lift (descending), 2. Confidence (descending), 3. Antecedent length (descending)
        sorted_rules = sorted(
            rules_list,
            key=lambda r: (r["lift"], r["confidence"], r["antecedent_len"]),
            reverse=True
        )

        primary_rule = sorted_rules[0]
        supporting_rules = sorted_rules[1:4]  # Up to 3 additional rules for rich explanation

        book_meta = CATALOG[cons_id]

        # Natural language explanation
        ant_names = primary_rule["antecedent_titles"].replace(" | ", " AND ")
        explanation = (
            f"Readers who loved {ant_names} are {primary_rule['lift']:.1f}x more likely "
            f"to also enjoy this book (Rule Confidence: {primary_rule['confidence']:.1%})."
        )

        recommendation_card = {
            "book_id": cons_id,
            "title": book_meta["title"],
            "authors": book_meta["authors"],
            "original_publication_year": book_meta["original_publication_year"],
            "average_rating": book_meta["average_rating"],
            "ratings_count": book_meta["ratings_count"],
            "image_url": book_meta["image_url"],
            "small_image_url": book_meta["small_image_url"],
            # Primary rule metrics
            "lift": primary_rule["lift"],
            "confidence": primary_rule["confidence"],
            "support": primary_rule["support"],
            "antecedent_len": primary_rule["antecedent_len"],
            "matched_antecedent_ids": primary_rule["antecedent_ids"],
            "matched_antecedent_titles": primary_rule["antecedent_titles"],
            "explanation": explanation,
            "total_supporting_rules": len(sorted_rules),
            "supporting_rules": [
                {
                    "antecedent_titles": r["antecedent_titles"],
                    "confidence": r["confidence"],
                    "lift": r["lift"]
                }
                for r in supporting_rules
            ]
        }
        ranked_recommendations.append(recommendation_card)

    # Rank all recommended books:
    # 1. Lift (strength of association above random baseline)
    # 2. Confidence (conditional probability)
    # 3. Antecedent Length (two-book rules reflect more specific taste)
    ranked_recommendations.sort(
        key=lambda x: (x["lift"], x["confidence"], x["antecedent_len"]),
        reverse=True
    )

    # Limit to top 24 recommendations
    final_recs = ranked_recommendations[:24]

    return jsonify({
        "success": True,
        "selected_count": len(valid_ids),
        "selected_books": selected_books_metadata,
        "recommendation_count": len(final_recs),
        "total_matching_books": len(ranked_recommendations),
        "recommendations": final_recs,
        "message": f"Successfully mined {len(final_recs)} personalized recommendations based on your book taste!"
    })


@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Resource not found"}), 404


@app.errorhandler(500)
def handle_500(e):
    return jsonify({"error": "An internal server error occurred"}), 500


if __name__ == "__main__":
    # Run locally on port 5000
    app.run(host="127.0.0.1", port=5000, debug=False)
