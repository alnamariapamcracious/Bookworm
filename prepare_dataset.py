"""
prepare_dataset.py
------------------
Data Cleaning and Preparation Pipeline for:
"Book Taste & Recommendation Mining" (goodbooks-10k dataset)

This script processes raw Goodreads ratings and book metadata to create
curated transaction baskets for the Apriori association rule mining algorithm.

Key Pipeline Steps:
1. Validate and clean raw books and ratings data.
2. Filter for positive taste preferences (rating >= 4).
3. Select the Top 500 most frequently positively rated books.
4. Retain users with at least 2 positive ratings (meaningful baskets).
5. Output processed tables and a detailed preparation summary.

Safety:
- Does NOT alter or overwrite any files in data/raw/.
- All outputs are saved to data/processed/.
"""

import os
import sys
import pandas as pd


def main():
    # ---------------------------------------------------------
    # 0. Path Definitions and Setup
    # ---------------------------------------------------------
    project_root = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")

    raw_books_path = os.path.join(raw_dir, "books.csv")
    raw_ratings_path = os.path.join(raw_dir, "ratings.csv")

    cleaned_ratings_path = os.path.join(processed_dir, "cleaned_ratings.csv")
    selected_books_path = os.path.join(processed_dir, "selected_books.csv")
    transactions_path = os.path.join(processed_dir, "transactions.csv")
    summary_path = os.path.join(processed_dir, "preparation_summary.txt")

    # Verify input files exist
    if not os.path.exists(raw_books_path):
        raise FileNotFoundError(f"Raw books file not found at: {raw_books_path}")
    if not os.path.exists(raw_ratings_path):
        raise FileNotFoundError(f"Raw ratings file not found at: {raw_ratings_path}")

    os.makedirs(processed_dir, exist_ok=True)
    summary_lines = []

    def log_print(msg: str):
        print(msg)
        summary_lines.append(msg)

    log_print("=" * 70)
    log_print("BOOK TASTE & RECOMMENDATION MINING - DATA PREPARATION")
    log_print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load Raw Datasets
    # ---------------------------------------------------------
    log_print("\n[Step 1/6] Loading raw CSV files from data/raw/...")
    
    books_df = pd.read_csv(raw_books_path)
    ratings_df = pd.read_csv(raw_ratings_path)

    raw_books_count = len(books_df)
    raw_ratings_count = len(ratings_df)
    log_print(f"  - Loaded books.csv: {raw_books_count:,} rows")
    log_print(f"  - Loaded ratings.csv: {raw_ratings_count:,} rows")

    # ---------------------------------------------------------
    # 2. Validation & Cleaning
    # ---------------------------------------------------------
    log_print("\n[Step 2/6] Validating schema, data types, and integrity...")

    # Validate required columns
    required_book_cols = ["book_id", "title", "authors"]
    required_rating_cols = ["user_id", "book_id", "rating"]

    for col in required_book_cols:
        if col not in books_df.columns:
            raise ValueError(f"Missing required column in books.csv: '{col}'")
    for col in required_rating_cols:
        if col not in ratings_df.columns:
            raise ValueError(f"Missing required column in ratings.csv: '{col}'")

    # Enforce integer data types for IDs and ratings
    ratings_df["user_id"] = ratings_df["user_id"].astype("int64")
    ratings_df["book_id"] = ratings_df["book_id"].astype("int64")
    ratings_df["rating"] = ratings_df["rating"].astype("int64")
    books_df["book_id"] = books_df["book_id"].astype("int64")

    # Check for missing values in ratings
    missing_ratings = ratings_df[["user_id", "book_id", "rating"]].isnull().sum().sum()
    log_print(f"  - Missing values in core rating fields: {missing_ratings}")
    if missing_ratings > 0:
        ratings_df = ratings_df.dropna(subset=["user_id", "book_id", "rating"])

    # Validate rating score boundary (Goodreads scale is 1 to 5)
    invalid_ratings = ratings_df[~ratings_df["rating"].between(1, 5)]
    log_print(f"  - Invalid rating values (outside 1-5): {len(invalid_ratings)}")
    if len(invalid_ratings) > 0:
        ratings_df = ratings_df[ratings_df["rating"].between(1, 5)]

    # Check and eliminate duplicate (user_id, book_id) interactions
    duplicate_ratings = ratings_df.duplicated(subset=["user_id", "book_id"]).sum()
    log_print(f"  - Duplicate (user_id, book_id) pairs found: {duplicate_ratings}")
    if duplicate_ratings > 0:
        ratings_df = ratings_df.drop_duplicates(subset=["user_id", "book_id"], keep="last")

    # Foreign key check: ratings book_id must exist in books.csv
    valid_book_ids = set(books_df["book_id"].unique())
    orphan_ratings = (~ratings_df["book_id"].isin(valid_book_ids)).sum()
    log_print(f"  - Orphan ratings (book_id not in books.csv): {orphan_ratings}")
    if orphan_ratings > 0:
        ratings_df = ratings_df[ratings_df["book_id"].isin(valid_book_ids)]

    cleaned_ratings_count = len(ratings_df)
    log_print(f"  - Cleaned ratings count: {cleaned_ratings_count:,}")

    # Save cleaned ratings table
    log_print(f"  - Saving cleaned ratings to: {cleaned_ratings_path}")
    ratings_df.to_csv(cleaned_ratings_path, index=False)

    # ---------------------------------------------------------
    # 3. Filter for Positive Taste Preferences (Rating >= 4)
    # ---------------------------------------------------------
    # RATIONALE FOR RATING >= 4 THRESHOLD:
    # On Goodreads' 1-5 star scale:
    #   5 stars = "it was amazing"
    #   4 stars = "really liked it"
    #   3 stars = "liked it" (lukewarm/average)
    #   2 stars = "it was ok / did not like"
    #   1 star  = "did not like it"
    # In association rule mining (Market Basket Analysis), transaction items
    # represent products or books that a user actively endorses and recommends.
    # If mediocre (3) or disliked (1-2) books were included in baskets, the Apriori
    # algorithm would generate rules based on shared disinterest or indifference.
    # Filtering for rating >= 4 ensures that mined rules reflect true affinity and taste.
    log_print("\n[Step 3/6] Filtering for positive taste preferences (rating >= 4)...")
    positive_ratings_df = ratings_df[ratings_df["rating"] >= 4].copy()
    positive_ratings_count = len(positive_ratings_df)
    log_print(f"  - Positive ratings (>= 4): {positive_ratings_count:,} ({positive_ratings_count / cleaned_ratings_count:.2%})")

    # ---------------------------------------------------------
    # 4. Select Top 500 Most Frequently Positively Rated Books
    # ---------------------------------------------------------
    # RATIONALE FOR TOP 500 BOOKS:
    # Apriori frequent itemset generation has exponential complexity O(2^k).
    # Mining 10,000 sparse items across 53,000 users produces an enormous candidate
    # itemset lattice, leading to excessive memory usage and uninterpretable rules.
    # Restricting to the 500 most popular books with distinct positive endorsements
    # ensures dense co-occurrence patterns, fast execution, and practical recommendations.
    log_print("\n[Step 4/6] Identifying Top 500 most frequently positively rated books...")
    book_positive_counts = (
        positive_ratings_df.groupby("book_id")["user_id"]
        .nunique()
        .reset_index(name="positive_user_count")
    )
    top_500_books = (
        book_positive_counts.sort_values(by="positive_user_count", ascending=False)
        .head(500)
    )
    selected_book_ids = set(top_500_books["book_id"])
    log_print(f"  - Selected {len(selected_book_ids)} books with positive user count range: "
              f"[{top_500_books['positive_user_count'].min():,} to {top_500_books['positive_user_count'].max():,}]")

    # Filter positive ratings to the selected 500 books
    positive_top500_df = positive_ratings_df[positive_ratings_df["book_id"].isin(selected_book_ids)].copy()
    log_print(f"  - Ratings retained for Top 500 books: {len(positive_top500_df):,}")

    # Prepare and save selected_books.csv with metadata for UI and rule interpretation
    book_meta_cols = [
        "book_id", "goodreads_book_id", "authors", "original_publication_year",
        "title", "language_code", "average_rating", "ratings_count",
        "image_url", "small_image_url"
    ]
    # Filter columns that exist in the raw dataframe
    available_cols = [c for c in book_meta_cols if c in books_df.columns]
    selected_books_df = books_df[books_df["book_id"].isin(selected_book_ids)][available_cols].copy()
    selected_books_df = selected_books_df.merge(top_500_books, on="book_id", how="left")
    selected_books_df = selected_books_df.sort_values(by="positive_user_count", ascending=False)

    log_print(f"  - Saving selected books metadata to: {selected_books_path}")
    selected_books_df.to_csv(selected_books_path, index=False)

    # ---------------------------------------------------------
    # 5. Filter Users with Basket Size >= 2 (Transactions)
    # ---------------------------------------------------------
    # RATIONALE FOR BASKET SIZE >= 2:
    # Association rule mining requires at least 2 items per basket (antecedent -> consequent).
    # A transaction with a single book cannot produce any association rules.
    log_print("\n[Step 5/6] Filtering users with at least 2 positively rated books in Top 500...")
    user_basket_sizes = (
        positive_top500_df.groupby("user_id")["book_id"]
        .nunique()
        .reset_index(name="basket_size")
    )
    qualified_users = set(user_basket_sizes[user_basket_sizes["basket_size"] >= 2]["user_id"])
    log_print(f"  - Distinct users with >= 2 positive ratings in Top 500: {len(qualified_users):,}")

    transactions_df = positive_top500_df[positive_top500_df["user_id"].isin(qualified_users)][["user_id", "book_id"]].copy()
    transactions_df = transactions_df.sort_values(by=["user_id", "book_id"])

    log_print(f"  - Final transactions count (interactions): {len(transactions_df):,}")
    log_print(f"  - Saving transactions to: {transactions_path}")
    transactions_df.to_csv(transactions_path, index=False)

    # ---------------------------------------------------------
    # 6. Recompute Transaction Statistics & Generate Summary
    # ---------------------------------------------------------
    log_print("\n[Step 6/6] Computing final transaction statistics...")
    final_user_counts = transactions_df.groupby("user_id")["book_id"].count()
    num_users = len(final_user_counts)
    num_items = transactions_df["book_id"].nunique()
    total_interactions = len(transactions_df)

    mean_basket = final_user_counts.mean()
    median_basket = final_user_counts.median()
    min_basket = final_user_counts.min()
    max_basket = final_user_counts.max()
    density = (total_interactions / (num_users * num_items)) * 100 if num_users and num_items else 0

    log_print("\n" + "=" * 70)
    log_print("FINAL TRANSACTION DATASET SUMMARY STATISTICS")
    log_print("=" * 70)
    log_print(f"  - Retained Unique Users (Transactions): {num_users:,}")
    log_print(f"  - Retained Unique Books:                {num_items:,} (out of 500 target)")
    log_print(f"  - Total User-Book Interactions:         {total_interactions:,}")
    log_print(f"  - Average Basket Size:                  {mean_basket:.2f} books")
    log_print(f"  - Median Basket Size:                   {median_basket:.0f} books")
    log_print(f"  - Min Basket Size:                      {min_basket} books")
    log_print(f"  - Max Basket Size:                      {max_basket} books")
    log_print(f"  - Interaction Matrix Density:           {density:.2f}% (Sparsity: {100 - density:.2f}%)")
    log_print("=" * 70)

    # Write summary report to preparation_summary.txt
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    log_print(f"\nPreparation report successfully written to: {summary_path}")


if __name__ == "__main__":
    main()
