"""
mine_apriori.py
---------------
Vectorized Apriori Association Rule Mining Pipeline for:
"Book Taste & Recommendation Mining"

Implemented entirely using NumPy and pandas without external ML libraries.

Core Apriori Workflow:
1. Load transactions.csv and selected_books.csv from data/processed/.
2. Construct a memory-efficient Boolean user-book interaction matrix.
3. Level 1: Discover frequent 1-itemsets (L1) with support >= MIN_SUPPORT.
4. Level 2: Vectorized pairwise co-occurrence calculation using int64 BLAS matrix
   multiplication to discover frequent 2-itemsets (L2).
5. Level 3: Candidate 3-itemset generation (L2 join L2) with strict Apriori
   anti-monotonicity pruning (all 2-item subsets must be frequent) to discover (L3).
6. Rule Generation: Formulate all valid single-book consequent rules (X -> Y),
   evaluating Support, Confidence, and Lift. Retain rules satisfying thresholds.
7. Enrich with book IDs, titles, and authors, sort by Lift and Confidence,
   and persist model artifacts and summary.

Configurable Hyperparameters:
- MIN_SUPPORT:    0.01 (1% minimum transaction support)
- MIN_CONFIDENCE: 0.30 (30% minimum rule confidence)
- MAX_LEN:        3    (1-item, 2-item, 3-item combinations)
- MIN_LIFT:       1.0  (strictly greater than 1.0 for positive correlation)
"""

import os
import sys
import time
from collections import defaultdict
import numpy as np
import pandas as pd


# -------------------------------------------------------------
# Configurable Mining Settings
# -------------------------------------------------------------
MIN_SUPPORT = 0.01      # Minimum support threshold (1%)
MIN_CONFIDENCE = 0.30   # Minimum confidence threshold (30%)
MAX_LEN = 3             # Maximum itemset size (1, 2, or 3 books)
MIN_LIFT = 1.0          # Minimum lift threshold (> 1.0 indicates positive association)


def load_data(project_root: str):
    """Loads transactions and book metadata from data/processed/."""
    processed_dir = os.path.join(project_root, "data", "processed")
    transactions_path = os.path.join(processed_dir, "transactions.csv")
    books_path = os.path.join(processed_dir, "selected_books.csv")

    if not os.path.exists(transactions_path):
        raise FileNotFoundError(
            f"Transactions file not found at: {transactions_path}. "
            "Please run 'prepare_dataset.py' first."
        )
    if not os.path.exists(books_path):
        raise FileNotFoundError(
            f"Selected books metadata not found at: {books_path}. "
            "Please run 'prepare_dataset.py' first."
        )

    print(f"Loading transactions from: {transactions_path}")
    transactions_df = pd.read_csv(transactions_path, dtype={"user_id": "int64", "book_id": "int64"})

    print(f"Loading book metadata from: {books_path}")
    books_df = pd.read_csv(books_path)

    return transactions_df, books_df


def build_basket_matrix(transactions_df: pd.DataFrame):
    """
    Constructs a contiguous 2D Boolean NumPy matrix where rows = users,
    columns = books, and matrix[u, b] indicates positive preference.
    """
    print("\nConstructing Boolean basket matrix from transactions...")
    unique_users = sorted(transactions_df["user_id"].unique())
    unique_books = sorted(transactions_df["book_id"].unique())

    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    book_to_idx = {b: i for i, b in enumerate(unique_books)}
    idx_to_book = {i: b for i, b in enumerate(unique_books)}

    num_users = len(unique_users)
    num_books = len(unique_books)

    # Use contiguous bool array (1 byte per cell, ~26.5 MB RAM)
    basket_matrix = np.zeros((num_users, num_books), dtype=bool)

    user_indices = transactions_df["user_id"].map(user_to_idx).values
    book_indices = transactions_df["book_id"].map(book_to_idx).values

    basket_matrix[user_indices, book_indices] = True

    print(f"  - Basket matrix created: {num_users:,} users x {num_books:,} books")
    print(f"  - Memory footprint: {basket_matrix.nbytes / (1024 * 1024):.2f} MB")

    return basket_matrix, unique_users, unique_books, book_to_idx, idx_to_book


def mine_apriori(basket_matrix: np.ndarray, unique_books: list, book_to_idx: dict, idx_to_book: dict, num_users: int):
    """
    Executes Apriori frequent itemset mining for k = 1, 2, 3 using vectorized NumPy operations
    with downward closure (anti-monotonicity) pruning.
    """
    frequent_itemsets = {}  # frozenset(book_ids): support
    min_count = int(np.ceil(MIN_SUPPORT * num_users))
    print(f"\nApriori Support Threshold: {MIN_SUPPORT:.2%} (requires at least {min_count:,} users)")

    # -------------------------------------------------------------
    # Level 1: Frequent 1-itemsets (L1)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    print("[Apriori Level 1] Evaluating individual book frequencies...")
    
    # Use int64 accumulation to prevent any integer overflow
    item_counts = basket_matrix.sum(axis=0, dtype=np.int64)
    item_supports = item_counts / num_users

    l1_indices = np.where(item_supports >= MIN_SUPPORT)[0]
    for idx in l1_indices:
        b_id = idx_to_book[idx]
        frequent_itemsets[frozenset([b_id])] = float(item_supports[idx])

    print(f"  - L1 Frequent 1-itemsets found: {len(l1_indices):,} (out of {len(unique_books)} candidates) in {time.perf_counter() - t0:.3f}s")

    if len(l1_indices) < 2 or MAX_LEN < 2:
        return frequent_itemsets, {}

    # -------------------------------------------------------------
    # Level 2: Frequent 2-itemsets (L2) via Vectorized BLAS
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    print("[Apriori Level 2] Generating and evaluating candidate 2-itemsets...")

    # Filter columns to only frequent 1-items (Apriori principle)
    b_l1 = basket_matrix[:, l1_indices].astype(np.int64)

    # Co-occurrence matrix: C[i, j] = number of users who liked both book i and book j
    # Executed via optimized BLAS integer dot product
    co_matrix = b_l1.T @ b_l1

    # Extract upper triangle indices (i < j) where count >= min_count
    triu_i, triu_j = np.triu_indices(len(l1_indices), k=1)
    pair_counts = co_matrix[triu_i, triu_j]
    valid_pair_mask = pair_counts >= min_count

    l2_pairs_set = set()  # set of tuple(sorted([id1, id2]))
    for idx_a, idx_b, count in zip(triu_i[valid_pair_mask], triu_j[valid_pair_mask], pair_counts[valid_pair_mask]):
        book_a = idx_to_book[l1_indices[idx_a]]
        book_b = idx_to_book[l1_indices[idx_b]]
        pair = tuple(sorted([book_a, book_b]))
        l2_pairs_set.add(pair)
        frequent_itemsets[frozenset(pair)] = float(count / num_users)

    print(f"  - L2 Frequent 2-itemsets found: {len(l2_pairs_set):,} in {time.perf_counter() - t0:.3f}s")

    if len(l2_pairs_set) == 0 or MAX_LEN < 3:
        return frequent_itemsets, l2_pairs_set

    # -------------------------------------------------------------
    # Level 3: Frequent 3-itemsets (L3) with Apriori Pruning
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    print("[Apriori Level 3] Generating candidate 3-itemsets with anti-monotonicity pruning...")

    # Candidate Generation (L2 join L2):
    # Two pairs (a, b) and (a, c) sharing the first item can form candidate (a, b, c) where a < b < c
    pair_dict = defaultdict(list)
    for a, b in l2_pairs_set:
        pair_dict[a].append(b)

    candidates_c3 = []
    for a, b_list in pair_dict.items():
        sorted_b = sorted(b_list)
        for i in range(len(sorted_b)):
            b = sorted_b[i]
            for j in range(i + 1, len(sorted_b)):
                c = sorted_b[j]
                # Apriori Pruning Check: All 2-item subsets must be in L2!
                # (a, b) and (a, c) are guaranteed in L2; we must verify (b, c) in L2.
                if (b, c) in l2_pairs_set:
                    candidates_c3.append((a, b, c))

    print(f"  - Candidate 3-itemsets generated after pruning: {len(candidates_c3):,}")

    # Evaluate support for surviving 3-item candidates
    l3_count = 0
    for a, b, c in candidates_c3:
        idx_a = book_to_idx[a]
        idx_b = book_to_idx[b]
        idx_c = book_to_idx[c]

        # Fast bitwise conjunction across users
        triplet_count = (basket_matrix[:, idx_a] & basket_matrix[:, idx_b] & basket_matrix[:, idx_c]).sum(dtype=np.int64)
        if triplet_count >= min_count:
            triplet_support = float(triplet_count / num_users)
            frequent_itemsets[frozenset([a, b, c])] = triplet_support
            l3_count += 1

    print(f"  - L3 Frequent 3-itemsets found: {l3_count:,} in {time.perf_counter() - t0:.3f}s")

    return frequent_itemsets, l2_pairs_set


def generate_association_rules(frequent_itemsets: dict, books_df: pd.DataFrame):
    """
    Generates association rules with single-book consequents satisfying
    MIN_CONFIDENCE and MIN_LIFT.
    """
    print("\nGenerating association rules with single-book consequents...")
    t0 = time.perf_counter()

    id_to_title = dict(zip(books_df["book_id"], books_df["title"]))
    id_to_author = dict(zip(books_df["book_id"], books_df["authors"]))

    rules_records = []
    total_evaluated_rules = 0

    # Iterate over all frequent itemsets of length >= 2
    for itemset, itemset_support in frequent_itemsets.items():
        k = len(itemset)
        if k < 2:
            continue

        itemset_list = list(itemset)

        # Generate all rules with exactly 1 consequent: (itemset - {consequent}) -> {consequent}
        for consequent_item in itemset_list:
            total_evaluated_rules += 1
            consequent = frozenset([consequent_item])
            antecedent = itemset - consequent

            antecedent_support = frequent_itemsets.get(antecedent)
            consequent_support = frequent_itemsets.get(consequent)

            if not antecedent_support or not consequent_support:
                continue

            confidence = itemset_support / antecedent_support
            lift = confidence / consequent_support

            # Retain only rules meeting thresholds
            if confidence >= MIN_CONFIDENCE and lift > MIN_LIFT:
                ant_sorted = sorted(list(antecedent))
                ant_ids_str = ", ".join(str(x) for x in ant_sorted)
                ant_titles_str = " | ".join(id_to_title.get(x, f"Book #{x}") for x in ant_sorted)

                cons_id = consequent_item
                cons_title = id_to_title.get(cons_id, f"Book #{cons_id}")
                cons_author = id_to_author.get(cons_id, "Unknown")

                # Optional metrics: leverage and conviction
                leverage = itemset_support - (antecedent_support * consequent_support)
                conviction = (
                    (1.0 - consequent_support) / (1.0 - confidence)
                    if confidence < 1.0
                    else float("inf")
                )

                rules_records.append({
                    "antecedent_ids": ant_ids_str,
                    "antecedent_titles": ant_titles_str,
                    "antecedent_len": len(ant_sorted),
                    "consequent_id": cons_id,
                    "consequent_title": cons_title,
                    "consequent_author": cons_author,
                    "support": round(itemset_support, 6),
                    "confidence": round(confidence, 6),
                    "lift": round(lift, 4),
                    "leverage": round(leverage, 6),
                    "conviction": round(conviction, 4) if conviction != float("inf") else 999.0
                })

    rules_df = pd.DataFrame(rules_records)

    if not rules_df.empty:
        # Sort by lift descending, then confidence descending
        rules_df = rules_df.sort_values(by=["lift", "confidence"], ascending=[False, False]).reset_index(drop=True)

    print(f"  - Evaluated {total_evaluated_rules:,} candidate single-book rules.")
    print(f"  - Retained {len(rules_df):,} rules meeting thresholds (confidence >= {MIN_CONFIDENCE:.2%}, lift > {MIN_LIFT}) in {time.perf_counter() - t0:.3f}s")

    return rules_df, total_evaluated_rules


def main():
    start_total_time = time.perf_counter()

    project_root = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(project_root, "data", "processed")

    itemsets_output_path = os.path.join(processed_dir, "frequent_itemsets.csv")
    rules_output_path = os.path.join(processed_dir, "association_rules.csv")
    summary_output_path = os.path.join(processed_dir, "model_summary.txt")

    summary_lines = []

    def log(msg: str):
        print(msg)
        summary_lines.append(msg)

    log("=" * 75)
    log("BOOK TASTE & RECOMMENDATION MINING - APRIORI MODEL (NUMPY/PANDAS)")
    log("=" * 75)
    log(f"Configuration Settings:")
    log(f"  - Minimum Support:            {MIN_SUPPORT:.2%} ({MIN_SUPPORT})")
    log(f"  - Minimum Confidence:         {MIN_CONFIDENCE:.2%} ({MIN_CONFIDENCE})")
    log(f"  - Maximum Itemset Length:     {MAX_LEN}")
    log(f"  - Minimum Recommendation Lift:{MIN_LIFT}")
    log("=" * 75)

    # 1. Load Data
    transactions_df, books_df = load_data(project_root)
    id_to_title = dict(zip(books_df["book_id"], books_df["title"]))

    # 2. Build Basket Matrix
    basket_matrix, unique_users, unique_books, book_to_idx, idx_to_book = build_basket_matrix(transactions_df)
    num_users = len(unique_users)
    num_books = len(unique_books)

    log(f"\nDataset Dimensions:")
    log(f"  - Total Transactions (Users): {num_users:,}")
    log(f"  - Target Book Catalog:        {num_books:,} books")
    log(f"  - Total Interaction Records:  {len(transactions_df):,}")

    # 3. Mine Frequent Itemsets
    frequent_itemsets, _ = mine_apriori(basket_matrix, unique_books, book_to_idx, idx_to_book, num_users)

    # Format frequent itemsets for persistence
    itemset_rows = []
    itemset_counts_by_len = defaultdict(int)

    for itemset, supp in frequent_itemsets.items():
        k = len(itemset)
        itemset_counts_by_len[k] += 1
        sorted_ids = sorted(list(itemset))
        ids_str = ", ".join(str(x) for x in sorted_ids)
        titles_str = " | ".join(id_to_title.get(x, f"Book #{x}") for x in sorted_ids)
        itemset_rows.append({
            "itemset_ids": ids_str,
            "itemset_titles": titles_str,
            "length": k,
            "support": round(supp, 6)
        })

    itemsets_df = pd.DataFrame(itemset_rows)
    if not itemsets_df.empty:
        itemsets_df = itemsets_df.sort_values(by=["support", "length"], ascending=[False, True]).reset_index(drop=True)

    # Deliberately save frequent itemsets
    itemsets_df.to_csv(itemsets_output_path, index=False)
    log(f"\nFrequent Itemsets Discovered: {len(itemsets_df):,}")
    for length_val in range(1, MAX_LEN + 1):
        log(f"  - {length_val}-item combinations: {itemset_counts_by_len[length_val]:,}")

    # 4. Generate Association Rules
    rules_df, total_evaluated_rules = generate_association_rules(frequent_itemsets, books_df)

    # Deliberately save association rules
    rules_df.to_csv(rules_output_path, index=False)
    log(f"\nAssociation Rules Generated:")
    log(f"  - Candidate Single-Book Rules Evaluated: {total_evaluated_rules:,}")
    log(f"  - Retained Usable Recommendation Rules:   {len(rules_df):,}")

    # Preview Top Rules in Summary
    if not rules_df.empty:
        log("\n" + "=" * 75)
        log("TOP 5 HIGHEST LIFT RECOMMENDATION RULES")
        log("=" * 75)
        top_lift = rules_df.head(5)
        for idx, row in top_lift.iterrows():
            log(f"Rule #{idx+1}:")
            log(f"  IF user likes:  {row['antecedent_titles']}")
            log(f"  RECOMMEND:      {row['consequent_title']} (by {row['consequent_author']})")
            log(f"  Metrics:        Lift={row['lift']:.2f} | Confidence={row['confidence']:.2%} | Support={row['support']:.4f}")

        log("\n" + "=" * 75)
        log("TOP 5 HIGHEST CONFIDENCE RECOMMENDATION RULES")
        log("=" * 75)
        top_conf = rules_df.sort_values(by=["confidence", "lift"], ascending=[False, False]).head(5)
        for idx, row in top_conf.iterrows():
            log(f"Rule #{idx+1}:")
            log(f"  IF user likes:  {row['antecedent_titles']}")
            log(f"  RECOMMEND:      {row['consequent_title']} (by {row['consequent_author']})")
            log(f"  Metrics:        Confidence={row['confidence']:.2%} | Lift={row['lift']:.2f} | Support={row['support']:.4f}")
    else:
        log("\n[NOTE] No association rules met the specified confidence and lift thresholds.")

    total_elapsed = time.perf_counter() - start_total_time

    log("\n" + "=" * 75)
    log("FINAL MODEL EXECUTION SUMMARY")
    log("=" * 75)
    log(f"  - Transactions (Users) Analyzed:        {num_users:,}")
    log(f"  - Books in Itemset Catalog:             {num_books:,}")
    log(f"  - Total Frequent Itemsets:              {len(itemsets_df):,}")
    log(f"  - Usable Single-Book Rules:             {len(rules_df):,}")
    log(f"  - Total Execution Runtime:              {total_elapsed:.2f} seconds")
    log("=" * 75)

    # Save summary report
    with open(summary_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    log(f"\nModel summary successfully written to: {summary_output_path}")


if __name__ == "__main__":
    main()
