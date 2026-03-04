# Assumes two CSVs with columns: test_index, y_true, y_pred
# Computes McNemar 2x2 table and exact two-sided p-value (binomial).

import math
import pandas as pd

# file paths - make more dynamic...
CSV_A = "preds_model_a.csv"
CSV_B = "preds_model_b.csv"

INDEX_COL = "test_index"
YTRUE_COL = "y_true"
YPRED_COL = "y_pred"

def mcnemar(n01: int, n10: int) -> float:
    """
    Exact McNemar test (two-sided) using binomial distribution.
    Under H0, discordant outcomes are symmetric: X ~ Bin(n01+n10, 0.5).
    p = 2 * min(P(X <= min(n01,n10)), P(X >= max(n01,n10))).
    """
    n = n01 + n10
    if n == 0:
        return 1.0
    
    k = min(n01, n10)

    # P(X <= k) where X ~ Bin(n, 0.5)
    # = sum_{i=0..k} C(n,i) * (0.5)^n
    cdf = 0.0
    for i in range(0, k + 1):
        cdf += math.comb(n, i)
    cdf *= 0.5 ** n

    return min(1.0, 2.0 * cdf)

def main():
    A = pd.read_csv(CSV_A)
    B = pd.read_csv(CSV_B)

    # Keep only needed columns + rename to avoid collisions
    A = A[[INDEX_COL, YTRUE_COL, YPRED_COL]].rename(
        columns={YTRUE_COL: "y_true_a", YPRED_COL: "y_pred_a"}
    )
    B = B[[INDEX_COL, YTRUE_COL, YPRED_COL]].rename(
        columns={YTRUE_COL: "y_true_b", YPRED_COL: "y_pred_b"}
    )

    M = A.merge(B, on=INDEX_COL, how="inner")
    if len(M) == 0:
        raise ValueError("No overlapping indices between the two CSVs.")
    
    # Sanity Checks
    if len(M) != len(A) or len(M) != len(B):
        print(f"Warning: merged rows={len(M)}; A rows={len(A)}; B rows={len(B)}."
              "Some indices are missing/extra in one file.")
        
    mismatch = (M["y_true_a"] != M["y_true_b"]).sum()
    if mismatch:
        # show a few examples to debug
        bad = M.loc[M["y_true_a"] != M["y_true_b"], [INDEX_COL, "y_true_a", "y_true_b"]].head(10)
        raise ValueError(f"y_true mismatch between CSVs for {mismatch} indices. Examples:\n{bad}")
    
    y_true = M["y_true_a"]
    a_correct = (M["y_pred_a"] == y_true)
    b_correct = (M["y_pred_b"] == y_true)

    n11 = int((a_correct & b_correct).sum())          # both correct
    n10 = int((a_correct & ~b_correct).sum())         # A correct, B wrong
    n01 = int((~a_correct & b_correct).sum())         # A wrong, B correct
    n00 = int((~a_correct & ~b_correct).sum())        # both wrong

    p = mcnemar(n01, n10)

    # Effect direction
    if n01 + n10 > 0:
        delta = (n10 - n01) / (n01 + n10)  # positive means A wins more often on discordant cases
    else:
        delta = 0.0

    print("McNemar 2x2 table (A rows vs B cols):")
    print(f"  n11 (A correct, B correct): {n11}")
    print(f"  n10 (A correct, B wrong)  : {n10}")
    print(f"  n01 (A wrong,   B correct): {n01}")
    print(f"  n00 (A wrong,   B wrong)  : {n00}")
    print()
    print(f"Discordant pairs (n01+n10): {n01 + n10}")
    print(f"Exact two-sided p-value   : {p:.6g}")
    print(f"Discordant direction delta: {delta:.4f}  (positive => A better, negative => B better)")


if __name__ == "__main__":
    main()