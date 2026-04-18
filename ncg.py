# =========================
# evaluate_ndcg.py
# Evaluate Recall / NDCG / MRR
# =========================

import pandas as pd
import numpy as np

# ---------- Config ----------
RESULT_PATH = "new-senior/bert4rec-main/scripts/all_user_recommendations-3_part1.csv"
K_LIST = [1, 5, 10]


# ---------- Metric utils ----------
def recall_at_k(gt, recs, k):
    return int(gt in recs[:k])


def dcg_at_k(gt, recs, k):
    if gt in recs[:k]:
        rank = recs.index(gt) + 1
        return 1.0 / np.log2(rank + 1)
    return 0.0


def ndcg_at_k(gt, recs, k):
    dcg = dcg_at_k(gt, recs, k)
    idcg = 1.0  # ideal DCG when GT is at rank 1
    return dcg / idcg


def mrr_at_k(gt, recs, k):
    if gt in recs[:k]:
        rank = recs.index(gt) + 1
        return 1.0 / rank
    return 0.0


# ---------- Main ----------
def main():
    df = pd.read_csv(RESULT_PATH)

    # Top10_Rec เป็น string → list[int]
    df["Top10_Rec"] = df["Top10_Rec"].apply(eval)

    metrics = {}

    for k in K_LIST:
        df[f"Recall@{k}"] = df.apply(
            lambda r: recall_at_k(r["GT_next_item"], r["Top10_Rec"], k), axis=1
        )
        df[f"NDCG@{k}"] = df.apply(
            lambda r: ndcg_at_k(r["GT_next_item"], r["Top10_Rec"], k), axis=1
        )
        df[f"MRR@{k}"] = df.apply(
            lambda r: mrr_at_k(r["GT_next_item"], r["Top10_Rec"], k), axis=1
        )

        metrics[f"Recall@{k}"] = df[f"Recall@{k}"].mean()
        metrics[f"NDCG@{k}"] = df[f"NDCG@{k}"].mean()
        metrics[f"MRR@{k}"] = df[f"MRR@{k}"].mean()

    # ---------- Print ----------
    print("\n===== Evaluation Result =====")
    for k in K_LIST:
        print(
            f"K={k:2d} | "
            f"Recall@{k}: {metrics[f'Recall@{k}']:.4f} | "
            f"NDCG@{k}: {metrics[f'NDCG@{k}']:.4f} | "
            f"MRR@{k}: {metrics[f'MRR@{k}']:.4f}"
        )

    # ---------- Save ----------
    out_path = RESULT_PATH.replace(".csv", "_with_metrics.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved detailed metrics to: {out_path}")


if __name__ == "__main__":
    main()
