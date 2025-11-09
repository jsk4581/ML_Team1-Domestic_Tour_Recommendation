# ===============================================
# User-based Collaborative Filtering with Tourist Persona 
# ===============================================
"""
user_cf_persona.py  — ABU Recall@N + (Regression error on hit items) + Recommend by user
- Persona-based cosine kNN (TRAVEL_STYL_* + GENDER/AGE_GRP)
- Recommend TRAVEL_STATUS_DESTINATION
- Evaluation: ABU (All-But-User) recall averaged per user
              For hit items: compute DGSTFN regression error (MAE_hit, RMSE_hit), and accuracy within ±0.5 (Acc@0.5_hit)
Usage:
  # Evaluation mode
  python user_cf_persona.py --csv travel_train_scaled_standard.csv --k 2 --topN 5 --max_users 5000

  # Recommendation mode (specific user)
  python user_cf_persona.py --csv travel_train_scaled_standard.csv --k 3 --topN 3 --user_id e007797
"""
import argparse, pandas as pd, numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def load_data(path):
    """Load dataset and filter to training split if applicable."""
    df = pd.read_csv(path)
    return df[df.get("split", "Training").astype(str).str.lower().str.contains("train")].copy()

def build_ui(df, user_col, dest_col, rating_col):
    """Construct user-item (destination) interaction matrix."""
    return (df[[user_col, dest_col, rating_col]]
            .groupby([user_col, dest_col], as_index=False)
            .agg({rating_col: "mean"}))

def build_persona(df, user_col, persona_cols):
    """Build per-user persona vectors (mean of feature columns)."""
    return (df[[user_col] + persona_cols]
            .groupby(user_col, as_index=False).mean()
            .set_index(user_col).sort_index())

def main(args):
    USER_COL, DEST_COL, RATING_COL = "TRAVELER_ID", "TRAVEL_STATUS_DESTINATION", "DGSTFN"
    df = load_data(args.csv)

    # Define persona feature columns
    persona_cols = [c for c in df.columns if c.startswith("TRAVEL_STYL_")]
    for bonus in ["GENDER", "AGE_GRP"]:
        if bonus in df.columns:
            persona_cols.append(bonus)

    # Build user-item matrix and persona embeddings
    ui = build_ui(df, USER_COL, DEST_COL, RATING_COL)
    user_persona = build_persona(df, USER_COL, persona_cols)

    user2idx = {u: i for i, u in enumerate(user_persona.index)}
    idx2user = {i: u for u, i in user2idx.items()}

    sim = cosine_similarity(user_persona.values)
    np.fill_diagonal(sim, 0.0)

    # User → (destination, rating)
    user_items = defaultdict(dict)
    for _, row in ui.iterrows():
        user_items[row[USER_COL]][row[DEST_COL]] = float(row[RATING_COL])

    all_users = [u for u in user_persona.index if u in user_items]
    all_dests = sorted(ui[DEST_COL].unique())

    # Deterministic subset of users for evaluation
    eval_users = list(all_users)
    if args.max_users is not None and len(eval_users) > args.max_users:
        eval_users = eval_users[:args.max_users]

    # Precompute neighbors for all users
    def precompute_neighbors(K):
        neigh = {}
        for i, u in enumerate(user_persona.index):
            sims = sim[i]
            k = min(K, len(sims))
            if k <= 0:
                neigh[u] = []
                continue
            idx = np.argpartition(-sims, range(k))[:k]
            idx = idx[np.argsort(-sims[idx])]  # sort by descending similarity
            pairs = [(idx2user[j], float(sims[j])) for j in idx if sims[j] > 0]
            neigh[u] = pairs[:K]
        return neigh

    neighbors = precompute_neighbors(args.k)

    def predict_score(u, d):
        """Predict DGSTFN score using weighted average from neighbors."""
        neighs = neighbors.get(u, [])[:args.k]
        num = den = 0.0
        for v, w in neighs:
            r = user_items.get(v, {}).get(d)
            if r is not None:
                num += w * r
                den += w
        if den > 0:
            return num / den
        # Fallback: user's mean or global mean
        if user_items.get(u):
            return float(np.mean(list(user_items[u].values())))
        return float(ui[RATING_COL].mean())

    def topn_preds(u, include_items=None):
        """Return sorted (destination, predicted score) list for user u.
        include_items: optionally include visited items in candidate set.
        """
        include_items = set(include_items or [])
        visited = set(user_items.get(u, {}).keys())
        cand = [d for d in all_dests if (d not in visited) or (d in include_items)]
        preds = [(d, predict_score(u, d)) for d in cand]
        preds.sort(key=lambda x: -x[1])
        return preds[:args.topN]

    # -------- Mode 1: Evaluation (ABU recall + hit regression error + Acc@0.5_hit) --------
    if not args.user_id:
        per_user_recalls = []
        n_used = 0
        # Accumulators for regression error on hits
        hit_abs_sum = 0.0
        hit_sq_sum = 0.0
        n_hits = 0
        # Accuracy within ±0.5 on hit items
        within05_hits = 0

        for u in eval_users:
            rel = set(user_items[u].keys())
            if not rel:
                continue
            preds = topn_preds(u, include_items=rel)
            topn_items = {d for d, _ in preds}

            # Per-user recall
            hit_cnt = len(topn_items & rel)
            rec_u = hit_cnt / len(rel)
            per_user_recalls.append(rec_u)
            n_used += 1

            # Compute DGSTFN regression errors for hit items
            for d, pred_score in preds:
                if d in rel and d in topn_items:
                    r_true = user_items[u][d]
                    err = float(r_true - pred_score)
                    hit_abs_sum += abs(err)
                    hit_sq_sum += err * err
                    n_hits += 1
                    # Count hits within ±0.5
                    if abs(err) <= 0.5:
                        within05_hits += 1

        if not per_user_recalls:
            print(f"Recall@{args.topN}: NaN (no evaluable users)")
        else:
            recall = float(np.mean(per_user_recalls))
            if n_hits > 0:
                mae_hit = hit_abs_sum / n_hits
                rmse_hit = (hit_sq_sum / n_hits) ** 0.5
                acc05_hit = within05_hits / n_hits
                print(f"Recall@{args.topN}: {recall:.6f} \nAcc@0.5_hit={acc05_hit:.6f} | hits={n_hits} "
                      f"\nMAE_hit={mae_hit:.6f} | RMSE_hit={rmse_hit:.6f}")
            else:
                print(f"Recall@{args.topN}: {recall:.6f} | hits=0 | Acc@0.5_hit=NaN | MAE_hit=NaN | RMSE_hit=NaN")
        return

    # -------- Mode 2: Inference (recommendations for a specific user) --------
    uid = args.user_id
    if uid not in user_persona.index:
        print(f"[ERROR] TRAVELER_ID '{uid}' not found in persona matrix.")
        return
    if uid not in user_items:
        print(f"[WARN] TRAVELER_ID '{uid}' has no interactions; recommendations will rely solely on neighbors.")

    # Generate recommendations excluding already visited destinations
    visited = set(user_items.get(uid, {}).keys())
    cand = [d for d in all_dests if d not in visited]
    preds = [(d, predict_score(uid, d)) for d in cand]
    preds.sort(key=lambda x: -x[1])
    topn = preds[:args.topN]

    print(f"[Recommend] TRAVELER_ID={uid} | k={args.k} | topN={args.topN}")
    for rank, (d, s) in enumerate(topn, start=1):
        print(f"{rank:2d}. {d}\tpred_dgstfn={s:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--topN", type=int, default=5)
    ap.add_argument("--max_users", type=int, default=5000)
    ap.add_argument("--user_id", type=str, default="")  # If provided → recommendation mode, otherwise evaluation mode
    args = ap.parse_args()
    main(args)
