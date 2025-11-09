# ===============================================
# Cluster keywords using BERT embeddings and HDBSCAN
# ===============================================
"""
Usage:
python BERT_cluster.py \
  --input "여행객현지인의 추천 텍스트 데이터.xlsx" \
  --col KWRD_NM \
  --model jhgan/ko-sroberta-multitask \
  --min_cluster_size 5 \
  --min_samples 1 \
  --output kw_clusters.json
"""

import argparse, json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import hdbscan

def dedup_preserve_order(series):
    """Case-insensitive deduplication while preserving the first occurrence. Returns original text."""
    seen, out = set(), []
    for x in series.dropna():
        s = str(x).strip()
        k = s.lower()
        if k and k not in seen:
            seen.add(k)
            out.append(s)
    return out

def main():
    p = argparse.ArgumentParser(description="Keyword semantic clustering with HDBSCAN")
    p.add_argument("--input", required=True, help="Input Excel file (.xlsx)")
    p.add_argument("--sheet", default="0", help="Sheet name or index (default: 0)")
    p.add_argument("--col", default="KWRD_NM", help="Column name containing keywords (default: KWRD_NM)")
    p.add_argument("--model", default="jhgan/ko-sroberta-multitask", help="SBERT model name or path")
    p.add_argument("--min_cluster_size", type=int, default=5, help="Minimum cluster size for HDBSCAN")
    p.add_argument("--min_samples", type=int, default=None, help="Minimum samples parameter for HDBSCAN")
    p.add_argument("--device", default=None, help="Device: 'cuda' or 'cpu' (auto-detected if None)")
    p.add_argument("--output", default="clusters.json", help="Output path for JSON results (default: clusters.json)")
    args = p.parse_args()

    # 1) Load data and deduplicate
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df = pd.read_excel(args.input, sheet_name=sheet, engine="openpyxl")
    if args.col not in df.columns:
        raise SystemExit(f"Column '{args.col}' not found. Available columns: {list(df.columns)}")
    keywords = dedup_preserve_order(df[args.col])
    if not keywords:
        raise SystemExit("No valid keywords found.")
    print(f"[INFO] Unique keywords loaded: {len(keywords)}")

    # 2) Embedding
    model = SentenceTransformer(args.model, device=args.device)
    vecs = model.encode(
        keywords,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False
    )
    X = normalize(vecs)  # Cosine distance ≈ Euclidean distance after normalization

    # 3) Run HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_epsilon=0.0,
        prediction_data=False,
    ).fit(X)
    labels = clusterer.labels_

    # 4) Build JSON output — exclude noise (-1), order clusters by size (largest first)
    clusters = {}
    valid_idx = np.where(labels != -1)[0]
    if len(valid_idx) == 0:
        print("[WARN] No valid clusters found (all points classified as noise). Saving empty JSON.")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        return

    # Group by cluster label
    label2items = {}
    for i in valid_idx:
        lab = int(labels[i])
        label2items.setdefault(lab, []).append(keywords[i])

    # Sort clusters by size (descending) → cluster1, cluster2, ...
    ordered = sorted(label2items.items(), key=lambda kv: len(kv[1]), reverse=True)
    for idx, (_, items) in enumerate(ordered, start=1):
        clusters[f"cluster{idx}"] = items

    # 5) Save JSON (formatted manually to ensure compact per-line style)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("{\n")
        for i, (ckey, items) in enumerate(clusters.items()):
            line = f'  "{ckey}": ["' + '","'.join(items) + '"]'
            if i < len(clusters) - 1:
                line += ","
            f.write(line + "\n")
        f.write("}\n")
    print(f"[SAVE] Results written to {args.output}")

if __name__ == "__main__":
    main()

