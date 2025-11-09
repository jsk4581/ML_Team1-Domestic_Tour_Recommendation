
## Quickstart
```bash
git clone https://github.com/jsk4581/ML_Team1-Domestic_Tour_Recommendation.git
pip install -r requirements.txt

# DomeTour - Tourism Recommendation 

This repository contains a small but complete toolkit for **tourist behavior analysis** and **recommendation** based on:

* User-based collaborative filtering
* K-Means clustering of traveler personas
* Text (keyword) clustering with SBERT + HDBSCAN
* Content-based filtering using keyword attributes

All components are implemented as simple, composable Python scripts.

* * *

## Repository Structure

```text
.
├─ 1-2_cluster_recommend.py   # K-Means clustering + hybrid recommendation table
├─ 1-2_cluster_eval.py        # Offline evaluation: Precision @ K
├─ 1_user_cf_persona.py       # User-based CF with persona (ABU Recall@N + regression errors)
├─ 2_BERT_cluster.py          # Keyword clustering using SBERT + HDBSCAN
└─ 2_cbf_kw.py                # Content-based region similarity (keyword matrix)
```

* * *

## High-Level Architecture

1. **User-based Collaborative Filtering (Persona kNN)**
    
    * `1_user_cf_persona.py`
        
        * Builds user–destination interactions and persona vectors.
            
        * Uses cosine kNN over personas (TRAVEL_STYL + GENDER/AGE_GRP).
            
        * Supports:
            
            * **ABU evaluation**: Recall@N + DGSTFN regression errors on hit items (MAE, RMSE, Acc@0.5).
                
            * **Online-style recommendation** for a given user.


2. **Persona-based K-Means Clustering + Hybrid Recommendation**
    
    * `1-2_cluster_recommend.py`
        
        * Clusters travelers using persona features (TRAVEL_STYL, GENDER, AGE_GRP).
            
        * Computes cluster profiles and **cluster × destination** average satisfaction.
            
        * Builds a **hybrid recommendation table** that combines absolute satisfaction and uplift (personalization).

            
    **Clustering-based Recommendation Evaluation**
    
    * `1-2_cluster_eval.py`
        
        * Loads the trained K-Means model + hybrid recommendation table.
            
        * Evaluates **Precision@K** on a validation set, using actual visit + satisfaction ≥ threshold as “good hit”.

                
3. **Keyword Clustering with SBERT + HDBSCAN**
    
    * `2_BERT_cluster.py`
        
        * Reads an Excel column of textual keywords (KWRD_NM).
            
        * Embeds with `SentenceTransformer` (e.g., `jhgan/ko-sroberta-multitask`).
            
        * Clusters with HDBSCAN and outputs semantic keyword clusters as JSON.

            
4. **Content-based Region Similarity**
    
    * `2_cbf_kw.py`
        
        * Reads an AREA_NM × attribute matrix from Excel.
            
        * Computes cosine similarity over attribute vectors.
            
        * Prints Top-K regions most similar to a target region (e.g., “강원도”).
            

These components can be used independently, but conceptually form a **tourism analysis and recommendation stack**:

* text clustering → defines/organizes attributes and keywords
    
* content-based and CF models → recommend regions and destinations
    
* clustering + hybrid scoring → segment travelers and rank destinations
    
* evaluation scripts → quantitatively assess recommendation quality.
    

* * *

## Module-by-Module Architecture

### `1_user_cf_persona.py` – User-based CF with Persona kNN

#### Purpose

Implements a **user-based collaborative filtering** model over:

* Persona features (`TRAVEL_STYL_*`, `GENDER`, `AGE_GRP`).
    
* Satisfaction ratings (`DGSTFN`) to destinations.
    

Provides:

* **Evaluation mode**
    
    * ABU (All-But-User) Recall@N
        
    * Regression error on “hit” items: MAE_hit, RMSE_hit, Acc@0.5_hit.
        
* **Recommendation mode**
    
    * Top-N destination recommendations for a specific user.
        

#### External Libraries

* `argparse` – CLI options.
    
* `pandas`, `numpy` – data handling, numerics.
    
* `collections.defaultdict` – nested dict for user→items.
    
* `sklearn.metrics.pairwise.cosine_similarity` – persona similarity.
    

#### Functions

1. `load_data(path)`
    
    * Reads a CSV.
        
    * Filters rows where `split` column contains “train” (case-insensitive).
        
    * Returns training subset as DataFrame.
        
2. `build_ui(df, user_col, dest_col, rating_col)`
    
    * Builds **user–item interaction matrix**:
        
        ```python
        df[[user_col, dest_col, rating_col]]
          .groupby([user_col, dest_col], as_index=False)
          .agg({rating_col: "mean"})
        ```
        
    * Aggregates multiple interactions per user–destination via mean.
        
3. `build_persona(df, user_col, persona_cols)`
    
    * Groups by user and averages persona columns → per-user persona vector.
        
4. `main(args)`
    
    High-level steps:
    
    * Define column names:
        
        * `USER_COL = "TRAVELER_ID"`
            
        * `DEST_COL = "TRAVEL_STATUS_DESTINATION"`
            
        * `RATING_COL = "DGSTFN"`
            
    * Load and split data (`load_data`).
        
    * Define persona columns:
        
        * `TRAVEL_STYL_*` plus optional `GENDER`, `AGE_GRP`.
            
    * Build:
        
        * `ui` – user–destination–rating table.
            
        * `user_persona` – user → persona vector.
            
    * Create index mappings: `user2idx`, `idx2user`.
        
    * Compute user–user similarity via `cosine_similarity(user_persona.values)` and zero out diagonals.
        
    
    **Intermediate structures:**
    
    * `user_items: dict[USER] → dict[DEST] → rating`
        
    * `all_users` – users with at least one interaction.
        
    * `all_dests` – all unique destinations.
        
5. Neighbor Precomputation: `precompute_neighbors(K)`
    
    * For each user index:
        
        * Get similarity vector `sims`.
            
        * Take top-K similar users via `np.argpartition`.
            
        * Sort by similarity descending.
            
        * Filter to similarity > 0.
            
    * Returns mapping: `user → list[(neighbor_user, sim_weight)]`.
        
6. Rating Prediction: `predict_score(u, d)`
    
    * Weighted average of neighbors’ ratings for destination `d`.
        
    * If no neighbor has rating:
        
        * Fallback to user’s own mean rating if available.
            
        * Otherwise global mean rating.
            
7. Top-N Prediction: `topn_preds(u, include_items=None)`
    
    * Candidate destinations:
        
        * All destinations not visited by `u`, **plus** any explicitly included items.
            
    * For each candidate, call `predict_score(u, d)`.
        
    * Sort by score descending; return top-N.
        

#### Modes

1. **Evaluation Mode** (no `--user_id`):
    
    * Evaluate using All-But-User style:
        
        * For each evaluable user:
            
            * `rel` = set of destinations user has interacted with.
                
            * `preds = topn_preds(u, include_items=rel)`
                
            * `topn_items` = destinations in `preds`.
                
            * **Recall@N**:
                
                * `hit_cnt = len(topn_items & rel)`
                    
                * `rec_u = hit_cnt / len(rel)`
                    
            * **Hit regression metrics**:
                
                * For each `(d, pred_score)` in `preds`:
                    
                    * If `d ∈ rel ∩ topn_items`:
                        
                        * `r_true = user_items[u][d]`
                            
                        * `err = r_true - pred_score`
                            
                        * accumulate:
                            
                            * `hit_abs_sum += |err|`
                                
                            * `hit_sq_sum += err^2`
                                
                            * increment `n_hits`
                                
                            * if `|err| ≤ 0.5`, increment `within05_hits`.
                                
    * Finally:
        
        * `Recall = mean(per_user_recalls)`
            
        * `MAE_hit = hit_abs_sum / n_hits`
            
        * `RMSE_hit = sqrt(hit_sq_sum / n_hits)`
            
        * `Acc@0.5_hit = within05_hits / n_hits`
            
2. **Recommendation Mode** (`--user_id` specified):
    
    * Validate that user exists in persona and has/has not interactions.
        
    * Compute candidate destinations (excluding visited ones).
        
    * `preds = [(d, predict_score(uid, d)) ...]`
        
    * Sort and show top-N with predicted DGSTFN.
        
* * * 

### `1-2_cluster_recommend.py` – K-Means Clustering & Hybrid Recommendation

1-2_cluster_recommend

#### Purpose

* Cluster travelers into persona groups using K-Means.
    
* Profile each cluster.
    
* Build a **cluster-specific recommendation table** with:
    
    * Mean satisfaction by (cluster, destination).
        
    * A 50:50 hybrid score of absolute satisfaction and uplift.
        

#### Top-Level Flow

1. **Imports**
    
    * `pandas` – CSV I/O and grouping/aggregation.
        
    * `matplotlib.pyplot`, `seaborn` – (optional) plotting for silhouette analysis.
        
    * `sklearn.cluster.KMeans` – K-Means clustering.
        
    * `sklearn.metrics.silhouette_score` – silhouette score for K selection.
        
    * `joblib` – model serialization (save KMeans model).
        
    * `warnings`, `os` – suppress warnings, check file existence.
        
    * Second part:
        
        * `sklearn.preprocessing.MinMaxScaler` – feature scaling for hybrid score.
            
2. **Data Loading**
    
    ```python
    data_df = pd.read_csv(data_file)
    ```
    
    * Loads pre-scaled training data (`travel_train_scaled_standard.csv`).
        
    * Critical columns:
        
        * `GENDER`, `AGE_GRP`, `TRAVEL_STYL_1..8` – persona features.
            
        * `TRAVEL_STATUS_DESTINATION`, `DGSTFN` – destination & satisfaction.
            
3. **Feature Definition**
    
    ```python
    features_for_clustering = ['GENDER', 'AGE_GRP'] + [f'TRAVEL_STYL_{i}' for i in range(1, 9)]
    X_cluster = data_df[features_for_clustering]
    ```
    
    * Defines the feature subset used for K-Means.
        
4. **Silhouette-based K Selection**
    
    ```python
    for k in range(2, 11):
        kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_model.fit(X_cluster)
        score = silhouette_score(X_cluster, labels, random_state=42)
    ```
    
    * Computes silhouette scores for different `k`.
        
    * Uses the **full dataset** (no sampling).
        
5. **Final K-Means Clustering**
    
    ```python
    OPTIMAL_K = 8
    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    data_df['CLUSTER_ID'] = kmeans.labels_
    ```
    
    * Uses pre-decided `K=8`.
        
    * Assigns cluster labels to each traveler.
        
6. **Cluster Profiling**
    
    ```python
    profile_cols = ['AGE_GRP', 'GENDER'] + [f'TRAVEL_STYL_{i}' for i in range(1, 9)]
    cluster_profile = data_df.groupby('CLUSTER_ID')[profile_cols].mean()
    cluster_profile.to_csv("cluster_profiles_scaled.csv", ...)
    ```
    
    * Produces per-cluster mean persona features.
        
7. **Recommendation Table (Cluster × Destination)**
    
    ```python
    recommendation_table = (
        data_df.groupby(['CLUSTER_ID', 'TRAVEL_STATUS_DESTINATION'])['DGSTFN']
        .mean()
        .reset_index()
    )
    recommendation_table.to_csv("recommendation_table_specific.csv", ...)
    ```
    
    * For each cluster and destination, computes **average DGSTFN**.
        
    * This forms the base popularity table.
        
8. **Model Persistence**
    
    ```python
    joblib.dump(kmeans, "kmeans_user_cluster_model.joblib")
    ```
    

#### Hybrid Recommendation Sub-module

1. **Load Recommendation Table**
    
    ```python
    reco_df = pd.read_csv("recommendation_table_specific.csv")
    ```
    
2. **Compute Overall Average Satisfaction per Destination**
    
    ```python
    overall_avg_satisfaction = (
        reco_df.groupby('TRAVEL_STATUS_DESTINATION')['DGSTFN']
        .mean()
        .reset_index()
    )
    merged_df = pd.merge(reco_df, overall_avg_satisfaction, on='TRAVEL_STATUS_DESTINATION')
    merged_df['Uplift_Score'] = merged_df['DGSTFN'] - merged_df['Overall_Avg_DGSTFN']
    ```
    
    * `DGSTFN` – cluster-specific mean satisfaction.
        
    * `Overall_Avg_DGSTFN` – global mean satisfaction per destination.
        
    * `Uplift_Score` – how much this cluster likes a destination above/below global average.
        
3. **Scale and Combine into Hybrid Score**
    
    ```python
    scaler = MinMaxScaler()
    merged_df['DGSTFN_scaled'] = scaler.fit_transform(merged_df[['DGSTFN']])
    merged_df['Uplift_scaled'] = scaler.fit_transform(merged_df[['Uplift_Score']])
    merged_df['Hybrid_Score_5050'] = (
        merged_df['DGSTFN_scaled'] * 0.5 + merged_df['Uplift_scaled'] * 0.5
    )
    ```
    
4. **Save Hybrid Table & Print Top-3 per Cluster**
    
    * Writes to `recommendation_table_hybrid_5050.csv`.
        
    * For each `CLUSTER_ID`, prints Top-3 destinations by `Hybrid_Score_5050`.
        

* * *

### `1-2_cluster_eval.py` – Clustering Evaluation (Precision@K)

#### Purpose

Evaluate how **useful** cluster-based recommendations are by checking:

> When the model recommends Top-K destinations for a user’s cluster,  
> how often does the user’s actual visited destination appear in the list,  
> and how often is that visit “satisfactory” (DGSTFN ≥ threshold)?

#### Top-Level Flow

1. **Imports**
    
    * `pandas` – load validation data and recommendation table.
        
    * `joblib` – load trained KMeans model.
        
    * `warnings`, `os` – environment handling.
        
2. **Config**
    
    * `MODEL_FILE` – KMeans model (`kmeans_user_cluster_model.joblib`).
        
    * `TEST_DATA_FILE` – validation set (scaled, contains actual visits).
        
    * `RECO_TABLE_FILE` – hybrid recommendation table by cluster.
        
    * `K_FOR_HITRATE` – K in Precision@K.
        
    * `SATISFACTION_THRESHOLD` – e.g. DGSTFN ≥ 4.0.
        
3. **Load Artifacts**
    
    ```python
    kmeans_model = joblib.load(MODEL_FILE)
    reco_table = pd.read_csv(RECO_TABLE_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)
    ```
    
4. **Evaluation Loop**
    
    * Features for cluster assignment:
        
        ```python
        features_for_clustering = ['GENDER', 'AGE_GRP'] + [f'TRAVEL_STYL_{i}' for i in range(1, 9)]
        ```
        
    * For each user row:
        
        1. Extract actual destination & satisfaction.
            
        2. Build a single-row persona DataFrame.
            
        3. Predict cluster with KMeans.
            
        4. Filter `reco_table` to that `CLUSTER_ID`.
            
        5. Take Top-K destinations.
            
        6. If actual destination ∈ Top-K → overlap.
            
        7. If overlapped and DGSTFN ≥ threshold → satisfied hit.
            
    * Counters:
        
        * `total_overlap_count` – denominator (all overlaps).
            
        * `satisfied_hit_count` – numerator (overlaps with high satisfaction).
            
5. **Metric**
    
    ```python
    precision = (satisfied_hit_count / total_overlap_count) * 100
    ```
    
    * Interpreted as:
        
        > “When cluster-based recommendation and actual visit overlap,  
        > how often is the visit satisfactory?”
        

* * *

### `2_BERT_cluster.py` – Keyword Clustering with SBERT + HDBSCAN

#### Purpose

Cluster **keyword strings** (e.g., tourism-related KWRD_NM) into **semantic groups** using:

* SBERT (Korean multi-task RoBERTa) embeddings.
    
* HDBSCAN density-based clustering.
    

#### External Libraries

* `argparse`, `json` – CLI parsing, JSON output.
    
* `pandas`, `numpy` – data and numerics.
    
* `sentence_transformers.SentenceTransformer` – text embeddings.
    
* `sklearn.preprocessing.normalize` – L2 normalization.
    
* `hdbscan` – clustering.
    

#### Functions

1. `dedup_preserve_order(series)`
    
    * Case-insensitive deduplication:
        
        * Uses `.lower()` keys to detect duplicates.
            
        * Keeps the **first** occurrence’s original string.
            
    * Returns list of unique keyword strings.
        
2. `main()`
    
    * CLI arguments:
        
        * `--input` – input Excel file.
            
        * `--sheet` – sheet name or index.
            
        * `--col` – keyword column (default `KWRD_NM`).
            
        * `--model` – SBERT model name.
            
        * `--min_cluster_size`, `--min_samples` – HDBSCAN parameters.
            
        * `--device` – `cuda` or `cpu`.
            
        * `--output` – JSON output file.
            
    
    Steps:
    
    1. **Load Excel & extract keywords** → `keywords` (deduplicated).
        
    2. **Embed** via `SentenceTransformer.encode`:
        
        * `batch_size=64`, `convert_to_numpy=True`, no internal normalization.
            
    3. **Normalize embeddings**:
        
        * `X = normalize(vecs)` for cosine ≈ Euclidean.
            
    4. **Run HDBSCAN**:
        
        * `metric="euclidean"`.
            
        * `labels = clusterer.labels_`.
            
    5. **Filter noise** (`label = -1`).
        
    6. **Aggregate by label** into `label2items`.
        
    7. **Sort clusters by size (descending)** and rename to `cluster1`, `cluster2`, ...
        
    8. **Write JSON** manually as:
        
        ```json
        {
          "cluster1": ["kw1","kw2",...],
          "cluster2": ["kw3","kw4",...]
        }
        ```
        

* * *

### `2_cbf_kw.py` – Content-Based Filtering for Regions

#### Purpose

Simple **content-based filtering** example for **tourist regions**, based on:

* Precomputed AREA_NM × attribute matrix (`지역별_관광속성_행렬.xlsx`).
    
* Cosine similarity of attribute vectors.
    

#### Flow

1. **Imports**
    
    * `pandas` – Excel I/O.
        
    * `sklearn.metrics.pairwise.cosine_similarity` – similarity matrix.
        
2. **Data Loading**
    
    ```python
    df = pd.read_excel("지역별_관광속성_행렬.xlsx")
    features = df.drop(columns=["AREA_NM"]).values
    names = df["AREA_NM"].values
    ```
    
3. **Similarity Matrix**
    
    ```python
    similarity_matrix = cosine_similarity(features)
    ```
    
4. **Target Region & Ranking**
    
    ```python
    target = "강원도"
    idx = list(names).index(target)
    similarities = list(enumerate(similarity_matrix[idx]))
    sorted_sim = sorted(similarities, key=lambda x: x[1], reverse=True)
    ```
    
    * Prints Top-5 most similar regions (excluding self).
        

* * *

## External Libraries and Their Roles

* **pandas**
    
    * Reading/writing CSV/Excel.
        
    * Groupby aggregations for cluster profiles and recommendation tables.
        
* **numpy**
    
    * Array operations, numerical computation, means, index/argpartition, etc.
        
* **scikit-learn**
    
    * `KMeans` – clustering travelers into persona-based groups.
        
    * `silhouette_score` – model selection for K.
        
    * `MinMaxScaler` – scaling satisfaction/uplift to [0,1] before combining.
        
    * `cosine_similarity` – similarity measures for personas and region content.
        
* **sentence-transformers**
    
    * `SentenceTransformer` – multilingual SBERT encoder used for keyword embeddings.
        
* **hdbscan**
    
    * Density-based clustering for text embeddings, automatically discovering cluster count and marking noise.
        
* **joblib**
    
    * Model persistence: saving and loading KMeans models.
        
* **argparse**
    
    * Command-line interfaces for flexible configuration of input paths, parameters, etc.
        
* **warnings, os, json, collections.defaultdict**
    
    * Environment safety (suppress warnings), file path checks, JSON output, and convenient nested dictionaries.
        

* * *

## How Components Fit Together

* **Clustering pipeline**
    
    1. Run `1-2_cluster_recommend.py` to:
        
        * Train KMeans on persona features.
            
        * Generate `kmeans_user_cluster_model.joblib`.
            
        * Generate `recommendation_table_specific.csv`.
            
        * Generate `recommendation_table_hybrid_5050.csv`.
            
            1-2_cluster_recommend
            
    2. Run `1-2_cluster_eval.py` to:
        
        * Evaluate **Precision@K** of cluster-based recommendations on validation data.
            
            1-2_cluster_eval
            
* **User-based CF pipeline**
    
    * Run `1_user_cf_persona.py` in evaluation mode for ABU metrics, or in recommendation mode to get personalized Top-N for a specific user.
        
        1_user_cf_persona
        
* **Keyword & content pipelines**
    
    * Run `2_BERT_cluster.py` to build semantic keyword clusters for analysis or feature engineering.
        
        2_BERT_cluster
        
    * Run `2_cbf_kw.py` to explore region similarity based on attribute matrix.
        
        2_cbf_kw
        

All together, this forms a **research-grade toolkit** for experimenting with:

* Persona clustering,
    
* Cluster & CF-based recommendation,
    
* Text/keyword clustering,
    
* Content-based region similarity,
    

on tourism datasets.
