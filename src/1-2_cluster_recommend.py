# ===============================================
# K-Means Clustering and Recommendation Table Generator - Tourists data
# ===============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  # import silhouette score
import joblib
import warnings
import os

# Suppress unnecessary warning messages
warnings.filterwarnings('ignore')

# --- 1. [Required] Load preprocessed data ---
print("1. 전처리된 데이터를 로드합니다...")
data_file = r"C:\Users\신승민\OneDrive\바탕 화면\scaled_results(3)\travel_train_scaled_standard.csv"
if not os.path.exists(data_file):
    print(f"오류: '{data_file}' 경로를 찾을 수 없습니다!")
    print(f"경로 확인: {data_file}")
    exit()
data_df = pd.read_csv(data_file)
print(f"-> 데이터 로드 완료: {data_df.shape}")

# --- 2. Define features for clustering ---
features_for_clustering = ['GENDER', 'AGE_GRP'] + [f'TRAVEL_STYL_{i}' for i in range(1, 9)]
X_cluster = data_df[features_for_clustering]
print(f"-> 클러스터링에 {len(features_for_clustering)}개의 특성을 사용합니다.")

# --- 3. Find optimal K using the Silhouette Method ---
print("\n2. 최적의 K 값을 찾기 위해 실루엣 계수를 계산합니다... (전체 데이터 사용. 매우 오래 걸릴 수 있음)")

silhouette_scores = []  # list to store silhouette scores
K_range = range(2, 11)  # recommended test range (2–10)

for k in K_range:
    print(f"  K={k} 계산 중...")
    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_model.fit(X_cluster)
    labels = kmeans_model.labels_
    
    # Removed sample_size parameter to use the full dataset
    score = silhouette_score(X_cluster, labels, random_state=42)
    silhouette_scores.append(score)
    print(f"  K={k}일 때 실루엣 점수: {score:.4f}")

# (Optional) Plotting code for silhouette scores can be added below

# --- 4. Perform final K-Means clustering ---
# Based on prior analysis, K=8 is selected
OPTIMAL_K = 8  

print(f"\n3. K={OPTIMAL_K}로 최종 클러스터링을 실행합니다...")
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
kmeans.fit(X_cluster)
print("-> 클러스터링 완료.")

# --- 5. Analyze results and create the recommendation table ---
print("4. 클러스터링 결과를 분석하고 '추천 엔진 테이블'을 생성합니다...")
data_df['CLUSTER_ID'] = kmeans.labels_

# 5-1. Cluster profiling
profile_cols = ['AGE_GRP', 'GENDER'] + [f'TRAVEL_STYL_{i}' for i in range(1, 9)]
cluster_profile = data_df.groupby('CLUSTER_ID')[profile_cols].mean()
print("\n[각 클러스터별 프로필 (스케일링된 평균값)]")
print(cluster_profile)
cluster_profile.to_csv("cluster_profiles_scaled.csv", encoding='utf-8-sig')
print("-> 'cluster_profiles_scaled.csv'에 프로필 저장 완료.")

# --- 5-2. Group by destination instead of region (bug fix) ---
print("\n[그룹-지역별 평균 만족도 (추천 엔진 테이블)]")
# Replace region column with TRAVEL_STATUS_DESTINATION
recommendation_table = data_df.groupby(['CLUSTER_ID', 'TRAVEL_STATUS_DESTINATION'])['DGSTFN'].mean().reset_index()
recommendation_table = recommendation_table.sort_values(by=['CLUSTER_ID', 'DGSTFN'], ascending=[True, False])

print(recommendation_table)
# Save with a new filename
recommendation_table.to_csv("recommendation_table_specific.csv", index=False, encoding='utf-8-sig')
print("-> 'recommendation_table_specific.csv'에 수정된 추천 테이블 저장 완료.")

# --- 6. Save trained model ---
model_filename = "kmeans_user_cluster_model.joblib"
joblib.dump(kmeans, model_filename)
print(f"\n[작업 완료] 학습된 K-Means 모델이 '{model_filename}'으로 저장되었습니다.")

# ===============================================
# Hybrid Recommendation Table Generation (50:50)
# ===============================================

import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # import for scaling
import warnings
import os

# Suppress unnecessary warning messages
warnings.filterwarnings('ignore')

# --- 1. [Required] Load input file ---
print("1. 'recommendation_table_specific.csv' 파일을 로드합니다...")

# File path
reco_file = r"C:\Users\신승민\.vscode\recommendation_table_specific.csv"

if not os.path.exists(reco_file):
    print(f"오류: '{reco_file}' 경로를 찾을 수 없습니다!")
    print(f"경로 확인: {reco_file}")
    exit()

reco_df = pd.read_csv(reco_file)
print(f"-> 로드 완료: {reco_df.shape}")

# --- 2. Compute 'Uplift Score' (personalization score) ---
print("2. 'Uplift 점수' (개인화 점수)를 계산합니다...")

overall_avg_satisfaction = reco_df.groupby('TRAVEL_STATUS_DESTINATION')['DGSTFN'].mean().reset_index()
overall_avg_satisfaction.rename(columns={'DGSTFN': 'Overall_Avg_DGSTFN'}, inplace=True)
merged_df = pd.merge(reco_df, overall_avg_satisfaction, on='TRAVEL_STATUS_DESTINATION')
merged_df['Uplift_Score'] = merged_df['DGSTFN'] - merged_df['Overall_Avg_DGSTFN']
print("-> Uplift 점수 계산 완료.")

# --- 3. Compute hybrid score (50:50 ratio) ---
print("3. '절대 만족도'와 'Uplift 점수'를 스케일링하여 하이브리드 점수를 만듭니다...")

scaler = MinMaxScaler()
merged_df['DGSTFN_scaled'] = scaler.fit_transform(merged_df[['DGSTFN']])
merged_df['Uplift_scaled'] = scaler.fit_transform(merged_df[['Uplift_Score']])

# Set weights to 50:50
w_satisfaction = 0.5  # absolute satisfaction (popularity) weight: 50%
w_uplift = 0.5        # uplift score (personalization) weight: 50%

# 3-3. Compute final hybrid score
merged_df['Hybrid_Score_5050'] = (merged_df['DGSTFN_scaled'] * w_satisfaction) + \
                                 (merged_df['Uplift_scaled'] * w_uplift)

print(f"-> 하이브리드 점수 계산 완료 (만족도 {w_satisfaction*100}%, 개인화 {w_uplift*100}%)")

# --- 4. Generate final recommendation table ---
# Sort by Hybrid_Score_5050 in descending order
final_hybrid_recommendations = merged_df.sort_values(
    by=['CLUSTER_ID', 'Hybrid_Score_5050'], 
    ascending=[True, False]
)

# Save as a new file
output_file = "recommendation_table_hybrid_5050.csv"
final_hybrid_recommendations.to_csv(output_file, index=False, encoding='utf-8-sig')

print("\n" + "="*50 + "\n")
print(f" [작업 완료] 50:50 하이브리드 추천 테이블을 '{output_file}'에 저장했습니다.")
print("이전 70:30 결과와 비교해보세요.")

# --- 5. Display Top 3 per cluster ---
print("\n--- [50:50 하이브리드 추천] 클러스터별 Top 3 ---")
cluster_ids = final_hybrid_recommendations['CLUSTER_ID'].unique()
cluster_ids.sort()

for cid in cluster_ids:
    print(f"\n[Cluster {cid}의 새로운 Top 3 추천 지역]")
    top_3_hybrid = final_hybrid_recommendations[
        final_hybrid_recommendations['CLUSTER_ID'] == cid
    ].head(3)
    
    print(top_3_hybrid[['TRAVEL_STATUS_DESTINATION', 'DGSTFN', 'Uplift_Score', 'Hybrid_Score_5050']].to_markdown(index=False, floatfmt=".3f"))
