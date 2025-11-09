# ===============================================
# Clustering Evaluation (Precision @ K)
# ===============================================

import pandas as pd
import joblib
import warnings
import os

# --- 0. [Required] File path setup ---
warnings.filterwarnings('ignore')

# 1. K-Means model
MODEL_FILE = r"kmeans_user_cluster_model.joblib"
# 2. Test dataset
TEST_DATA_FILE = r"C:\Users\신승민\OneDrive\바탕 화면\scaled_results(3)\travel_val_scaled_standard.csv"
# 3. Hybrid recommendation table
RECO_TABLE_FILE = r"C:\Users\신승민\.vscode\recommendation_table_hybrid_5050.csv"

# 4. Evaluation criteria
K_FOR_HITRATE = 5
SATISFACTION_THRESHOLD = 4.0

print("--- [모델 평가 파이프라인 시작 (Precision @ K)] ---")

# --- 1. Load required files ---
print("1. 모델 2종 및 테스트 데이터를 로드합니다...")
try:
    kmeans_model = joblib.load(MODEL_FILE)
    reco_table = pd.read_csv(RECO_TABLE_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)
    print("-> 로드 성공!")
except Exception as e:
    print(f"파일 로드 중 오류 발생: {e}")
    exit()

# --- 2. Start evaluation ---
print(f"\n2. K={K_FOR_HITRATE}, 만족도 {SATISFACTION_THRESHOLD}점 이상 기준으로 평가를 시작합니다...")

# Numerator: overlap between recommendation and actual visit where satisfaction ≥ threshold
satisfied_hit_count = 0
# Denominator: all overlaps between recommendation and actual visit (regardless of satisfaction)
total_overlap_count = 0 

# Feature set used for clustering and prediction
features_for_clustering = ['GENDER', 'AGE_GRP'] + [f'TRAVEL_STYL_{i}' for i in range(1, 9)]

# Iterate over all users in the test dataset
for _, user_row in test_df.iterrows():

    actual_destination = user_row['TRAVEL_STATUS_DESTINATION']
    actual_satisfaction = user_row['DGSTFN']

    user_profile_scaled = pd.DataFrame([user_row], columns=features_for_clustering)
    predicted_cluster = kmeans_model.predict(user_profile_scaled)[0]
    
    reco_list_df = reco_table[reco_table['CLUSTER_ID'] == predicted_cluster]
    top_k_recommendations = reco_list_df.head(K_FOR_HITRATE)['TRAVEL_STATUS_DESTINATION'].values
    
    # --- 2-4. Precision evaluation logic ---
    
    # Condition 1: Check if actual visit appears in recommendation list (independent of satisfaction)
    is_match = actual_destination in top_k_recommendations
    
    # Denominator: increment when recommendation and actual visit overlap
    if is_match:
        total_overlap_count += 1
        
        # Numerator: among overlaps, count only those with satisfaction ≥ threshold
        is_satisfied = actual_satisfaction >= SATISFACTION_THRESHOLD
        
        if is_satisfied:
            satisfied_hit_count += 1

# --- 3. Final results ---
print("\n3. 평가 완료!")

precision = 0.0
if total_overlap_count > 0:
    # Precision = (satisfied overlaps) / (total overlaps)
    precision = (satisfied_hit_count / total_overlap_count) * 100
else:
    print("경고: 추천과 방문이 겹친 사례(overlap)가 0건입니다.")

print("\n" + "="*50)
print("  [지표: Precision @ K (추천의 '질')]")
print(f"  총 평가 사용자: {len(test_df)} 명")
print(f"  추천-방문 겹친 횟수 (분모): {total_overlap_count} 회")
print(f"  (그 중 4.0점 이상 만족) (분자): {satisfied_hit_count} 회")
print(f"  ▶ Precision @ {K_FOR_HITRATE}: {precision:.2f} %")
print("="*50)
print(f"\n(해석: 우리 모델의 추천과 사용자의 실제 방문이 겹쳤을 때,")
print(f"       그 방문은 {precision:.2f}% 확률로 '만족스러운' 방문이었습니다.)")
