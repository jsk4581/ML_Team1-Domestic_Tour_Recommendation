# ===============================================
# Content Filtering Based Recommendation System for Tourist Regions
# ===============================================

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_excel("지역별_관광속성_행렬.xlsx")

# Extract only attributes (excluding the first column AREA_NM)
features = df.drop(columns=["AREA_NM"]).values
names = df["AREA_NM"].values

# Calculating the cosine similarity matrix
similarity_matrix = cosine_similarity(features)

# Target region settings
target = "강원도"
idx = list(names).index(target)

# Similarity sorting between target and other regions
similarities = list(enumerate(similarity_matrix[idx]))
sorted_sim = sorted(similarities, key=lambda x: x[1], reverse=True)

print(f"'{target}'와 가장 유사한 지역 Top 5:")
for i, score in sorted_sim[1:6]:  # Excluding myself
    print(f"- {names[i]}: {score:.3f}")