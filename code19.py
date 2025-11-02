# ------------------------------------------------------
# Hierarchical Clustering and Comparison with K-Means
# ------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans

# 1️⃣ Create Dataset (same as previous K-Means example)
data = {
    'Customer_ID': range(1, 21),
    'Annual_Income': [40000, 42000, 43000, 45000, 47000, 49000, 50000, 52000, 54000, 56000,
                      60000, 62000, 65000, 67000, 70000, 75000, 80000, 85000, 90000, 95000],
    'Spending_Score': [80, 78, 65, 77, 70, 68, 75, 72, 74, 76, 70, 67, 60, 55, 50, 45, 40, 35, 30, 25]
}

df = pd.DataFrame(data)
X = df[['Annual_Income', 'Spending_Score']]

# 2️⃣ Hierarchical Clustering
Z = linkage(X, method='ward')

# 3️⃣ Plot Dendrogram
plt.figure(figsize=(8, 5))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Distance')
plt.show()

# 4️⃣ Determine Clusters (based on dendrogram observation)
hier_clusters = fcluster(Z, t=3, criterion='maxclust')
df['Hier_Cluster'] = hier_clusters

# 5️⃣ Compare with K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X)

# 6️⃣ Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(df['Annual_Income'], df['Spending_Score'], c=df['Hier_Cluster'], cmap='viridis')
axes[0].set_title("Hierarchical Clustering Results")
axes[0].set_xlabel("Annual_Income")
axes[0].set_ylabel("Spending_Score")

axes[1].scatter(df['Annual_Income'], df['Spending_Score'], c=df['KMeans_Cluster'], cmap='plasma')
axes[1].set_title("K-Means Clustering Results")
axes[1].set_xlabel("Annual_Income")
axes[1].set_ylabel("Spending_Score")

plt.tight_layout()
plt.show()

# 7️⃣ Display cluster comparison
print("\n--- Cluster Assignments ---")
print(df[['Customer_ID', 'Hier_Cluster', 'KMeans_Cluster']])
