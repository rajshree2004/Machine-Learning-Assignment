# ------------------------------------------------------
# Customer Segmentation using K-Means and Hierarchical Clustering
# ------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 1️⃣ Create synthetic customer dataset
data = {
    'CustomerID': range(1, 21),
    'Age': [25, 45, 31, 35, 40, 23, 30, 48, 50, 29, 33, 42, 55, 28, 32, 47, 38, 41, 27, 36],
    'Annual_Income': [40000, 80000, 52000, 58000, 62000, 38000, 45000, 85000, 90000, 43000,
                      50000, 70000, 95000, 42000, 48000, 78000, 64000, 72000, 39000, 60000],
    'Spending_Score': [65, 40, 75, 70, 60, 80, 68, 35, 30, 77, 72, 55, 25, 75, 71, 42, 59, 53, 79, 67]
}
df = pd.DataFrame(data)
df.to_csv("customer_segmentation.csv", index=False)

print("\n--- Sample Customer Dataset ---")
print(df.head())

# 2️⃣ Feature selection and scaling
X = df[['Age', 'Annual_Income', 'Spending_Score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# 4️⃣ Hierarchical Clustering
hier = AgglomerativeClustering(n_clusters=3)
df['Hier_Cluster'] = hier.fit_predict(X_scaled)

# 5️⃣ Visualization - K-Means
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.scatterplot(x=df['Annual_Income'], y=df['Spending_Score'], hue=df['KMeans_Cluster'], palette='Set1', s=80)
plt.title('K-Means Clustering Results')

# 6️⃣ Visualization - Hierarchical Clustering
plt.subplot(1,2,2)
sns.scatterplot(x=df['Annual_Income'], y=df['Spending_Score'], hue=df['Hier_Cluster'], palette='Set2', s=80)
plt.title('Hierarchical Clustering Results')
plt.tight_layout()
plt.show()

# 7️⃣ Dendrogram (for Hierarchical Clustering)
plt.figure(figsize=(8,5))
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Distance')
plt.show()

# 8️⃣ Compare Clustering Results
print("\n--- Cluster Comparison ---")
print(df[['CustomerID', 'KMeans_Cluster', 'Hier_Cluster']])

# 9️⃣ Insights
print("\n📊 Insights:")
print("• K-Means and Hierarchical clustering both grouped customers based on income and spending behavior.")
print("• High-income & low-spending customers can be targeted for promotions.")
print("• Moderate-income & high-spending customers are likely loyal customers.")
print("• Young, low-income, high-spending customers may respond well to discounts.")
