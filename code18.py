# ------------------------------------------------------
# K-Means Clustering with Elbow Method and Interpretation
# ------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load the dataset (Customer Segmentation Example)
data = {
    'CustomerID': range(1, 21),
    'Annual_Income': [45000, 54000, 58000, 61000, 72000, 75000, 80000, 85000, 90000, 95000,
                      40000, 42000, 48000, 52000, 60000, 68000, 78000, 83000, 87000, 93000],
    'Spending_Score': [80, 75, 70, 68, 65, 55, 50, 45, 40, 30,
                       82, 78, 75, 72, 68, 60, 52, 47, 43, 35]
}

df = pd.DataFrame(data)
X = df[['Annual_Income', 'Spending_Score']]

# 2️⃣ Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Determine optimal K using the Elbow Method
inertia = []
K_range = range(1, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# 4️⃣ Apply K-Means with chosen K (e.g., 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 5️⃣ Visualize the final clusters
plt.figure(figsize=(6, 4))
plt.scatter(df['Annual_Income'], df['Spending_Score'],
            c=df['Cluster'], cmap='viridis', s=100)
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

# 6️⃣ Display clustered data
print("\n--- Clustered Data ---")
print(df.sort_values(by='Cluster'))

# 7️⃣ Cluster Characteristics (Interpretation)
cluster_summary = df.groupby('Cluster')[['Annual_Income', 'Spending_Score']].mean()
print("\n--- Cluster Characteristics ---")
print(cluster_summary)
