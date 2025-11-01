# ------------------------------------------------------
# K-Nearest Neighbors (KNN) with Cross-Validation
# Using Iris Dataset (CSV Version)
# ------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# 1️⃣ Load and Save Dataset as CSV (run once)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.to_csv("iris_knn_dataset.csv", index=False)
print("✅ Dataset saved as 'iris_knn_dataset.csv'")

# 2️⃣ Load Dataset from CSV
df = pd.read_csv("iris_knn_dataset.csv")
X = df.drop("target", axis=1)
y = df["target"]

# 3️⃣ Data Preprocessing — Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Define different K values to test
k_values = [3, 5, 7]

# 5️⃣ Evaluate models using Cross-Validation
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=cv, scoring='accuracy')
    results[k] = scores.mean()
    print(f"K = {k} | Cross-Validation Accuracy = {scores.mean():.4f}")

# 6️⃣ Identify Best K
best_k = max(results, key=results.get)
print(f"\n✅ Best K value: {best_k} (Accuracy = {results[best_k]:.4f})")

# 7️⃣ Train final model with best K
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_scaled, y)
print("🎯 Final KNN model trained successfully!")
