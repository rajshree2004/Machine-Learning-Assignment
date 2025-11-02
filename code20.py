# ------------------------------------------------------
# Support Vector Machine (SVM) Classification
# Kernel Comparison: Linear vs RBF
# ------------------------------------------------------

import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Load Dataset (Iris)
iris = load_iris()
X = iris.data
y = iris.target

# 2️⃣ Split Data into Train & Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3️⃣ Standardize the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4️⃣ Define Kernels to Compare
kernels = ['linear', 'rbf']
results = []

for kernel in kernels:
    print(f"\n🔹 Training SVM with '{kernel}' kernel...")
    start_time = time.time()
    
    # Train the model
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    
    # Predict
    y_pred = svm.predict(X_test)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    train_time = time.time() - start_time
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Training Time: {train_time:.4f} seconds")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    results.append([kernel, acc, train_time])

# 5️⃣ Compare Results
results_df = pd.DataFrame(results, columns=['Kernel', 'Accuracy', 'Training Time'])
print("\n--- Kernel Comparison Summary ---")
print(results_df)
