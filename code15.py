# ------------------------------------------------------
# Multi-Model Classification: Iris Dataset Comparison
# ------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# 1️⃣ Load the dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=['target'])

# 2️⃣ Data Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels for ANN/ROC
lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 3️⃣ Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "ANN": MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
}

# 4️⃣ Train & Evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc = roc_auc_score(y_encoded, model.predict_proba(X_scaled), multi_class='ovr')
    
    results.append([name, acc, prec, rec, f1, roc])

# 5️⃣ Results Summary
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
print("\n--- Model Comparison ---")
print(results_df.sort_values(by='Accuracy', ascending=False))

# 6️⃣ Identify Best Model
best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
print(f"\n✅ Best Performing Model: {best_model}")
