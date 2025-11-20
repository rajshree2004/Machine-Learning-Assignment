# imbalanced_classification.py
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ---- 1) Create imbalanced dataset ----
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                           n_redundant=2, n_clusters_per_class=1, 
                           weights=[0.9, 0.1], flip_y=0, random_state=42)
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
df['target'] = y

# ---- 2) Train-test split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ---- 3) Train classifier on original imbalanced data ----
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Before SMOTE:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---- 4) Apply SMOTE to balance classes ----
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# ---- 5) Train classifier on balanced data ----
clf.fit(X_res, y_res)
y_pred_res = clf.predict(X_test)
print("After SMOTE:")
print(confusion_matrix(y_test, y_pred_res))
print(classification_report(y_test, y_pred_res))
