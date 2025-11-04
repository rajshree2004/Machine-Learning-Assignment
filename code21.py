# ----------------------------------------------------------
# END-TO-END DATA ANALYSIS WORKFLOW: TITANIC SURVIVAL PREDICTION
# ----------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# 1️⃣ Load Dataset
# You can replace this path with your CSV file path
data = pd.read_csv(r"C:\Users\rajsh\OneDrive\Desktop\ML\ques 21\titanic_dataset.csv")

print("\n--- Step 1: Dataset Loaded ---")
print(data.head())

# 2️⃣ Handle Missing Values
# Fill missing 'Age' with median and 'Embarked' with mode
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True, errors='ignore')
 # Drop unnecessary columns

print("\n--- Step 2: Missing Values Handled ---")
print(data.isnull().sum())

# 3️⃣ Encode Categorical Variables
label_enc = LabelEncoder()
data['Sex'] = label_enc.fit_transform(data['Sex'])
data['Embarked'] = label_enc.fit_transform(data['Embarked'])

# 4️⃣ Split Features and Target
X = data.drop('Survived', axis=1)
y = data['Survived']

# 5️⃣ Split into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Feature Scaling / Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7️⃣ Model Selection & Training (Logistic Regression)
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# 8️⃣ Predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# 9️⃣ Evaluation Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

# 🔟 Reporting Results
print("\n--- Step 9: Model Evaluation ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 1️⃣1️⃣ Short Summary
print("\n✅ End-to-End Workflow Summary:")
print("""
Dataset: Titanic Survival Data
Preprocessing:
 - Missing 'Age' handled with median
 - Missing 'Embarked' handled with mode
 - Categorical features encoded using LabelEncoder
 - Features scaled using StandardScaler

Model: Logistic Regression
Evaluation Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
Interpretation:
 - High ROC-AUC and F1 indicate good balance between precision and recall.
 - Model can help predict passenger survival probability.
""")
