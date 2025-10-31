# -------------------------------------------
# Logistic Regression for Binary Classification
# -------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# 1️⃣ Create a sample dataset
data = {
    'Customer_ID': range(1, 21),
    'Age': [25, 45, 35, 50, 23, 40, 30, 60, 28, 48, 33, 55, 31, 26, 42, 37, 29, 53, 34, 41],
    'Monthly_Charges': [20, 80, 50, 90, 25, 65, 45, 100, 30, 75, 55, 85, 60, 35, 70, 52, 40, 88, 49, 72],
    'Tenure_Months': [5, 40, 20, 60, 6, 35, 15, 70, 10, 50, 22, 65, 28, 12, 45, 18, 14, 68, 21, 42],
    'Contract_Type': ['Month-to-Month', 'One year', 'Month-to-Month', 'Two year', 'Month-to-Month',
                      'One year', 'Two year', 'Two year', 'Month-to-Month', 'One year',
                      'Two year', 'Two year', 'One year', 'Month-to-Month', 'One year',
                      'Month-to-Month', 'One year', 'Two year', 'Month-to-Month', 'One year'],
    'Churn': [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)
print("\n--- Original Dataset ---")
print(df.head())

# 2️⃣ Encode categorical variables
label_encoder = LabelEncoder()
df['Contract_Type'] = label_encoder.fit_transform(df['Contract_Type'])

# 3️⃣ Define independent (X) and dependent (y) variables
X = df[['Age', 'Monthly_Charges', 'Tenure_Months', 'Contract_Type']]
y = df['Churn']

# 4️⃣ Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5️⃣ Feature scaling (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6️⃣ Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 7️⃣ Make predictions
y_pred = model.predict(X_test_scaled)

# 8️⃣ Evaluate performance
print("\n--- Model Evaluation ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall:", round(recall_score(y_test, y_pred), 3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
