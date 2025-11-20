# --------------------------------------------------------
# Q27: Regression + Classification (Two Models Each)
# --------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# --------------------------------------------------------
# 1️. REGRESSION PROBLEM (Housing Prices)
# --------------------------------------------------------

# Sample Housing Dataset (embedded)
reg_data = {
    "Area": [800, 1200, 1500, 1800, 2200, 2500, 3000, 3500],
    "Bedrooms": [1, 2, 2, 3, 3, 4, 4, 5],
    "Price": [75, 120, 150, 180, 220, 260, 310, 360]
}
df_reg = pd.DataFrame(reg_data)

X_reg = df_reg[["Area", "Bedrooms"]]
y_reg = df_reg["Price"]

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Model 1: Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
pred_lin = lin_reg.predict(X_test)

# Model 2: Non-linear Regression (Decision Tree)
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)
pred_tree = tree_reg.predict(X_test)

# Metrics
print("\n--- REGRESSION RESULTS ---")
print("Linear Regression R2:", r2_score(y_test, pred_lin))
print("Decision Tree R2:", r2_score(y_test, pred_tree))


# --------------------------------------------------------
# 2. CLASSIFICATION PROBLEM (Student Exam Pass/Fail)
# --------------------------------------------------------

# Sample Classification Dataset (embedded)
class_data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Attendance": [40, 50, 60, 65, 70, 80, 85, 90],
    "Pass": [0, 0, 0, 1, 1, 1, 1, 1]
}
df_class = pd.DataFrame(class_data)

X_clf = df_class[["StudyHours", "Attendance"]]
y_clf = df_class["Pass"]

scaler = StandardScaler()
X_clf = scaler.fit_transform(X_clf)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

# Model 1: Logistic Regression
log_clf = LogisticRegression()
log_clf.fit(X_train_c, y_train_c)
pred_log = log_clf.predict(X_test_c)

# Model 2: KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_c, y_train_c)
pred_knn = knn.predict(X_test_c)

print("\n--- CLASSIFICATION RESULTS ---")
print("Logistic Regression Accuracy:", accuracy_score(y_test_c, pred_log))
print("KNN Accuracy:", accuracy_score(y_test_c, pred_knn))
