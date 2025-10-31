# --------------------------------------------
# Linear Regression with Feature Scaling & Regularization
# --------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1️⃣ Create a sample dataset (House Prices)
np.random.seed(42)
data = {
    'Area': np.random.randint(800, 4000, 50),          # square feet
    'Bedrooms': np.random.randint(1, 5, 50),           # number of bedrooms
    'Bathrooms': np.random.randint(1, 4, 50),          # number of bathrooms
    'Age': np.random.randint(1, 30, 50),               # age of house in years
}
df = pd.DataFrame(data)
df['Price'] = 50000 + (df['Area'] * 150) + (df['Bedrooms'] * 20000) + \
              (df['Bathrooms'] * 15000) - (df['Age'] * 800) + np.random.randint(-20000, 20000, 50)

print("\n--- Original Dataset ---")
print(df.head())

# 2️⃣ Split dataset into features (X) and target (y)
X = df[['Area', 'Bedrooms', 'Bathrooms', 'Age']]
y = df['Price']

# 3️⃣ Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Apply Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Scaled Features Example (first 5 rows) ---")
print(X_train_scaled[:5])

# 5️⃣ Train Linear Regression Model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# 6️⃣ Predictions and Evaluation
y_pred = lr.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n--- Linear Regression Performance ---")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# 7️⃣ Regularization (to avoid overfitting)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
lasso_pred = lasso.predict(X_test_scaled)

# 8️⃣ Compare Models
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name}")
    print(f"R²: {r2_score(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")

evaluate_model("Ridge Regression", y_test, ridge_pred)
evaluate_model("Lasso Regression", y_test, lasso_pred)

# 9️⃣ Discussion (text summary)
print("\n--- Insights ---")
print("Feature scaling ensured fair comparison across features with different units (e.g., Area vs Bedrooms).")
print("Linear Regression performed well with high R², but regularization (Ridge/Lasso) helps reduce overfitting.")
print("Ridge applies L2 penalty to shrink coefficients; Lasso applies L1 and can zero out less important features.")
