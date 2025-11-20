import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---- Sample dataset (housing prices) ----
data = {
    "Area": [800, 900, 1000, 1100, 1200, 1500, 1800, 2000],
    "Bedrooms": [2, 2, 3, 3, 3, 4, 4, 5],
    "Price": [80, 95, 110, 130, 150, 180, 210, 240]
}
df = pd.DataFrame(data)

X = df[["Area", "Bedrooms"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---- Linear Regression ----
lin = LinearRegression()
lin.fit(X_train, y_train)
pred_lin = lin.predict(X_test)

# ---- Polynomial Regression (degree=2) ----
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)
pred_poly = poly_model.predict(X_poly_test)

# ---- Ridge Regularization (reduces overfitting) ----
ridge = Ridge(alpha=1)
ridge.fit(X_poly_train, y_train)
pred_ridge = ridge.predict(X_poly_test)

# ---- Evaluation Function ----
def evaluate(y_true, y_pred, name):
    print(f"\n{name}:")
    print("R²:", r2_score(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("MAE:", mean_absolute_error(y_true, y_pred))

evaluate(y_test, pred_lin, "Linear Regression")
evaluate(y_test, pred_poly, "Polynomial Regression")
evaluate(y_test, pred_ridge, "Polynomial + Ridge (Regularized)")
