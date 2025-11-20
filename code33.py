# student_predictive_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---- 1) Dataset (embedded) ----
data = {
    "gender": ["M","F","F","M","M","F","F","M","F","M"],
    "hours_studied": [10, 12, 9, 14, 7, 15, 8, 11, 13, 6],
    "attendance": [90, 95, 85, 80, 70, 100, 75, 88, 92, 65],
    "previous_grade": [75, 88, 70, 90, 60, 95, 65, 80, 85, 55],
    "pass_fail": ["Pass","Pass","Fail","Pass","Fail","Pass","Fail","Pass","Pass","Fail"]
}

df = pd.DataFrame(data)

# ---- 2) Data Wrangling ----
# Check for missing values (none here, but in real datasets we fill/drop)
print("Missing values:\n", df.isnull().sum())

# ---- 3) Preprocessing ----
# Encode categorical variable 'gender'
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

# Encode target variable
df['pass_fail'] = le.fit_transform(df['pass_fail'])  # Pass=1, Fail=0

# Features and target
X = df.drop('pass_fail', axis=1)
y = df['pass_fail']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- 4) Feature Selection ----
# Using feature importance from a RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:\n", importances)

# ---- 5) Train-Test Split and Model ----
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# ---- 6) Result Interpretation ----
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Highlight key interpretation
print("\nInterpretation: 'hours_studied' and 'previous_grade' are the most important features influencing student pass/fail outcome.")
