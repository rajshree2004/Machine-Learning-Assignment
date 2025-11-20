import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans

# ---- 1) Data Import & Cleaning ----
# Assume you have downloaded the Kaggle dataset `StudentsPerformance.csv`
import pandas as pd

# Change path according to your system
df = pd.read_csv(r"C:\Users\rajsh\Downloads\StudentsPerformance.csv")


# Quick cleaning
print("Missing values per column:\n", df.isnull().sum())
df.drop_duplicates(inplace=True)

# ---- 2) EDA & Preprocessing ----
# EDA: Visualize pairwise relationships
sns.pairplot(df, vars=["math score", "reading score", "writing score"], hue="gender")
plt.show()

# Encode categorical columns
le = LabelEncoder()
for col in ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]:
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop(["math score", "reading score", "writing score"], axis=1)
y = df["math score"]  # example: predict math score (regression)  
# (Alternatively, you could classify pass/fail, or average score)

# Scale numeric features
num_cols = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[num_cols])

# ---- 3) Supervised Model ----
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train.astype(int))  # converting to int for classification example
y_pred = clf.predict(X_test)

print("Supervised Model Evaluation:")
print("Confusion Matrix:\n", confusion_matrix(y_test.astype(int), y_pred))
print("Classification Report:\n", classification_report(y_test.astype(int), y_pred))

# ---- 4) Unsupervised Model (Clustering) ----
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Visualize clusters in two score dimensions
plt.figure(figsize=(6,5))
sns.scatterplot(x=df["reading score"], y=df["writing score"], hue=df["cluster"], palette="Set2")
plt.title("KMeans Clustering (3 Clusters)")
plt.show()

# ---- 5) Insights & Reporting ----
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importances:\n", importances)

print("\nInsights:")
print("- The most important features for predicting (math) score are:", list(importances.index[:3]))
print("- Clustering shows that students group into clusters based on encoded demographic / prep-course features, which may reveal sub‑populations with different performance patterns.")
