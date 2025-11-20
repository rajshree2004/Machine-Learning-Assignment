# vis_with_pandas_seaborn.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# ---- Load dataset into a DataFrame ----
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = pd.Categorical([iris.target_names[i] for i in iris.target])

# ---- 1) Correlation heatmap (features only) ----
plt.figure(figsize=(8,6))
corr = df[iris.feature_names].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# ---- 2) Scatter plot with regression fit (example: petal length vs petal width) ----
plt.figure(figsize=(6,5))
sns.scatterplot(data=df, x="petal length (cm)", y="petal width (cm)", hue="species", s=60)
sns.regplot(data=df, x="petal length (cm)", y="petal width (cm)", scatter=False, color="black")
plt.title("Petal length vs Petal width (with regression line)")
plt.tight_layout()
plt.show()


# ---- 3) Pair plot (scatter matrix colored by species) ----
sns.pairplot(df, vars=iris.feature_names, hue="species", corner=True, plot_kws={'s':40})
plt.suptitle("Pairplot of Iris features", y=1.02)
plt.show()

# ---- 4) Feature importance via RandomForest ----
X = df[iris.feature_names]
y = df['target']
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=iris.feature_names).sort_values(ascending=True)

plt.figure(figsize=(6,4))
imp.plot(kind='barh')
plt.xlabel("Importance")
plt.title("Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()
