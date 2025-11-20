# kfold_tuning_short.py — k-fold CV + GridSearchCV/RandomizedSearchCV (Iris example)
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Common CV splitter
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------- SVM with GridSearchCV ----------
svm_pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=False, random_state=42))])
svm_param_grid = {"svc__C": [0.1, 1, 10], "svc__kernel": ["linear", "rbf"], "svc__gamma": ["scale", "auto"]}
gs_svm = GridSearchCV(svm_pipe, svm_param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
gs_svm.fit(X_train, y_train)

# baseline (default SVC) vs tuned
base_svm = SVC(random_state=42).fit(X_train, y_train)
print("SVM baseline test acc:", accuracy_score(y_test, base_svm.predict(X_test)))
print("SVM tuned test acc :", accuracy_score(y_test, gs_svm.best_estimator_.predict(X_test)))
print("SVM best params    :", gs_svm.best_params_)

# ---------- Decision Tree with RandomizedSearchCV ----------
dt = DecisionTreeClassifier(random_state=42)
dt_param_dist = {"max_depth": [None, 2, 3, 4, 5, 6], "min_samples_split": [2,3,4,5], "criterion": ["gini","entropy"]}
rs_dt = RandomizedSearchCV(dt, dt_param_dist, n_iter=6, cv=cv, scoring="accuracy", random_state=42, n_jobs=-1)
rs_dt.fit(X_train, y_train)

base_dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print("\nDT baseline test acc:", accuracy_score(y_test, base_dt.predict(X_test)))
print("DT tuned test acc   :", accuracy_score(y_test, rs_dt.best_estimator_.predict(X_test)))
print("DT best params      :", rs_dt.best_params_)
