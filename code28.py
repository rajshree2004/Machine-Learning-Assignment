# combine_files_dynamic.py
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ---------------------------
# 1) Create example files
# ---------------------------
# CSV content
csv_text = """id,name,score,age
1,Alice,85,23
2,Bob,90,25
3,Charlie,78,
4,Daniel,92,30
"""
Path("example.csv").write_text(csv_text, encoding="utf-8")

# Excel content (DataFrame -> Excel)
df_excel = pd.DataFrame({
    "id": [5, 6],
    "name": ["Eve", "Frank"],
    "score": [88, None],
    "age": [27, 29]
})
df_excel.to_excel("example.xlsx", index=False)

# JSON content
json_data = [
    {"id": 7, "name": "Grace", "score": 91, "age": 24},
    {"id": 8, "name": "Hank", "score": None, "age": None}
]
with open("example.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2)

# ---------------------------
# 2) Dynamic loader function
# ---------------------------
def load_file(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    suf = p.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(p)
    elif suf in (".xls", ".xlsx"):
        return pd.read_excel(p)
    elif suf == ".json":
        return pd.read_json(p)
    else:
        raise ValueError("Unsupported file type: " + suf)

# ---------------------------
# 3) Load all three files dynamically
# ---------------------------
files = ["example.csv", "example.xlsx", "example.json"]
dfs = [load_file(f) for f in files]

# ---------------------------
# 4) Combine and clean
# ---------------------------
df = pd.concat(dfs, ignore_index=True, sort=False)

# Basic cleaning:
# - trim string columns
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].astype(str).str.strip().replace({'nan': None})

# - convert numeric columns where possible
for col in ["score", "age", "id"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# - drop exact duplicates
df = df.drop_duplicates().reset_index(drop=True)

# - set 'id' as index if present and unique-ish; otherwise create RangeIndex
if "id" in df.columns and df["id"].notna().all() and df["id"].is_unique:
    df = df.set_index("id")
else:
    df.index.name = "row"

# - impute numeric NaNs with column mean (only for numeric cols)
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    mean_val = df[col].mean()
    df[col] = df[col].fillna(mean_val)

# ---------------------------
# 5) Simple descriptive statistic: print means
# ---------------------------
print("\nCombined DataFrame (cleaned):")
print(df)
print("\nColumn means (numeric columns):")
print(df[num_cols].mean())

# Save combined cleaned file optionally
df.to_csv("combined_cleaned.csv")
print("\nSaved combined_cleaned.csv")
