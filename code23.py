# corrected_code23.py
import pandas as pd
import numpy as np
from io import StringIO

# ----------------------------------------------------
# DATASET INCLUDED INSIDE THE CODE (fixed: quoted fields with commas)
# ----------------------------------------------------
raw_data = """
EmployeeID,Name,Age,Department,Salary,JoinDate,Active
101, Alice Johnson, 29, hr , 52000, 2018-03-15, yes
102, Robert Brown,  , Finance, "$58,000" , 2019/07/22, Y
103, Emily Davis, 41, IT, 72000 , 2016-11-01, No
104, Michael Lee, 35, it , 68000, 2017-05-19, YES
105, Sarah Wilson,  , hr, "$49,500" ,  , no
106, Daniel White, 50, finance , 83000, 2014-09-10, y
107, Chris Evans, 28, Marketing , "$45,200", 2020-01-05, YES
108, Emma Thomas, 32, marketing, 47000, 2019-12-11, NO
109, Olivia Clark, , HR, "$51,000",  , Yes
110, Liam Walker, 38, IT, 76000, 2015-08-30, no
"""

# Convert multiline string to DataFrame
# use skipinitialspace to handle spaces after commas
data = pd.read_csv(StringIO(raw_data), skipinitialspace=True)
print("\n--- Original Raw Dataset ---")
print(data)

# ----------------------------------------------------
# FUNCTIONAL PROGRAMMING CLEANING FUNCTIONS
# ----------------------------------------------------

# Lambda function to trim whitespace
trim = lambda s: s.strip() if isinstance(s, str) else s

def clean_salary(s):
    """Remove $ and commas and convert to float; return NaN for invalid"""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    txt = str(s).replace("$", "").replace(",", "").strip()
    try:
        return float(txt)
    except Exception:
        return np.nan

def standardize_department(d):
    if pd.isna(d):
        return np.nan
    txt = str(d).strip().lower()
    mapping = {
        'hr': 'HR', 'human resources': 'HR',
        'it': 'IT', 'finance': 'Finance',
        'marketing': 'Marketing', 'sales': 'Sales'
    }
    return mapping.get(txt, txt.capitalize())

def clean_age(a):
    if pd.isna(a):
        return np.nan
    try:
        return int(float(str(a).strip()))
    except Exception:
        return np.nan

def clean_date(d):
    # try to parse, return NaT on failure
    try:
        return pd.to_datetime(d, errors='coerce')
    except Exception:
        return pd.NaT

def normalize_active(a):
    if pd.isna(a):
        return 0
    txt = str(a).strip().lower()
    return 1 if txt in ("yes", "y", "true", "1") else 0

def remove_duplicates(df, subset):
    return df.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)

def clean_dataframe(df):
    d = df.copy()

    # Trim whitespace in object columns
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].map(trim)

    # Parsers
    if 'Salary' in d.columns:
        d['Salary'] = d['Salary'].apply(clean_salary)
    if 'Age' in d.columns:
        d['Age'] = d['Age'].apply(clean_age)
    if 'Department' in d.columns:
        d['Department'] = d['Department'].apply(standardize_department)
    if 'JoinDate' in d.columns:
        d['JoinDate'] = d['JoinDate'].apply(clean_date)
    if 'Active' in d.columns:
        d['Active'] = d['Active'].apply(normalize_active)

    # Derive tenure if join date available
    if 'JoinDate' in d.columns:
        today = pd.Timestamp.today()
        d['Tenure_Years'] = d['JoinDate'].apply(lambda x: round((today - x).days / 365, 2) if not pd.isna(x) else np.nan)

    # Flag missing critical info
    d['Missing_Critical'] = d.apply(lambda r: pd.isna(r.get('Age')) or pd.isna(r.get('Salary')) or pd.isna(r.get('Department')), axis=1)

    # Remove duplicates using EmployeeID + Name if present
    subset = [col for col in ('EmployeeID', 'Name') if col in d.columns]
    if subset:
        d = remove_duplicates(d, subset=subset)

    return d.reset_index(drop=True)

# ----------------------------------------------------
# Run cleaning pipeline
# ----------------------------------------------------
cleaned = clean_dataframe(data)

print("\n--- Cleaned Dataset ---")
print(cleaned)

# Optionally save to CSV in your working folder
cleaned.to_csv("processed_dataset_incode.csv", index=False)
print("\nSaved cleaned file as 'processed_dataset_incode.csv'")
