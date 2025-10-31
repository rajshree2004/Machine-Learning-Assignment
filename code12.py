import pandas as pd
import numpy as np

# sample data (same as yours)
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': ['25', '-30', '40', 'twenty', '35'],       # some invalid / strings
    'Salary': ['50000', '60000', 'abc', '70000', '80000'],  # one invalid
    'Department': ['HR', 'IT', 'Finance', 'IT', 'HR']
}
df_sample = pd.DataFrame(data)

def clean_and_validate(df, impute_invalid=False):
    """
    - df: pandas DataFrame
    - impute_invalid: if True, invalid numeric values (NaN or out-of-range) are replaced by column median
    Returns: (df_cleaned, df_invalid)
    """
    df = df.copy()  # don't modify original
    # 1) Coerce numeric columns: try to infer Age and Salary columns exist
    numeric_cols = []
    for candidate in ['Age', 'Salary']:
        if candidate in df.columns:
            # coerce non-numeric to NaN
            df[candidate] = pd.to_numeric(df[candidate], errors='coerce')
            numeric_cols.append(candidate)

    # 2) Define business rules
    # Age: must be > 0 and reasonable (< 120)
    # Salary: must be > 0
    valid_mask = pd.Series(True, index=df.index)

    if 'Age' in df.columns:
        age_ok = df['Age'].notna() & (df['Age'] > 0) & (df['Age'] < 120)
        valid_mask &= age_ok

    if 'Salary' in df.columns:
        sal_ok = df['Salary'].notna() & (df['Salary'] > 0)
        valid_mask &= sal_ok

    # 3) Flag validity
    df['Valid'] = valid_mask

    # 4) Capture invalid rows
    df_invalid = df.loc[~df['Valid']].copy()

    # 5) Optional: impute invalid numeric values with median (if requested)
    if impute_invalid and len(numeric_cols) > 0:
        for col in numeric_cols:
            median = df.loc[df[col].notna(), col].median()
            # replace NaN or out-of-range (we'll use Valid mask to decide)
            # For Age/Salary, replace only where invalid (either NaN or failing rule)
            invalid_idx = df.index[~valid_mask]
            df.loc[invalid_idx, col] = df.loc[invalid_idx, col].fillna(median)
        # after imputation recompute validity
        valid_mask2 = pd.Series(True, index=df.index)
        if 'Age' in df.columns:
            valid_mask2 &= df['Age'].notna() & (df['Age'] > 0) & (df['Age'] < 120)
        if 'Salary' in df.columns:
            valid_mask2 &= df['Salary'].notna() & (df['Salary'] > 0)
        df['Valid'] = valid_mask2
        df_invalid = df.loc[~df['Valid']].copy()  # updated invalids after imputation

    return df, df_invalid

# === RUN the function ===
print("\n--- Original Dataset ---")
print(df_sample)

df_cleaned, df_invalid = clean_and_validate(df_sample, impute_invalid=False)

print("\n--- Invalid Records (flagged, impute_invalid=False) ---")
print(df_invalid)

print("\n--- Cleaned & Flagged Dataset ---")
print(df_cleaned)

# If you want automatic imputation instead of just flagging:
df_imputed, df_invalid_after_impute = clean_and_validate(df_sample, impute_invalid=True)

print("\n--- After Imputation (invalids remaining) ---")
print(df_invalid_after_impute)
print("\n--- Dataset after Imputation ---")
print(df_imputed)
