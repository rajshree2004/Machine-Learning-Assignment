import pandas as pd

# ---- STEP 1: Choose your large CSV file (>100MB) ----
file_path = r"C:\Users\rajsh\Downloads\large_100mb_file.csv"


# ---- STEP 2: Read first few rows to detect optimal dtypes ----
sample = pd.read_csv(file_path, nrows=5000)

# Convert numeric columns to smallest possible dtypes
opt_dtypes = {}
for col in sample.columns:
    if sample[col].dtype == "int64":
        opt_dtypes[col] = "int32"
    elif sample[col].dtype == "float64":
        opt_dtypes[col] = "float32"
    else:
        opt_dtypes[col] = "category"

# ---- STEP 3: Load file in chunks with optimized dtypes ----
chunks = []
for chunk in pd.read_csv(file_path, dtype=opt_dtypes, chunksize=500000):  
    chunks.append(chunk)

df = pd.concat(chunks)

# ---- STEP 4: Set index for faster operations ----
df = df.reset_index(drop=True)

# ---- STEP 5: Simple calculation (mean of numeric columns) ----
print("Mean values:")
print(df.mean(numeric_only=True))

# ---- Documenting Improvements ----
print("\nMemory usage after optimization:", df.memory_usage(deep=True).sum() / (1024**2), "MB")
