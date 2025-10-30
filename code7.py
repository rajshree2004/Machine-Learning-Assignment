import pandas as pd
import numpy as np

# 1️⃣ Create sample dataset
data = {
    'Region': ['North', 'South', 'East', 'West', 'North', 'East', 'South', 'West'],
    'Product': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'],
    'Sales': [450, 600, 300, 800, 1200, 700, 200, 950],
    'Quantity': [5, 10, 3, 8, 12, 9, 2, 11]
}

df = pd.DataFrame(data)

# 2️⃣ Filter records with Sales > 500
filtered_df = df[df['Sales'] > 500]

# 3️⃣ Group by Region and calculate total sales & average quantity
grouped = filtered_df.groupby('Region').agg({
    'Sales': ['sum', 'mean', 'count'],
    'Quantity': ['sum', 'mean']
}).reset_index()

# 4️⃣ Generate descriptive report
print("=== Dataset Information ===")
print(df.info())

print("\n=== Descriptive Statistics ===")
print(df.describe())

print("\n=== Filtered Records (Sales > 500) ===")
print(filtered_df)

print("\n=== Aggregated Sales by Region ===")
print(grouped)
