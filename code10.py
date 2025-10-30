# -------------------------------------------
# Feature Engineering and Data Quality Improvement
# -------------------------------------------

import pandas as pd
import numpy as np
import os

# 1️⃣ Check if the dataset exists, otherwise create one
file_path = "sales_data.csv"

if not os.path.exists(file_path):
    print("⚠️ File not found. Creating a sample dataset...")
    data = {
        'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'Customer_ID': [1, 2, 3, 1, 2, 4, 5, 3, 2, 1],
        'Category': ['Grocery', 'Clothing', 'Electronics', 'Grocery', 'Clothing',
                     'Grocery', 'Electronics', 'Clothing', 'Grocery', 'Clothing'],
        'Quantity': [10, 5, 2, 8, 3, 6, 1, 4, 7, 2],
        'Unit_Price': [20, 50, 200, 25, 60, 30, 250, 55, 22, 65],
        'Invoice_ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print("✅ Sample dataset 'sales_data.csv' created.")
else:
    df = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully.")

print("\n--- Original Dataset ---")
print(df.head())

# 2️⃣ Create new features
df['Total_Revenue'] = df['Quantity'] * df['Unit_Price']
df['Customer_Frequency'] = df.groupby('Customer_ID')['Invoice_ID'].transform('count')
df['Category_Sales'] = df.groupby('Category')['Total_Revenue'].transform('sum')

# 3️⃣ Apply transformations (reduce skewness)
df['Log_Revenue'] = np.log1p(df['Total_Revenue'])

# 4️⃣ Apply discretization (bin continuous data)
df['Revenue_Level'] = pd.cut(
    df['Total_Revenue'],
    bins=[0, 100, 500, 1000, 2000],
    labels=['Low', 'Medium', 'High', 'Very High']
)

# 5️⃣ Detect and remove outliers using IQR method
Q1 = df['Total_Revenue'].quantile(0.25)
Q3 = df['Total_Revenue'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[~((df['Total_Revenue'] < (Q1 - 1.5 * IQR)) | (df['Total_Revenue'] > (Q3 + 1.5 * IQR)))]

# 6️⃣ Display descriptive statistics
print("\n--- Descriptive Statistics ---")
print(df_clean.describe())

# 7️⃣ Display info summary
print("\n--- Dataset Info ---")
print(df_clean.info())

# 8️⃣ Save cleaned data
df_clean.to_csv("sales_data_cleaned.csv", index=False)
print("\n✅ Cleaned dataset saved as 'sales_data_cleaned.csv'")
