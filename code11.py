# ----------------------------------------------------
# Outlier Detection using Z-Score and IQR Methods
# ----------------------------------------------------

import pandas as pd
import numpy as np

# 1️⃣ Create a numerical dataset (Sales data)
data = {
    'Sales': [200, 220, 250, 300, 280, 275, 260, 1000, 150, 270,
              265, 290, 1200, 240, 255, 230, 310, 285, 950, 210]
}
df = pd.DataFrame(data)
print("\n--- Original Dataset ---")
print(df)

# 2️⃣ Z-Score Method
mean = df['Sales'].mean()
std = df['Sales'].std()
df['Z_Score'] = (df['Sales'] - mean) / std
z_outliers = df[(df['Z_Score'] > 3) | (df['Z_Score'] < -3)]
print("\n--- Z-Score Outliers ---")
print(z_outliers)

# 3️⃣ IQR Method
Q1 = df['Sales'].quantile(0.25)
Q3 = df['Sales'].quantile(0.75)
IQR = Q3 - Q1
iqr_outliers = df[(df['Sales'] < (Q1 - 1.5 * IQR)) | (df['Sales'] > (Q3 + 1.5 * IQR))]
print("\n--- IQR Outliers ---")
print(iqr_outliers)

# 4️⃣ Handle Outliers (Capping)
upper_limit = Q3 + 1.5 * IQR
lower_limit = Q1 - 1.5 * IQR
df['Sales_Cleaned'] = np.where(df['Sales'] > upper_limit, upper_limit,
                               np.where(df['Sales'] < lower_limit, lower_limit, df['Sales']))

# 5️⃣ Compare before and after cleaning
print("\n--- Before Cleaning ---")
print(df['Sales'].describe())
print("\n--- After Cleaning ---")
print(df['Sales_Cleaned'].describe())

# 6️⃣ Save cleaned data
df.to_csv("outlier_detection_result.csv", index=False)
print("\n✅ Cleaned dataset saved as 'outlier_detection_result.csv'")
