import pandas as pd
import numpy as np

# 1️⃣ Create dataset (Product Prices)
np.random.seed(42)
price = np.append(np.random.normal(500, 100, 100), [1200, 1300, 1500])  # outliers
ad_spend = np.random.normal(50, 10, 103)
df = pd.DataFrame({'Ad_Spend': ad_spend, 'Price': price})

print("Original Data Summary:")
print(df['Price'].describe())

# 2️⃣ Z-Score Method
z = np.abs((df['Price'] - df['Price'].mean()) / df['Price'].std())
df_z = df[z < 3]
print("\nAfter Z-score Outlier Removal:")
print(df_z['Price'].describe())

# 3️⃣ IQR Method
Q1, Q3 = df['Price'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df_iqr = df[(df['Price'] >= lower) & (df['Price'] <= upper)]

print("\nAfter IQR Outlier Removal:")
print(df_iqr['Price'].describe())

# 4️⃣ Compare mean before and after
print("\nMean Before Removal:", df['Price'].mean())
print("Mean After Z-score:", df_z['Price'].mean())
print("Mean After IQR:", df_iqr['Price'].mean())
