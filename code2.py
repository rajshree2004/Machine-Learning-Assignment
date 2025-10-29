# -------------------------------------------------------------
# Combine and Clean Multiple World Happiness Report Datasets
# -------------------------------------------------------------

import pandas as pd

# 1️⃣ Load multiple datasets
import pandas as pd

df_2015 = pd.read_csv(r'C:\Users\rajsh\OneDrive\Desktop\ML\ques 2\2015.csv')
df_2016 = pd.read_csv(r'C:\Users\rajsh\OneDrive\Desktop\ML\ques 2\2016.csv')
df_2017 = pd.read_csv(r'C:\Users\rajsh\OneDrive\Desktop\ML\ques 2\2017.csv')


# Add a 'Year' column to each
df_2015['Year'] = 2015
df_2016['Year'] = 2016
df_2017['Year'] = 2017

# 2️⃣ Standardize column names (since each file may differ slightly)
df_2015.columns = df_2015.columns.str.strip().str.lower().str.replace(' ', '_')
df_2016.columns = df_2016.columns.str.strip().str.lower().str.replace(' ', '_')
df_2017.columns = df_2017.columns.str.strip().str.lower().str.replace(' ', '_')

# Merge all datasets
df = pd.concat([df_2015, df_2016, df_2017], ignore_index=True)
print("✅ Combined Dataset Shape:", df.shape)

# 3️⃣ Remove duplicates
df = df.drop_duplicates()
print("\nAfter Removing Duplicates:", df.shape)

# 4️⃣ Handle missing values
print("\nMissing Values Before Cleaning:\n", df.isnull().sum())

# Fill numeric missing values with mean
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Fill categorical missing values with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# 5️⃣ Create a new calculated feature
# Example: Happiness-to-GDP Ratio
if 'happiness_score' in df.columns and 'economy_(gdp_per_capita)' in df.columns:
    df['happiness_to_gdp_ratio'] = df['happiness_score'] / df['economy_(gdp_per_capita)']

# 6️⃣ Export cleaned data
df.to_csv('world_happiness_combined_cleaned.csv', index=False)

print("\n✅ Cleaned combined dataset saved as 'world_happiness_combined_cleaned.csv'")
print("\n--- Sample of Cleaned Data ---")
print(df.head())
