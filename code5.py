# -----------------------------------------------------
# Data Cleaning and Scaling Techniques (Titanic Dataset)
# -----------------------------------------------------

import pandas as pd
import numpy as np

# 1️⃣ Load dataset
df = pd.read_csv(r"C:\Users\rajsh\OneDrive\Desktop\ML\ques 5\titanic.csv")



# 2️⃣ Handle missing values
df.fillna({
    'Age': df['Age'].mean(),
    'Embarked': df['Embarked'].mode()[0]
}, inplace=True)

# 3️⃣ Remove duplicates
df.drop_duplicates(inplace=True)

# 4️⃣ Select numeric columns for scaling
numeric_cols = ['Age', 'Fare']
data = df[numeric_cols]

# 5️⃣ Min-Max Scaling (Manual)
df['Age_MinMax'] = (data['Age'] - data['Age'].min()) / (data['Age'].max() - data['Age'].min())
df['Fare_MinMax'] = (data['Fare'] - data['Fare'].min()) / (data['Fare'].max() - data['Fare'].min())

# 6️⃣ Z-score Normalization (Manual)
df['Age_Zscore'] = (data['Age'] - data['Age'].mean()) / data['Age'].std()
df['Fare_Zscore'] = (data['Fare'] - data['Fare'].mean()) / data['Fare'].std()

# 7️⃣ Log Transformation
df['Fare_Log'] = np.log1p(data['Fare'])  # log(1 + x) handles 0 safely

# 8️⃣ Compare results
print("\nOriginal Mean Fare:", round(df['Fare'].mean(), 2))
print("MinMax Mean Fare:", round(df['Fare_MinMax'].mean(), 2))
print("Zscore Mean Fare:", round(df['Fare_Zscore'].mean(), 2))
print("Log Mean Fare:", round(df['Fare_Log'].mean(), 2))

# 9️⃣ Save cleaned dataset
df.to_csv('titanic_cleaned_scaled.csv', index=False)

print("\n✅ Cleaned & scaled data saved as 'titanic_cleaned_scaled.csv'")
