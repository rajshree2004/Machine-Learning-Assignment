# -----------------------------------------------
# Exploratory Data Analysis on COVID-19 Statistics
# -----------------------------------------------

import pandas as pd
import numpy as np
import os

# 1️⃣ Set working directory and load the dataset
print("Current working directory:", os.getcwd())
os.chdir(r"C:\Users\rajsh\OneDrive\Desktop\ML\ques 9")  # ✅ Make sure this path exists

# Load dataset
df = pd.read_csv("covid_data.csv", parse_dates=['Date'])

# 2️⃣ Basic dataset info
print("\n--- Dataset Info ---")
df.info()

# 3️⃣ Descriptive statistics
print("\n--- Descriptive Statistics ---")
print(df.describe())

# 4️⃣ Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 5️⃣ Trend analysis using NumPy and Pandas
# Ensure required columns exist
required_cols = {'Confirmed', 'Deaths', 'Recovered'}
if required_cols.issubset(df.columns):
    df['Active'] = df['Confirmed'] - (df['Deaths'] + df['Recovered'])
    df['Daily_Confirmed'] = df['Confirmed'].diff()
    df['7Day_Avg'] = df['Confirmed'].rolling(window=3).mean()
else:
    print("\n⚠️ Missing one or more required columns: Confirmed, Deaths, Recovered")
    print(f"Columns found: {df.columns.tolist()}")

# 6️⃣ Find key insights
print("\n--- Insights ---")
if 'Confirmed' in df.columns:
    max_cases = df['Confirmed'].max()
    max_date = df.loc[df['Confirmed'].idxmax(), 'Date']
    print(f"Highest Confirmed Cases: {max_cases} on {max_date.date()}")
if 'Deaths' in df.columns:
    print(f"Total Deaths: {df['Deaths'].sum()}")
if 'Recovered' in df.columns and 'Confirmed' in df.columns:
    recovery_rate = np.mean(df['Recovered'] / df['Confirmed'] * 100)
    print(f"Average Recovery Rate: {recovery_rate:.2f}%")
if 'Daily_Confirmed' in df.columns:
    print(f"Peak Daily Increase: {df['Daily_Confirmed'].max():.0f} cases")

# 7️⃣ Text-based trend summary (last 5 days)
print("\n--- Trend Summary (Last 5 Days) ---")
for i in range(len(df) - 5, len(df)):
    date = df.loc[i, 'Date']
    conf = df.loc[i, 'Confirmed']
    deaths = df.loc[i, 'Deaths']
    print(f"{date.date()} - Confirmed: {conf}, Deaths: {deaths}")

# 8️⃣ Save the processed data
df.to_csv("covid_data_analysis.csv", index=False)
print("\n✅ Processed data saved as 'covid_data_analysis.csv'")
