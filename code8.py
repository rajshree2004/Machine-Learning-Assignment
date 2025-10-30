import pandas as pd
import numpy as np

# 1️⃣ Load dataset
df = pd.read_csv(r"C:\Users\rajsh\OneDrive\Desktop\ML\ques 8\stock_prices.csv", parse_dates=['Date'])

df.set_index('Date', inplace=True)

# 2️⃣ Resampling – Weekly average closing price
weekly_avg = df['Close'].resample('W').mean()

# 3️⃣ Rolling mean – 3-day moving average
df['Rolling_Mean'] = df['Close'].rolling(window=3).mean()

# 4️⃣ Differencing – Daily change in price
df['Diff'] = df['Close'].diff()

# 5️⃣ Manual Regression using NumPy
df['Day'] = np.arange(len(df))
df = df.dropna()

# Formula: y = a*x + b
x = df['Day']
y = df['Close']

a, b = np.polyfit(x, y, 1)  # slope (a), intercept (b)
df['Predicted'] = a * x + b

# 6️⃣ Display outputs
print("=== Weekly Average Closing Price ===")
print(weekly_avg)

print("\n=== Data with Rolling Mean and Differencing ===")
print(df[['Close', 'Rolling_Mean', 'Diff']].head(10))

print("\n=== Manual Regression Coefficients ===")
print(f"Slope (a): {a:.4f}")
print(f"Intercept (b): {b:.4f}")
