import numpy as np
import statistics as stats  # For built-in Python comparisons

# 1️⃣ Create sample weather data using NumPy arrays
# Data for 7 days (rows = days, columns = [Temperature, Humidity, Rainfall])
weather_data = np.array([
    [30, 70, 5],
    [32, 65, 0],
    [29, 80, 10],
    [35, 60, 2],
    [33, 75, 7],
    [28, 85, 15],
    [31, 72, 4]
])

print("Original Weather Data:\n", weather_data)

# 2️⃣ Reshape data (e.g., flatten to 1D and back to 2D)
reshaped_data = weather_data.reshape(3, 7)  # 3 rows, 7 columns
print("\nReshaped Data (3x7):\n", reshaped_data)

# 3️⃣ Broadcasting example — convert temperature from °C to °F
# Formula: (°C × 9/5) + 32
temp_c = weather_data[:, 0]
temp_f = temp_c * 9/5 + 32
print("\nTemperatures in °F (Broadcasting applied):\n", temp_f)

# 4️⃣ Calculate statistical measures using NumPy
mean_values = np.mean(weather_data, axis=0)
median_values = np.median(weather_data, axis=0)
variance_values = np.var(weather_data, axis=0)
std_dev_values = np.std(weather_data, axis=0)

print("\n--- NumPy Statistical Measures ---")
print("Mean (Temp, Humidity, Rainfall):", mean_values)
print("Median (Temp, Humidity, Rainfall):", median_values)
print("Variance (Temp, Humidity, Rainfall):", variance_values)
print("Standard Deviation (Temp, Humidity, Rainfall):", std_dev_values)

# 5️⃣ Compare with built-in Python functions (for temperature)
temps = weather_data[:, 0]

py_mean = stats.mean(temps.tolist())
py_median = stats.median(temps.tolist())
py_variance = stats.variance(temps.tolist())
py_std_dev = stats.stdev(temps.tolist())

print("\n--- Comparison (Temperature only) ---")
print(f"Mean → NumPy: {mean_values[0]:.2f}, Python: {py_mean:.2f}")
print(f"Median → NumPy: {median_values[0]:.2f}, Python: {py_median:.2f}")
print(f"Variance → NumPy: {variance_values[0]:.2f}, Python: {py_variance:.2f}")
print(f"Std Dev → NumPy: {std_dev_values[0]:.2f}, Python: {py_std_dev:.2f}")
