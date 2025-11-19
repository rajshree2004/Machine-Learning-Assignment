# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer

# Step 2: Create sample dataset
np.random.seed(42)
data = pd.DataFrame({
    'Income': np.random.exponential(scale=50000, size=200),  # continuous skewed data
    'Purchase': np.random.randint(0, 2, 200)  # target variable
})

print("Original Data Sample:\n", data.head())

# Step 3: Apply transformations
data['log_income'] = np.log1p(data['Income'])
data['sqrt_income'] = np.sqrt(data['Income'])
data['boxcox_income'], _ = stats.boxcox(data['Income'] + 1)  # Box-Cox requires positive values

# Step 4: Apply discretization (binning)
binning = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
data['binned_income'] = binning.fit_transform(data[['Income']])

# Step 5: Visualization of transformations
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(data['Income'], bins=30, ax=axes[0,0], color='blue', kde=True)
axes[0,0].set_title('Original Income Distribution')
sns.histplot(data['log_income'], bins=30, ax=axes[0,1], color='green', kde=True)
axes[0,1].set_title('Log-Transformed Income')
sns.histplot(data['sqrt_income'], bins=30, ax=axes[1,0], color='orange', kde=True)
axes[1,0].set_title('Square Root Transformed Income')
sns.histplot(data['boxcox_income'], bins=30, ax=axes[1,1], color='red', kde=True)
axes[1,1].set_title('Box-Cox Transformed Income')
plt.tight_layout()
plt.show()

# Step 6: Correlation with target
corrs = {
    'Original': data['Income'].corr(data['Purchase']),
    'Log': data['log_income'].corr(data['Purchase']),
    'Sqrt': data['sqrt_income'].corr(data['Purchase']),
    'Box-Cox': data['boxcox_income'].corr(data['Purchase']),
    'Binned': data['binned_income'].corr(data['Purchase'])
}

print("\nCorrelation of each transformation with Purchase target:\n")
for key, val in corrs.items():
    print(f"{key}: {val:.4f}")
