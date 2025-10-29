import numpy as np

# ------------------------------------------
# 1️⃣ Create two large random NumPy matrices (100x100)
# ------------------------------------------
# Matrix A → represents temperature data
# Matrix B → represents humidity data
A = np.random.rand(100, 100) * 50   # random values between 0–50
B = np.random.rand(100, 100) * 100  # random values between 0–100

print("Matrix A shape:", A.shape)
print("Matrix B shape:", B.shape)

# ------------------------------------------
# 2️⃣ Define functions for operations
# ------------------------------------------
def element_wise_operations(A, B):
    """Performs element-wise addition, subtraction, multiplication, and division."""
    add = A + B
    sub = A - B
    mul = A * B
    div = np.divide(A, B, out=np.zeros_like(A), where=B!=0)  # avoid division by zero
    return add, sub, mul, div

def statistical_summary(M):
    """Calculates mean, median, variance, and standard deviation for a matrix."""
    return {
        'mean': np.mean(M),
        'median': np.median(M),
        'variance': np.var(M),
        'std_dev': np.std(M)
    }

# ------------------------------------------
# 3️⃣ Perform element-wise operations
# ------------------------------------------
add, sub, mul, div = element_wise_operations(A, B)

# ------------------------------------------
# 4️⃣ Calculate statistical measures for each matrix
# ------------------------------------------
stats_A = statistical_summary(A)
stats_B = statistical_summary(B)

print("\n--- Statistical Summary ---")
print("Matrix A:", stats_A)
print("Matrix B:", stats_B)

# ------------------------------------------
# 5️⃣ Perform matrix multiplication (if possible)
# ------------------------------------------
matmul_result = np.dot(A, B)
print("\nMatrix Multiplication Result Shape:", matmul_result.shape)

# ------------------------------------------
# 6️⃣ Broadcasting example
# ------------------------------------------
# Add average of Matrix B to Matrix A using broadcasting
A_broadcast = A + np.mean(B)
print("\nBroadcasting Applied: A + mean(B)")
print("New Matrix A (sample):\n", A_broadcast[:3, :3])  # display small sample
