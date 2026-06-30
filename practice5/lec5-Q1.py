import numpy as np

def main():
    # Step 1: Generate a random 100 x 3 dataset using np.random.default_rng()
    rng = np.random.default_rng(seed=42)  # Use a seed for reproducibility
    X = rng.random((100, 3))
    print("--- Step 1: Generate Dataset ---")
    print(f"Shape of X: {X.shape}")
    print(f"First 3 rows of X:\n{X[:3]}\n")

    # Step 2: Normalize each feature (subtract mean, divide by std) via vectorized operation
    # Compute mean and std along the column axis (axis=0)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0, ddof=1)  # Using sample standard deviation (ddof=1)
    
    X_norm = (X - means) / stds
    print("--- Step 2: Normalize Features ---")
    print(f"Means: {means}")
    print(f"Standard Deviations: {stds}")
    print(f"First 3 rows of Normalized X:\n{X_norm[:3]}\n")

    # Step 3: Compute the covariance matrix using np.einsum
    # Formula: C = (1 / (n - 1)) * X^T @ X
    n = X_norm.shape[0]
    # np.einsum('ji,jk->ik', X_norm, X_norm) is equivalent to X_norm.T @ X_norm
    C = (1 / (n - 1)) * np.einsum('ji,jk->ik', X_norm, X_norm)
    print("--- Step 3: Compute Covariance Matrix ---")
    print(f"Covariance Matrix C:\n{C}\n")
    
    # Verification using np.cov (optional, just to show correctness)
    # C_verify = np.cov(X_norm, rowvar=False)
    # print(f"Covariance Matrix from np.cov (for verification):\n{C_verify}\n")

    # Step 4: Use np.linalg.eig to find eigenvalues/eigenvectors of C
    eigenvalues, eigenvectors = np.linalg.eig(C)
    print("--- Step 4: Compute Eigenvalues & Eigenvectors ---")
    print(f"Eigenvalues:\n{eigenvalues}")
    print(f"Eigenvectors (columns):\n{eigenvectors}\n")

    # Step 5: Get the index of the largest eigenvalue using vectorized np.argmax()
    max_eigenvalue_idx = np.argmax(eigenvalues)
    print("--- Step 5: Find Largest Eigenvalue ---")
    print(f"Index of largest eigenvalue: {max_eigenvalue_idx}")
    print(f"Largest eigenvalue: {eigenvalues[max_eigenvalue_idx]}\n")

    # Step 6: Use this index to select the leading eigenvector (first principal component)
    # Note: Eigenvectors are returned as columns in the matrix
    first_principal_component = eigenvectors[:, max_eigenvalue_idx]
    print("--- Step 6: First Principal Component ---")
    print(f"Leading Eigenvector (First PC): {first_principal_component}")

if __name__ == "__main__":
    main()
