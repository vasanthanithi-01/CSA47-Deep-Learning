import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load medical dataset
data = load_breast_cancer()
X = data.data

print("Original Dataset Shape:", X.shape)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Display transformed data
print("Reduced Dataset Shape:", X_pca.shape)
print("\nFirst 5 Transformed Samples:")
print(X_pca[:5])

# Display explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("Total Information Retained:",
      round(sum(pca.explained_variance_ratio_) * 100, 2), "%")