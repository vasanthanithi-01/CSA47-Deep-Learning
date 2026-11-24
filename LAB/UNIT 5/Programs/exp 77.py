import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import mean_squared_error

# Load dataset
data = load_wine()
X = data.data

print("Original Dataset Shape:", X.shape)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_pca_reconstructed = pca.inverse_transform(X_pca)

# Apply Factor Analysis
fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X_scaled)
X_fa_reconstructed = X_fa @ fa.components_

# Calculate reconstruction errors
pca_error = mean_squared_error(X_scaled, X_pca_reconstructed)
fa_error = mean_squared_error(X_scaled, X_fa_reconstructed)

# Display results
print("\nPCA Reduced Shape:", X_pca.shape)
print("Factor Analysis Reduced Shape:", X_fa.shape)

print("\nPCA Reconstruction Error:", round(pca_error, 4))
print("Factor Analysis Reconstruction Error:", round(fa_error, 4))

print("\nPCA Explained Variance:",
      round(sum(pca.explained_variance_ratio_) * 100, 2), "%")