from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
data = load_wine()
X = data.data

print("Original Dataset Shape:", X.shape)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA while retaining 95% information
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Display results
print("Reduced Dataset Shape:", X_pca.shape)
print("Original Number of Features:", X.shape[1])
print("Required Principal Components:", pca.n_components_)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("Total Explained Variance:",
      round(sum(pca.explained_variance_ratio_) * 100, 2), "%")