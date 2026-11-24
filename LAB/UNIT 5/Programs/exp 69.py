import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Generate high-dimensional dataset
X, y = make_blobs(
    n_samples=300,
    centers=3,
    n_features=8,
    random_state=42
)

print("Original Dataset Shape:", X.shape)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to two components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Reduced Dataset Shape:", X_pca.shape)
print("Explained Variance:",
      round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

# Visualize the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis",
    edgecolors="black"
)

plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("PCA Visualization of High-Dimensional Data")
plt.colorbar(label="Cluster")
plt.show()