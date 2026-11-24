import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA

# Load dataset
data = load_wine()
X = data.data

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply ICA
ica = FastICA(n_components=2, random_state=42, max_iter=1000)
X_ica = ica.fit_transform(X_scaled)

# Display explained variance
print("PCA Explained Variance:",
      round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

print("PCA Components Shape:", X_pca.shape)
print("ICA Components Shape:", X_ica.shape)

# Visualize PCA and ICA results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA Feature Extraction")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.subplot(1, 2, 2)
plt.scatter(X_ica[:, 0], X_ica[:, 1])
plt.title("ICA Feature Extraction")
plt.xlabel("Independent Component 1")
plt.ylabel("Independent Component 2")

plt.tight_layout()
plt.show()