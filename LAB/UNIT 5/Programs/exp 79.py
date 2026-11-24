import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Generate customer purchasing data
X, _ = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    random_state=42
)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Gaussian Mixture Model
gmm = GaussianMixture(
    n_components=3,
    random_state=42
)
gmm.fit(X_scaled)

# Predict cluster labels and probabilities
labels = gmm.predict(X_scaled)
probabilities = gmm.predict_proba(X_scaled)

# Display results
print("Number of Clusters:", gmm.n_components)
print("Cluster Labels:")
print(labels[:10])

print("\nCluster Membership Probabilities:")
print(np.round(probabilities[:10], 3))

print("\nNumber of Customers in Each Cluster:")
for i in range(3):
    print("Cluster", i, ":", np.sum(labels == i))

# Visualize clusters
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap="viridis"
)

plt.xlabel("Purchasing Feature 1")
plt.ylabel("Purchasing Feature 2")
plt.title("Customer Segmentation Using GMM")
plt.show()