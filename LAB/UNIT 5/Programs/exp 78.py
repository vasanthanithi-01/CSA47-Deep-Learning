import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define mean and covariance matrix
mean = [5, 10]
covariance = [[2, 0.5], [0.5, 3]]

# Generate multivariate Gaussian data
np.random.seed(42)
data = np.random.multivariate_normal(mean, covariance, 1000)

# Estimate statistical parameters
estimated_mean = np.mean(data, axis=0)
estimated_covariance = np.cov(data, rowvar=False)

# Display results
print("Generated Data Shape:", data.shape)
print("\nOriginal Mean:", mean)
print("Estimated Mean:", estimated_mean)

print("\nOriginal Covariance Matrix:")
print(covariance)

print("\nEstimated Covariance Matrix:")
print(estimated_covariance)

# Evaluate generated distribution
distribution = multivariate_normal(
    mean=estimated_mean,
    cov=estimated_covariance
)

probabilities = distribution.pdf(data)

print("\nAverage Probability Density:",
      np.mean(probabilities))

# Visualize the generated data
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Multivariate Gaussian Distribution")
plt.show()