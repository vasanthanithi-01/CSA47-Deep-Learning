import numpy as np

X = np.array([[1, 0]])
y = np.array([[1]])

W = np.array([[0.5], [0.5]])

lr = 0.1

# Forward propagation
z = np.dot(X, W)
output = 1 / (1 + np.exp(-z))

# Backpropagation
error = output - y
gradient = np.dot(X.T, error)

# Weight update
W = W - lr * gradient

print("Predicted Output:", output)
print("Updated Weights:")
print(W)
