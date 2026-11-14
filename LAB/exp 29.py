import numpy as np

X = np.array([[1, 2]])

W1 = np.array([[0.5, 0.2],
               [0.3, 0.4]])

W2 = np.array([[0.6], [0.7]])

hidden = np.maximum(0, np.dot(X, W1))
output = 1 / (1 + np.exp(-np.dot(hidden, W2)))

print("Hidden Layer Output:", hidden)
print("Output Layer:", output)
