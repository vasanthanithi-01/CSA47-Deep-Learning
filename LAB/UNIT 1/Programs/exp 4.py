import numpy as np

A = np.array([[0.5, 0.2],
              [0.5, 0.8]])

values, vectors = np.linalg.eig(A)

print("Eigenvalues:")
print(values)

print("Eigenvectors:")
print(vectors)
