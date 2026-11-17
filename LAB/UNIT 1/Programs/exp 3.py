import numpy as np

A = np.array([[2, 3], [1, 4]])

print("Determinant:", np.linalg.det(A))
print("Inverse:")
print(np.linalg.inv(A))
print("Rank:", np.linalg.matrix_rank(A))

if np.linalg.det(A) != 0:
    print("Matrix is suitable for encryption")
else:
    print("Matrix is not suitable for encryption")
