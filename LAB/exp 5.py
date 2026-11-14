import numpy as np

A = np.array([2, 3, 4])
B = np.array([1, 2, 3])

dot = np.dot(A, B)
cosine = dot / (np.linalg.norm(A) * np.linalg.norm(B))

print("Dot Product:", dot)
print("Cosine Similarity:", round(cosine, 4))
