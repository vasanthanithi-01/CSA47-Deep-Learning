import numpy as np
from sklearn.linear_model import Lasso, Ridge

X = np.array([[1, 2], [2, 3], [3, 4],
              [4, 5], [5, 6]])

y = np.array([3, 5, 7, 9, 11])

l1 = Lasso(alpha=0.1)
l2 = Ridge(alpha=0.1)

l1.fit(X, y)
l2.fit(X, y)

print("L1 Coefficients:", l1.coef_)
print("L2 Coefficients:", l2.coef_)
