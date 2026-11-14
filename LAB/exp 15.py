import numpy as np
X = np.array([1, 2, 3, 4])
Y = np.array([2, 4, 6, 8])
w = 0
for i in range(100):
    grad = np.mean(2 * X * (w * X - Y))
    w -= 0.01 * grad

print("Batch GD:", round(w, 4))
w = 0
for i in range(100):
    for x, y in zip(X, Y):
        grad = 2 * x * (w * x - y)
        w -= 0.01 * grad

print("Stochastic GD:", round(w, 4))
w = 0
for i in range(100):
    for j in range(0, 4, 2):
        xb = X[j:j+2]
        yb = Y[j:j+2]
        grad = np.mean(2 * xb * (w * xb - yb))
        w -= 0.01 * grad

print("Mini-Batch GD:", round(w, 4))
