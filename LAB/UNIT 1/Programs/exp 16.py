import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [1000, 2],
    [1500, 3],
    [2000, 3],
    [2500, 4],
    [3000, 4]
])

Y = np.array([20, 30, 40, 50, 60])

model = LinearRegression()
model.fit(X, Y)

prediction = model.predict([[1800, 3]])

print("Predicted House Price:", prediction[0])
print("Model Score:", model.score(X, Y))
