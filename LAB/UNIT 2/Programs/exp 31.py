import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

X = np.array([[1000], [1500], [2000],
              [2500], [3000]])

y = np.array([20, 30, 40, 50, 60])

model = MLPRegressor(
    hidden_layer_sizes=(10,),
    max_iter=5000,
    random_state=42
)

model.fit(X, y)

pred = model.predict(X)

print("Actual Prices:", y)
print("Predicted Prices:", np.round(pred, 2))
print("MSE:", round(mean_squared_error(y, pred), 4))
