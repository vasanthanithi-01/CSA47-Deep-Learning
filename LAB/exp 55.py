import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Generate sample stock price data
np.random.seed(42)
data = 100 + np.cumsum(np.random.randn(500) * 0.5)

# Prepare input sequences
X, y = [], []

for i in range(10, len(data)):
    X.append(data[i-10:i])
    y.append(data[i])

X = np.array(X).reshape(-1, 10, 1)
y = np.array(y)

# Normalize data
minimum = X.min()
maximum = X.max()

X = (X - minimum) / (maximum - minimum)
y = (y - minimum) / (maximum - minimum)

# Split dataset
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(10, 1)),
    tf.keras.layers.Dense(1)
])

# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict stock prices
predictions = model.predict(X_test)

# Convert values back to original scale
actual = y_test * (maximum - minimum) + minimum
predicted = predictions.flatten() * (maximum - minimum) + minimum

# Evaluate model
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)

print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))