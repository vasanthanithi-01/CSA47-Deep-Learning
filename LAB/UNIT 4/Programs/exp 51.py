import numpy as np
import tensorflow as tf

# Generate sample stock price data
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

# Build RNN model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, activation='tanh', input_shape=(10, 1)),
    tf.keras.layers.Dense(1)
])

# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate model
loss = model.evaluate(X_test, y_test)

# Predict stock price
prediction = model.predict(X_test[-1].reshape(1, 10, 1))
predicted_price = prediction[0][0] * (maximum - minimum) + minimum
actual_price = y_test[-1] * (maximum - minimum) + minimum

print("Test Loss:", round(loss, 4))
print("Actual Stock Price:", round(actual_price, 2))
print("Predicted Stock Price:", round(predicted_price, 2))