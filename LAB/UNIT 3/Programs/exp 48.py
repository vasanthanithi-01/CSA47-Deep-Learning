import numpy as np
import tensorflow as tf

# Generate daily temperature data
data = 25 + 5 * np.sin(np.arange(500) * 0.05)

# Prepare sequences
X, y = [], []

for i in range(10, len(data)):
    X.append(data[i-10:i])
    y.append(data[i])

X = np.array(X).reshape(-1, 10, 1)
y = np.array(y)

# Split data into training and testing sets
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

# Evaluate and predict
loss = model.evaluate(X_test, y_test)
prediction = model.predict(X_test[-1].reshape(1, 10, 1))

print("Test Loss:", round(loss, 4))
print("Actual Temperature:", round(y_test[-1], 2))
print("Predicted Temperature:", round(float(prediction[0][0]), 2))