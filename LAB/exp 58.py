import numpy as np
import tensorflow as tf

# Generate sample machine sensor data
np.random.seed(42)
data = np.column_stack([
    50 + np.random.randn(500) * 5,   # Temperature
    100 + np.random.randn(500) * 10, # Pressure
    20 + np.random.randn(500) * 3    # Vibration
])

# Normalize data
minimum = data.min(axis=0)
maximum = data.max(axis=0)
data = (data - minimum) / (maximum - minimum)

# Prepare input sequences
X, y = [], []

for i in range(10, len(data)):
    X.append(data[i-10:i])
    y.append(data[i])

X = np.array(X)
y = np.array(y)

# Split dataset
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build stacked RNN model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, return_sequences=True,
                              input_shape=(10, 3)),
    tf.keras.layers.SimpleRNN(16),
    tf.keras.layers.Dense(3)
])

# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate model
loss = model.evaluate(X_test, y_test)

# Predict machine conditions
prediction = model.predict(X_test[-1].reshape(1, 10, 3))

print("Test Loss:", round(loss, 4))
print("Predicted Temperature:", round(prediction[0][0] * (maximum[0] - minimum[0]) + minimum[0], 2))
print("Predicted Pressure:", round(prediction[0][1] * (maximum[1] - minimum[1]) + minimum[1], 2))
print("Predicted Vibration:", round(prediction[0][2] * (maximum[2] - minimum[2]) + minimum[2], 2))