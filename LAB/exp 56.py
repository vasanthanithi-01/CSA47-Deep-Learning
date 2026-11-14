import numpy as np
import tensorflow as tf

# Generate sample weather data
np.random.seed(42)
days = np.arange(500)

temperature = 25 + 5 * np.sin(days * 0.05) + np.random.randn(500)
humidity = 60 + 10 * np.cos(days * 0.05) + np.random.randn(500)
pressure = 1010 + 5 * np.sin(days * 0.03) + np.random.randn(500)

data = np.column_stack((temperature, humidity, pressure))

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

# Build GRU model
model = tf.keras.Sequential([
    tf.keras.layers.GRU(32, input_shape=(10, 3)),
    tf.keras.layers.Dense(3)
])

# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate model
loss = model.evaluate(X_test, y_test)

# Predict weather conditions
prediction = model.predict(X_test[-1].reshape(1, 10, 3))

print("Test Loss:", round(loss, 4))
print("Predicted Temperature:", round(prediction[0][0] * (maximum[0] - minimum[0]) + minimum[0], 2))
print("Predicted Humidity:", round(prediction[0][1] * (maximum[1] - minimum[1]) + minimum[1], 2))
print("Predicted Pressure:", round(prediction[0][2] * (maximum[2] - minimum[2]) + minimum[2], 2))