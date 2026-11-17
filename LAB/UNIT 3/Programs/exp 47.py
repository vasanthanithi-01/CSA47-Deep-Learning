import numpy as np
import tensorflow as tf

# Generate sequential sensor data
data = np.sin(np.arange(1000) * 0.02)

# Prepare input sequences and target values
X, y = [], []

for i in range(10, len(data)):
    X.append(data[i-10:i])
    y.append(data[i])

X = np.array(X).reshape(-1, 10, 1)
y = np.array(y)

# Build RNN model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, activation='tanh', input_shape=(10, 1)),
    tf.keras.layers.Dense(1)
])

# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# Predict the next sensor value
prediction = model.predict(X[-1].reshape(1, 10, 1))

print("Actual Value:", round(y[-1], 4))
print("Predicted Value:", round(float(prediction[0][0]), 4))