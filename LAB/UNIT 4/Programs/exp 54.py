import numpy as np
import tensorflow as tf

# Generate sample sales data
data = 100 + 20 * np.sin(np.arange(500) * 0.05) + np.random.randn(500) * 3

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

# Create RNN model
rnn_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=(10, 1)),
    tf.keras.layers.Dense(1)
])

# Create LSTM model
lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(10, 1)),
    tf.keras.layers.Dense(1)
])

# Compile models
rnn_model.compile(optimizer='adam', loss='mse')
lstm_model.compile(optimizer='adam', loss='mse')

# Train models
print("Training RNN Model")
rnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

print("Training LSTM Model")
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Evaluate models
rnn_loss = rnn_model.evaluate(X_test, y_test, verbose=0)
lstm_loss = lstm_model.evaluate(X_test, y_test, verbose=0)

print("RNN Test Loss:", round(rnn_loss, 4))
print("LSTM Test Loss:", round(lstm_loss, 4))

if rnn_loss < lstm_loss:
    print("RNN performs better")
else:
    print("LSTM performs better")