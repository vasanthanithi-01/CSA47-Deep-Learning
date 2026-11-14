import numpy as np
import tensorflow as tf

# Generate sequential data
np.random.seed(42)
data = np.sin(np.arange(600) * 0.05)

# Prepare long input sequences
X, y = [], []

for i in range(50, len(data)):
    X.append(data[i-50:i])
    y.append(data[i])

X = np.array(X).reshape(-1, 50, 1)
y = np.array(y)

# Split dataset
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Function to train and evaluate models
def train_model(layer):
    model = tf.keras.Sequential([
        layer(32, input_shape=(50, 1)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    return model.evaluate(X_test, y_test, verbose=0)

# Compare RNN and LSTM
rnn_loss = train_model(tf.keras.layers.SimpleRNN)
lstm_loss = train_model(tf.keras.layers.LSTM)

print("RNN Test Loss:", round(rnn_loss, 4))
print("LSTM Test Loss:", round(lstm_loss, 4))

if rnn_loss < lstm_loss:
    print("RNN performs better")
else:
    print("LSTM performs better")