import numpy as np
import tensorflow as tf

# Generate sample sequential data
np.random.seed(42)
data = np.sin(np.arange(500) * 0.05) + np.random.randn(500) * 0.05

# Prepare input sequences
X, y = [], []

for i in range(10, len(data)):
    X.append(data[i-10:i])
    y.append(data[i])

X = np.array(X).reshape(-1, 10, 1)
y = np.array(y)

# Split dataset
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Function to create and train a model
def train_model(layer):
    model = tf.keras.Sequential([
        layer(32, input_shape=(10, 1)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    return model.evaluate(X_test, y_test, verbose=0)

# Compare models
rnn_loss = train_model(tf.keras.layers.SimpleRNN)
lstm_loss = train_model(tf.keras.layers.LSTM)
gru_loss = train_model(tf.keras.layers.GRU)

print("RNN Test Loss:", round(rnn_loss, 4))
print("LSTM Test Loss:", round(lstm_loss, 4))
print("GRU Test Loss:", round(gru_loss, 4))

# Identify best model
results = {
    "RNN": rnn_loss,
    "LSTM": lstm_loss,
    "GRU": gru_loss
}

print("Best Model:", min(results, key=results.get))