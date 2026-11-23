import numpy as np
import tensorflow as tf

# Generate sample temperature data
np.random.seed(42)
data = 25 + 5 * np.sin(np.arange(500) * 0.05) + np.random.randn(500)

# Function to prepare sequences
def prepare_data(length):
    X, y = [], []

    for i in range(length, len(data)):
        X.append(data[i-length:i])
        y.append(data[i])

    X = np.array(X).reshape(-1, length, 1)
    y = np.array(y)

    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:]

# Compare different sequence lengths
for length in [5, 10, 20]:
    X_train, X_test, y_train, y_test = prepare_data(length)

    # Build RNN model
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(32, input_shape=(length, 1)),
        tf.keras.layers.Dense(1)
    ])

    # Compile and train
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Evaluate model
    loss = model.evaluate(X_test, y_test, verbose=0)

    print("Sequence Length:", length)
    print("Test Loss:", round(loss, 4))
    print()