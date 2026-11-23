import numpy as np
import tensorflow as tf

# Generate sample sales data
np.random.seed(42)
data = 100 + 20 * np.sin(np.arange(500) * 0.05) + np.random.randn(500) * 3

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

# Experiment with different hyperparameters
for units, rate, batch in [(16, 0.001, 32), (32, 0.001, 32), (32, 0.01, 64)]:

    # Build LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units, input_shape=(10, 1)),
        tf.keras.layers.Dense(1)
    ])

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=rate),
        loss='mse'
    )

    # Train model
    model.fit(X_train, y_train, epochs=5, batch_size=batch, verbose=0)

    # Evaluate model
    loss = model.evaluate(X_test, y_test, verbose=0)

    print("Hidden Units:", units)
    print("Learning Rate:", rate)
    print("Batch Size:", batch)
    print("Test Loss:", round(loss, 4))
    print()