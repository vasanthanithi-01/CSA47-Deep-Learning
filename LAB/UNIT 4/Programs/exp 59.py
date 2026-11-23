import tensorflow as tf
import numpy as np

# Generate sequential data
X = np.random.rand(100, 50, 1).astype('float32')
y = np.random.rand(100, 1).astype('float32')

# Build a deep SimpleRNN model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, activation='sigmoid',
                              return_sequences=True,
                              input_shape=(50, 1)),
    tf.keras.layers.SimpleRNN(32, activation='sigmoid'),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Calculate gradients
with tf.GradientTape() as tape:
    predictions = model(X)
    loss = tf.reduce_mean(tf.square(y - predictions))

gradients = tape.gradient(loss, model.trainable_variables)

# Display gradient values
print("Gradient Values:")

for i, gradient in enumerate(gradients):
    if gradient is not None:
        print("Layer", i + 1, ":", tf.reduce_mean(tf.abs(gradient)).numpy())

# Train model
model.fit(X, y, epochs=5, batch_size=16)

print("Vanishing Gradient Demonstration Completed")