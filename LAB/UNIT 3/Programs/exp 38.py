import tensorflow as tf
import numpy as np

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Select cat and dog images (Cat = 3, Dog = 5)
train_mask = np.isin(y_train.flatten(), [3, 5])
test_mask = np.isin(y_test.flatten(), [3, 5])

X_train = X_train[train_mask] / 255.0
X_test = X_test[test_mask] / 255.0

y_train = (y_train[train_mask] == 5).astype(int)
y_test = (y_test[test_mask] == 5).astype(int)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(accuracy * 100, 2), "%")