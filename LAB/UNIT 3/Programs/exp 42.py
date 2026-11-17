import tensorflow as tf
import numpy as np

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Select cat and dog images
train_mask = np.isin(y_train.flatten(), [3, 5])
test_mask = np.isin(y_test.flatten(), [3, 5])

X_train = X_train[train_mask][:2000] / 255.0
X_test = X_test[test_mask][:400] / 255.0

y_train = (y_train[train_mask][:2000] == 5).astype(int)
y_test = (y_test[test_mask][:400] == 5).astype(int)

# Compare different filter configurations
for filters, size in [(16, 3), (32, 3), (32, 5)]:

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, (size, size),
                               activation='relu',
                               input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=3,
              batch_size=32, verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print("\nNumber of Filters:", filters)
    print("Filter Size:", size)
    print("Test Accuracy:", round(accuracy * 100, 2), "%")