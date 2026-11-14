import tensorflow as tf
import numpy as np

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize data
X_train = X_train[:5000] / 255.0
y_train = y_train[:5000]

X_test = X_test[:1000] / 255.0
y_test = y_test[:1000]

# Compare models with and without batch normalization
for bn in [False, True]:

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               input_shape=(32, 32, 3))
    ])

    if bn:
        model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=3, verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print("\nBatch Normalization:", "Enabled" if bn else "Disabled")
    print("Test Loss:", round(loss, 4))
    print("Test Accuracy:", round(accuracy * 100, 2), "%")