import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate classification dataset
X, y = make_classification(n_samples=500, n_features=10,
                           n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Compare optimizers
for opt in ['SGD', 'RMSprop', 'Adam']:

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10,
                        batch_size=32, verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print("\nOptimizer:", opt)
    print("Training Loss:", round(history.history['loss'][-1], 4))
    print("Test Accuracy:", round(accuracy * 100, 2), "%")