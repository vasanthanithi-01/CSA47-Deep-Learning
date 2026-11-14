import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample dataset
X, y = make_classification(n_samples=500, n_features=10,
                           n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Compare models
for reg in [False, True]:

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            32, activation='relu', input_shape=(10,),
            kernel_regularizer=tf.keras.regularizers.l2(0.01) if reg else None
        ),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print("\nRegularization:", "L2" if reg else "None")
    print("Test Loss:", round(loss, 4))
    print("Test Accuracy:", round(accuracy * 100, 2), "%")