import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)

X = np.random.rand(100, 32, 32, 1)
y = np.random.randint(0, 2, 100)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(32, 32, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=2)

print("Model training successful!")