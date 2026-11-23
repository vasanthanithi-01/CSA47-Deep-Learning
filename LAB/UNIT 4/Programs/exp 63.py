import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score

# Generate sample customer message data
np.random.seed(42)
X = np.random.randint(1, 100, (1000, 20))
y = np.random.randint(0, 3, 1000)

# Split dataset
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Build RNN classification model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(100, 32),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile and train
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=5, batch_size=32)

# Predict complaint categories
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

# Calculate evaluation metrics
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))