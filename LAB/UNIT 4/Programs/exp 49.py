import tensorflow as tf

# Load IMDB movie review dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=10000
)

# Pad sequences to equal length
X_train = tf.keras.preprocessing.sequence.pad_sequences(
    X_train, maxlen=200
)
X_test = tf.keras.preprocessing.sequence.pad_sequences(
    X_test, maxlen=200
)

# Build RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 32),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    X_train[:5000], y_train[:5000],
    epochs=3,
    batch_size=64,
    validation_split=0.2
)

# Evaluate model
loss, accuracy = model.evaluate(X_test[:2000], y_test[:2000])

print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(accuracy * 100, 2), "%")

# Predict sentiment
prediction = model.predict(X_test[:1])

if prediction[0][0] >= 0.5:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")