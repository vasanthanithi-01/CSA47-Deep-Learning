import tensorflow as tf
import numpy as np

# Sample sentences and sentiment labels
sentences = [
    "this movie is good",
    "this movie is excellent",
    "this movie is bad",
    "this movie is terrible",
    "the story is amazing",
    "the story is boring"
]

labels = np.array([1, 1, 0, 0, 1, 0])

# Tokenize sentences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
X = tf.keras.preprocessing.sequence.pad_sequences(sequences)

# Build neural network for sentiment classification
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        len(tokenizer.word_index) + 1, 16
    ),
    tf.keras.layers.SimpleRNN(16),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X, labels, epochs=50, verbose=0)

# Predict sentiment
predictions = model.predict(X, verbose=0)

for i, prediction in enumerate(predictions):
    sentiment = "Positive" if prediction[0] >= 0.5 else "Negative"
    print(sentences[i], ":", sentiment)