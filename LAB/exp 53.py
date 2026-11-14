import tensorflow as tf
import numpy as np

# Sample text dataset
text = """
deep learning uses neural networks.
neural networks learn from data.
recurrent neural networks process sequences.
deep learning helps computers recognize patterns.
"""

# Tokenize text
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([text])

sequences = tokenizer.texts_to_sequences([text])[0]
vocab_size = len(tokenizer.word_index) + 1

# Create input sequences and next-word targets
X, y = [], []

for i in range(1, len(sequences)):
    X.append(sequences[:i])
    y.append(sequences[i])

# Pad sequences
X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='pre')
y = np.array(y)

# Build RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 32),
    tf.keras.layers.SimpleRNN(64),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

# Compile and train
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=100, verbose=1)

# Predict next word
input_text = "deep learning"
input_sequence = tokenizer.texts_to_sequences([input_text])
input_sequence = tf.keras.preprocessing.sequence.pad_sequences(
    input_sequence, maxlen=X.shape[1], padding='pre'
)

prediction = model.predict(input_sequence, verbose=0)
next_word_id = np.argmax(prediction[0])

for word, index in tokenizer.word_index.items():
    if index == next_word_id:
        print("Input Text:", input_text)
        print("Predicted Next Word:", word)
        break