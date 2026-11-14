import tensorflow as tf
import numpy as np

# Sample training text
text = """
deep learning is a branch of machine learning.
neural networks learn patterns from data.
recurrent neural networks process sequential data.
"""

# Create character vocabulary
chars = sorted(set(text))
char_to_num = {c: i for i, c in enumerate(chars)}
num_to_char = np.array(chars)

# Convert text into numbers
data = np.array([char_to_num[c] for c in text])

# Prepare input and target sequences
seq_length = 20
X, y = [], []

for i in range(len(data) - seq_length):
    X.append(data[i:i + seq_length])
    y.append(data[i + seq_length])

X = np.array(X)
y = np.array(y)

# Build character-level RNN
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(chars), 32),
    tf.keras.layers.SimpleRNN(64),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

# Compile and train
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=50, batch_size=16, verbose=1)

# Generate new text
input_seq = X[0].reshape(1, seq_length)
generated = ""

for _ in range(100):
    prediction = model.predict(input_seq, verbose=0)
    next_char = np.argmax(prediction[0])
    generated += num_to_char[next_char]

    input_seq = np.append(input_seq[:, 1:], [[next_char]], axis=1)

print("Generated Text:")
print(generated)