import tensorflow as tf
import numpy as np

# Sample parallel sentences
english = [
    "hello", "how are you", "good morning",
    "thank you", "good night", "i am fine"
]

french = [
    "bonjour", "comment allez vous", "bonjour matin",
    "merci", "bonne nuit", "je vais bien"
]

# Tokenize input and output sentences
input_tokenizer = tf.keras.preprocessing.text.Tokenizer()
output_tokenizer = tf.keras.preprocessing.text.Tokenizer()

input_tokenizer.fit_on_texts(english)
output_tokenizer.fit_on_texts(french)

encoder_input = input_tokenizer.texts_to_sequences(english)
decoder_input = output_tokenizer.texts_to_sequences(french)

# Add start and end tokens
decoder_input = [[1] + seq for seq in decoder_input]
decoder_target = [seq + [2] for seq in output_tokenizer.texts_to_sequences(french)]

# Pad sequences
encoder_input = tf.keras.preprocessing.sequence.pad_sequences(encoder_input)
decoder_input = tf.keras.preprocessing.sequence.pad_sequences(decoder_input)
decoder_target = tf.keras.preprocessing.sequence.pad_sequences(decoder_target)

# Prepare target data
decoder_target = np.expand_dims(decoder_target, -1)

# Define vocabulary sizes
input_vocab = len(input_tokenizer.word_index) + 1
output_vocab = len(output_tokenizer.word_index) + 3

# Encoder
encoder_inputs = tf.keras.Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(input_vocab, 32)(encoder_inputs)
_, state_h, state_c = tf.keras.layers.LSTM(
    64, return_state=True
)(encoder_embedding)

# Decoder
decoder_inputs = tf.keras.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(output_vocab, 32)(decoder_inputs)

decoder_lstm = tf.keras.layers.LSTM(
    64, return_sequences=True
)
decoder_outputs = decoder_lstm(
    decoder_embedding, initial_state=[state_h, state_c]
)

outputs = tf.keras.layers.Dense(output_vocab, activation='softmax')(decoder_outputs)

# Build and train model
model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    [encoder_input, decoder_input],
    decoder_target,
    epochs=100,
    verbose=0
)

print("Sequence-to-Sequence LSTM Model Trained Successfully")
print("Training Accuracy:", round(model.evaluate(
    [encoder_input, decoder_input], decoder_target, verbose=0
)[1] * 100, 2), "%")