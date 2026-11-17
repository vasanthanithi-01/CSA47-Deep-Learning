import tensorflow as tf

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Resize images and normalize
X_train = tf.image.resize(X_train[:2000], (224, 224)) / 255.0
X_test = tf.image.resize(X_test[:400], (224, 224)) / 255.0

y_train = y_train[:2000]
y_test = y_test[:400]

# Load pretrained ResNet50 model
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# Build transfer learning model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=3, batch_size=32)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(accuracy * 100, 2), "%")