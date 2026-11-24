import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load healthcare dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("Dataset Shape:", X.shape)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build deep learning model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
print("Test Loss:", round(loss, 4))

# Generate predictions
predictions = model.predict(X_test, verbose=0)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Display performance metrics
print("\nClassification Report:")
print(classification_report(y_test, predicted_classes,
                            target_names=data.target_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted_classes))

# Predict for a new patient
new_patient = X_test[0].reshape(1, -1)
prediction = model.predict(new_patient, verbose=0)[0][0]

print("\nHealthcare Prediction:")
if prediction > 0.5:
    print("Benign")
else:
    print("Malignant")