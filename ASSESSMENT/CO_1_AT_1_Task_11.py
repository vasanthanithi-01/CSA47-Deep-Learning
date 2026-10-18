
# CO_1_AT_1_Task_11.py
# Task 11: Justification for Deep Learning
# Compare Logistic Regression, Random Forest, and a NumPy Neural Network

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Dataset configuration
DATA_DIR = Path(r"C:\Users\VASANTH\Downloads\raw\color")

CLASS_NAMES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

IMAGE_SIZE = (64, 64)
MAX_IMAGES_PER_CLASS = 300
EPOCHS = 30
LEARNING_RATE = 0.01
HIDDEN_UNITS = 64

# Load images
X, y = [], []

for label, class_name in enumerate(CLASS_NAMES):
    folder = DATA_DIR / class_name

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])[:MAX_IMAGES_PER_CLASS]

    print(f"Loading {class_name}: {len(files)} images")

    for file in files:
        try:
            image = Image.open(file).convert("L")
            image = image.resize(IMAGE_SIZE)
            X.append(np.asarray(image, dtype=np.float32).flatten() / 255.0)
            y.append(label)
        except Exception as error:
            print(f"Skipping {file.name}: {error}")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

if len(np.unique(y)) != len(CLASS_NAMES):
    raise ValueError("All three classes must contain usable images.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Standardize features for traditional models and neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

num_classes = len(CLASS_NAMES)

# 1. Logistic Regression
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# 2. Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 3. Simple neural network using NumPy
print("Training NumPy Neural Network...")

rng = np.random.default_rng(42)
input_units = X_train_scaled.shape[1]

W1 = rng.normal(
    0, np.sqrt(2 / input_units),
    (input_units, HIDDEN_UNITS)
)
b1 = np.zeros((1, HIDDEN_UNITS))

W2 = rng.normal(
    0, np.sqrt(2 / HIDDEN_UNITS),
    (HIDDEN_UNITS, num_classes)
)
b2 = np.zeros((1, num_classes))

y_onehot = np.eye(num_classes)[y_train]
loss_history = []

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

for epoch in range(EPOCHS):
    # Forward propagation
    z1 = X_train_scaled @ W1 + b1
    a1 = np.maximum(0, z1)

    z2 = a1 @ W2 + b2
    probs = softmax(z2)

    loss = -np.mean(
        np.sum(y_onehot * np.log(probs + 1e-12), axis=1)
    )
    loss_history.append(loss)

    # Backpropagation
    dz2 = (probs - y_onehot) / len(X_train_scaled)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)

    dW1 = X_train_scaled.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Gradient descent update
    W2 -= LEARNING_RATE * dW2
    b2 -= LEARNING_RATE * db2
    W1 -= LEARNING_RATE * dW1
    b1 -= LEARNING_RATE * db1

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss:.4f}")

# Neural network predictions
hidden = np.maximum(0, X_test_scaled @ W1 + b1)
nn_probs = softmax(hidden @ W2 + b2)
nn_pred = np.argmax(nn_probs, axis=1)

# Evaluate all models
models = {
    "Logistic Regression": lr_pred,
    "Random Forest": rf_pred,
    "NumPy Neural Network": nn_pred
}

results = []

print("\n========== MODEL COMPARISON ==========")

for name, predictions in models.items():
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1-score": f1
    })

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {f1:.4f}")

results_df = pd.DataFrame(results)

# Save results
output_dir = Path("task11_outputs")
output_dir.mkdir(exist_ok=True)

results_df.to_csv(output_dir / "model_comparison.csv", index=False)

# Plot model accuracy
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Traditional ML vs Neural Network")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(output_dir / "model_accuracy.png", dpi=300)
plt.show()

# Plot neural network loss
plt.figure(figsize=(7, 5))
plt.plot(range(1, EPOCHS + 1), loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Cross-Entropy Loss")
plt.title("Neural Network Training Loss")
plt.tight_layout()
plt.savefig(output_dir / "neural_network_loss.png", dpi=300)
plt.show()

print("\nResults saved in:", output_dir.resolve())
