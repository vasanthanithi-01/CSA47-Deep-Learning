
# CO_1_AT_1_Task_10.py
# Task 10: Theoretical vs Experimental Comparison

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# Dataset configuration
DATA_DIR = Path(r"C:\Users\VASANTH\Downloads\raw\color")

CLASS_NAMES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

IMAGE_SIZE = (64, 64)
MAX_IMAGES_PER_CLASS = 500

# Load dataset
X = []
y = []

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

            X.append(np.asarray(image, dtype=np.float32).flatten())
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

# Normalize features using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)

print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

# Predictions and probabilities
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_prob = model.predict_proba(X_train)
test_prob = model.predict_proba(X_test)

# Experimental results
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

train_loss = log_loss(
    y_train, train_prob, labels=np.arange(len(CLASS_NAMES))
)
test_loss = log_loss(
    y_test, test_prob, labels=np.arange(len(CLASS_NAMES))
)

# Theoretical properties of Logistic Regression
num_features = X_train.shape[1]
num_classes = len(CLASS_NAMES)

# Multiclass linear model parameter count, including biases
parameter_count = num_features * num_classes + num_classes

print("\n========== THEORETICAL VS EXPERIMENTAL ==========")

print("\nTheoretical properties:")
print("Model: Multiclass Logistic Regression")
print("Input features:", num_features)
print("Number of classes:", num_classes)
print("Trainable parameters:", parameter_count)
print("Output: Class probabilities")
print("Decision boundary: Linear in feature space")
print("Expected behaviour: Training performance may exceed test performance.")

print("\nExperimental results:")
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Testing accuracy : {test_accuracy:.4f}")
print(f"Training loss    : {train_loss:.4f}")
print(f"Testing loss     : {test_loss:.4f}")

print("\nGeneralization gap:")
print(f"Accuracy gap: {train_accuracy - test_accuracy:.4f}")
print(f"Loss gap    : {test_loss - train_loss:.4f}")

if train_accuracy > test_accuracy:
    print("Observation: Training accuracy exceeds testing accuracy.")
else:
    print("Observation: Testing accuracy is equal to or greater than training accuracy.")

# Save results
output_dir = Path("task10_outputs")
output_dir.mkdir(exist_ok=True)

results = pd.DataFrame({
    "Measure": [
        "Training Accuracy",
        "Testing Accuracy",
        "Training Log Loss",
        "Testing Log Loss",
        "Accuracy Generalization Gap",
        "Loss Generalization Gap",
        "Trainable Parameters"
    ],
    "Value": [
        train_accuracy,
        test_accuracy,
        train_loss,
        test_loss,
        train_accuracy - test_accuracy,
        test_loss - train_loss,
        parameter_count
    ]
})

results.to_csv(output_dir / "comparison_results.csv", index=False)

# Plot accuracy comparison
plt.figure(figsize=(7, 5))
plt.bar(
    ["Training Accuracy", "Testing Accuracy"],
    [train_accuracy, test_accuracy]
)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Theoretical Expectations vs Experimental Accuracy")
plt.tight_layout()
plt.savefig(output_dir / "accuracy_comparison.png", dpi=300)
plt.show()

# Plot loss comparison
plt.figure(figsize=(7, 5))
plt.bar(
    ["Training Loss", "Testing Loss"],
    [train_loss, test_loss]
)
plt.ylabel("Cross-Entropy Loss")
plt.title("Training vs Testing Loss")
plt.tight_layout()
plt.savefig(output_dir / "loss_comparison.png", dpi=300)
plt.show()

print("\nResults saved in:", output_dir.resolve())
