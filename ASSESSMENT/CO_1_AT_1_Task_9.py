
# CO_1_AT_1_Task_9.py
# Task 9: Experimental Performance Evaluation
# Dataset: PlantVillage - 3 tomato classes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Dataset path
DATA_DIR = Path(r"C:\Users\VASANTH\Downloads\raw\color")

CLASS_NAMES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

IMAGE_SIZE = (64, 64)
MAX_IMAGES_PER_CLASS = 500

# Load and preprocess images
X = []
y = []

for label, class_name in enumerate(CLASS_NAMES):
    folder = DATA_DIR / class_name

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    image_files = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    image_files = image_files[:MAX_IMAGES_PER_CLASS]

    print(f"Loading {class_name}: {len(image_files)} images")

    for image_path in image_files:
        try:
            image = Image.open(image_path).convert("L")
            image = image.resize(IMAGE_SIZE)

            # Flatten 64 x 64 image into 4096 features
            X.append(np.asarray(image, dtype=np.float32).flatten())
            y.append(label)

        except Exception as error:
            print(f"Skipping {image_path.name}: {error}")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

if len(np.unique(y)) < len(CLASS_NAMES):
    raise ValueError("One or more classes have no usable images.")

print("\nDataset shape:", X.shape)
print("Labels shape:", y.shape)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Standardize using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train classifier
model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42
)

print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, average="macro", zero_division=0
)
recall = recall_score(
    y_test, y_pred, average="macro", zero_division=0
)
f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

print("\n========== PERFORMANCE RESULTS ==========")
print(f"Accuracy          : {accuracy:.4f}")
print(f"Macro Precision   : {precision:.4f}")
print(f"Macro Recall      : {recall:.4f}")
print(f"Macro F1-score    : {f1:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
    target_names=CLASS_NAMES,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(CLASS_NAMES))
)

print("\nConfusion Matrix:")
print(cm)

# Create output directory
OUTPUT_DIR = Path("task9_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Save metrics
metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1-score"
    ],
    "Score": [accuracy, precision, recall, f1]
})

metrics.to_csv(OUTPUT_DIR / "performance_metrics.csv", index=False)

# Save confusion matrix
cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)
cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

# Plot confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Healthy",
        "Early blight",
        "Late blight"
    ]
)

display.plot(cmap="Blues", values_format="d")
plt.title("Tomato Disease Classification - Confusion Matrix")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300
)
plt.show()

# Plot performance metrics
plt.figure(figsize=(8, 5))
plt.bar(
    metrics["Metric"],
    metrics["Score"]
)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Evaluation")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "performance_metrics.png",
    dpi=300
)
plt.show()

print("\nResults saved in:", OUTPUT_DIR.resolve())
