
# ============================================================
# CSA47 - DEEP LEARNING
# ASSESSMENT 1 - TASK 8 (4 MARKS)
# Gradient-Based Optimisation
# Dataset: PlantVillage - Tomato Healthy, Early Blight,
#          and Late Blight
# ============================================================

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\VASANTH\Downloads\raw\color"
)

OUTPUT_DIR = Path("task8_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)
MAX_IMAGES_PER_CLASS = 500
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATES = [0.001, 0.01, 0.1]
RANDOM_STATE = 42

CLASS_NAMES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ------------------------------------------------------------
# 2. LOAD AND PREPROCESS IMAGES
# ------------------------------------------------------------

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR}"
    )

features = []
labels = []

for class_name in CLASS_NAMES:

    class_dir = DATA_DIR / class_name

    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Class folder not found: {class_dir}"
        )

    image_paths = sorted(
        p for p in class_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in VALID_EXTENSIONS
    )

    image_paths = image_paths[:MAX_IMAGES_PER_CLASS]

    print(f"{class_name}: using {len(image_paths)} images")

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                image = image.convert("L")
                image = image.resize(
                    IMAGE_SIZE,
                    Image.Resampling.LANCZOS
                )

                pixels = (
                    np.asarray(image, dtype=np.float32) / 255.0
                )

            features.append(pixels.flatten())
            labels.append(class_name)

        except Exception as error:
            print("Skipped:", image_path, error)

X = np.asarray(features, dtype=np.float64)
y_text = np.asarray(labels)

if len(X) < 30:
    raise ValueError("Not enough images were loaded.")

class_to_index = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

y = np.array([class_to_index[label] for label in y_text])

print("\n========== DATASET ==========")
print("Feature matrix:", X.shape)
print("Number of classes:", len(CLASS_NAMES))


# ------------------------------------------------------------
# 3. TRAINING AND VALIDATION SPLIT
# ------------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))


# ------------------------------------------------------------
# 4. FEATURE SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Add a bias feature of 1 to every sample.
X_train = np.column_stack([
    X_train,
    np.ones(len(X_train))
])

X_val = np.column_stack([
    X_val,
    np.ones(len(X_val))
])

number_of_features = X_train.shape[1]
number_of_classes = len(CLASS_NAMES)


# ------------------------------------------------------------
# 5. SOFTMAX AND CROSS-ENTROPY FUNCTIONS
# ------------------------------------------------------------

def softmax(logits):
    # Subtracting the maximum improves numerical stability.
    shifted = logits - np.max(
        logits,
        axis=1,
        keepdims=True
    )

    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )


def cross_entropy_loss(X, y, weights):
    logits = X @ weights
    probabilities = softmax(logits)

    epsilon = 1e-12

    correct_probabilities = probabilities[
        np.arange(len(y)), y
    ]

    loss = -np.mean(
        np.log(np.clip(correct_probabilities, epsilon, 1.0))
    )

    return loss


# ------------------------------------------------------------
# 6. TRAIN SOFTMAX REGRESSION USING MINI-BATCH GD
# ------------------------------------------------------------

all_results = []
loss_histories = {}

for learning_rate in LEARNING_RATES:

    print(
        f"\nTraining with learning rate = {learning_rate}"
    )

    rng = np.random.default_rng(RANDOM_STATE)

    # Initialise weights with small random values.
    weights = rng.normal(
        loc=0.0,
        scale=0.01,
        size=(number_of_features, number_of_classes)
    )

    train_losses = []
    val_losses = []
    epoch_times = []

    for epoch in range(EPOCHS):

        epoch_start = time.perf_counter()

        # Shuffle the training samples each epoch.
        indices = rng.permutation(len(X_train))

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Process mini-batches.
        for start in range(0, len(X_shuffled), BATCH_SIZE):

            end = start + BATCH_SIZE

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward pass
            logits = X_batch @ weights
            probabilities = softmax(logits)

            # One-hot encode the true labels.
            y_one_hot = np.eye(number_of_classes)[y_batch]

            # Gradient of cross-entropy w.r.t. logits.
            gradient_logits = (
                probabilities - y_one_hot
            ) / len(X_batch)

            # Gradients of weights and bias.
            gradient_weights = (
                X_batch.T @ gradient_logits
            )

            # Gradient descent update.
            weights -= (
                learning_rate * gradient_weights
            )

        # Record loss after each epoch.
        train_loss = cross_entropy_loss(
            X_train,
            y_train,
            weights
        )

        val_loss = cross_entropy_loss(
            X_val,
            y_val,
            weights
        )

        epoch_time = time.perf_counter() - epoch_start

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(epoch_time)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} | "
                f"Train loss: {train_loss:.4f} | "
                f"Validation loss: {val_loss:.4f}"
            )

        # Stop if numerical instability occurs.
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            print(
                "Non-finite loss detected. "
                "Stopping this learning-rate experiment."
            )
            break

    # Evaluate final model for this learning rate.
    val_probabilities = softmax(X_val @ weights)
    predictions = np.argmax(val_probabilities, axis=1)

    val_accuracy = accuracy_score(
        y_val,
        predictions
    )

    result = {
        "learning_rate": learning_rate,
        "epochs_completed": len(train_losses),
        "final_training_loss": train_losses[-1],
        "final_validation_loss": val_losses[-1],
        "validation_accuracy": val_accuracy,
        "mean_epoch_time_seconds": np.mean(epoch_times)
    }

    all_results.append(result)

    loss_histories[learning_rate] = {
        "training": train_losses,
        "validation": val_losses
    }

    print("Final training loss:", round(train_losses[-1], 6))
    print("Final validation loss:", round(val_losses[-1], 6))
    print("Validation accuracy:", round(val_accuracy, 4))


# ------------------------------------------------------------
# 7. SAVE EXPERIMENTAL RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    OUTPUT_DIR / "task8_learning_rate_results.csv",
    index=False
)

print("\n========== LEARNING RATE COMPARISON ==========")
print(results_df.to_string(index=False))


# ------------------------------------------------------------
# 8. PLOT TRAINING LOSS
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

for learning_rate, history in loss_histories.items():

    plt.plot(
        range(1, len(history["training"]) + 1),
        history["training"],
        label=f"LR = {learning_rate}"
    )

plt.title("Training Loss for Different Learning Rates")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "task8_training_loss.png",
    dpi=200
)

plt.show()


# ------------------------------------------------------------
# 9. PLOT VALIDATION LOSS
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

for learning_rate, history in loss_histories.items():

    plt.plot(
        range(1, len(history["validation"]) + 1),
        history["validation"],
        label=f"LR = {learning_rate}"
    )

plt.title("Validation Loss for Different Learning Rates")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "task8_validation_loss.png",
    dpi=200
)

plt.show()


# ------------------------------------------------------------
# 10. SAVE SUMMARY
# ------------------------------------------------------------

best_result = results_df.loc[
    results_df["final_validation_loss"].idxmin()
]

summary = {
    "total_images_used": len(X),
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "epochs_requested": EPOCHS,
    "batch_size": BATCH_SIZE,
    "best_learning_rate_by_final_validation_loss":
        best_result["learning_rate"],
    "best_final_validation_loss":
        best_result["final_validation_loss"],
    "best_validation_accuracy":
        best_result["validation_accuracy"]
}

pd.Series(summary).to_csv(
    OUTPUT_DIR / "task8_summary.csv",
    header=["value"]
)

print("\n========== TASK 8 COMPLETE ==========")
print("Best learning rate by final validation loss:",
      best_result["learning_rate"])
print("Output folder:", OUTPUT_DIR.resolve())
print("Saved files:")
print("- task8_learning_rate_results.csv")
print("- task8_training_loss.png")
print("- task8_validation_loss.png")
print("- task8_summary.csv")
