
# ============================================================
# CSA47 - DEEP LEARNING
# ASSESSMENT 1 - TASK 3 (4 MARKS)
# Uncertainty Analysis using Probability and Information Theory
# Dataset: PlantVillage - Tomato Healthy, Early Blight,
#          and Late Blight
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\VASANTH\Downloads\raw\color"
)

OUTPUT_DIR = Path("task3_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)
RANDOM_STATE = 42
EPSILON = 1e-12

CLASS_NAMES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ------------------------------------------------------------
# 2. LOAD AND PREPROCESS DATASET
# ------------------------------------------------------------

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR}"
    )

images = []
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

    print(f"{class_name}: {len(image_paths)} images")

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

            images.append(pixels.flatten())
            labels.append(class_name)

        except Exception as error:
            print("Skipped:", image_path, error)

X = np.asarray(images, dtype=np.float32)
y = np.asarray(labels)

if len(X) < 10:
    raise ValueError("Not enough images were loaded.")

print("\n========== DATASET SUMMARY ==========")
print("Feature matrix shape:", X.shape)
print("Total images:", len(y))

class_to_index = {
    name: index for index, name in enumerate(CLASS_NAMES)
}
y_encoded = np.array([class_to_index[label] for label in y])


# ------------------------------------------------------------
# 3. EMPIRICAL CLASS DISTRIBUTION AND ENTROPY
# ------------------------------------------------------------

class_counts = np.bincount(
    y_encoded,
    minlength=len(CLASS_NAMES)
)

p = class_counts / class_counts.sum()

def entropy(distribution):
    """Shannon entropy in bits."""
    distribution = np.asarray(distribution, dtype=float)
    distribution = distribution[distribution > 0]
    return -np.sum(distribution * np.log2(distribution))

H_p = entropy(p)
maximum_entropy = np.log2(len(CLASS_NAMES))

print("\n========== CLASS DISTRIBUTION ==========")

for name, count, probability in zip(CLASS_NAMES, class_counts, p):
    print(
        f"{name}: count={count}, "
        f"probability={probability:.6f}"
    )

print("\nEntropy H(p):", round(H_p, 6), "bits")
print("Maximum possible entropy:", round(maximum_entropy, 6), "bits")

distribution_df = pd.DataFrame({
    "class_name": CLASS_NAMES,
    "count": class_counts,
    "probability": p
})

distribution_df.to_csv(
    OUTPUT_DIR / "task3_class_distribution.csv",
    index=False
)

plt.figure(figsize=(8, 5))
plt.bar(CLASS_NAMES, p)
plt.title("Empirical Tomato Class Distribution")
plt.xlabel("Class")
plt.ylabel("Probability")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "task3_class_distribution.png",
    dpi=200
)
plt.show()


# ------------------------------------------------------------
# 4. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print("\n========== DATA SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 5. DEFINE CLASSICAL BASELINE MODELS
# ------------------------------------------------------------

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# ------------------------------------------------------------
# 6. CROSS-ENTROPY AND KL DIVERGENCE FUNCTIONS
# ------------------------------------------------------------

def cross_entropy(p_true, q_pred):
    """
    Cross-entropy H(p, q) = -sum p_i log2(q_i).
    Both inputs must be probability distributions.
    """
    p_true = np.asarray(p_true, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)

    q_pred = np.clip(q_pred, EPSILON, 1.0)
    q_pred = q_pred / q_pred.sum()

    return -np.sum(p_true * np.log2(q_pred))


def kl_divergence(p_true, q_pred):
    """
    KL(p || q) = sum p_i log2(p_i / q_i).
    """
    p_true = np.asarray(p_true, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)

    p_safe = np.clip(p_true, EPSILON, 1.0)
    q_safe = np.clip(q_pred, EPSILON, 1.0)

    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()

    return np.sum(
        p_safe * np.log2(p_safe / q_safe)
    )


# ------------------------------------------------------------
# 7. TRAIN MODELS AND COMPARE DISTRIBUTIONS
# ------------------------------------------------------------

# Empirical distribution of actual labels in the test set
test_counts = np.bincount(
    y_test,
    minlength=len(CLASS_NAMES)
)
p_test = test_counts / test_counts.sum()

H_test = entropy(p_test)

print("\n========== TEST SET TRUE DISTRIBUTION ==========")
for name, probability in zip(CLASS_NAMES, p_test):
    print(f"{name}: {probability:.6f}")

print("H(p):", round(H_test, 6), "bits")

results = []
classwise_results = []

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    # Predicted probabilities for each test image
    probabilities = model.predict_proba(X_test)

    # Align probabilities to the three expected class indices
    aligned_probabilities = np.zeros(
        (len(X_test), len(CLASS_NAMES)),
        dtype=float
    )

    for column, class_index in enumerate(model.classes_):
        aligned_probabilities[:, int(class_index)] = (
            probabilities[:, column]
        )

    # Average predicted probability across test samples.
    # q is the model's aggregate predicted class distribution.
    q = aligned_probabilities.mean(axis=0)
    q = np.clip(q, EPSILON, 1.0)
    q = q / q.sum()

    # Information-theoretic metrics
    H_q = entropy(q)
    H_pq = cross_entropy(p_test, q)
    KL_pq = kl_divergence(p_test, q)
    KL_qp = kl_divergence(q, p_test)

    # Numerical identity:
    # H(p, q) = H(p) + KL(p || q)
    identity_difference = H_pq - (H_test + KL_pq)

    accuracy = np.mean(model.predict(X_test) == y_test)

    results.append({
        "model": model_name,
        "test_accuracy": accuracy,
        "H_p_bits": H_test,
        "H_q_bits": H_q,
        "cross_entropy_H_pq_bits": H_pq,
        "KL_pq_bits": KL_pq,
        "KL_qp_bits": KL_qp,
        "identity_difference": identity_difference
    })

    print("\nModel:", model_name)
    print("Test accuracy:", round(accuracy, 6))
    print("Predicted distribution q:", np.round(q, 6))
    print("H(p):", round(H_test, 6), "bits")
    print("H(q):", round(H_q, 6), "bits")
    print("Cross-entropy H(p, q):", round(H_pq, 6), "bits")
    print("KL(p || q):", round(KL_pq, 6), "bits")
    print("KL(q || p):", round(KL_qp, 6), "bits")
    print(
        "Identity difference:",
        f"{identity_difference:.12f}"
    )

    for i, class_name in enumerate(CLASS_NAMES):
        classwise_results.append({
            "model": model_name,
            "class_name": class_name,
            "true_probability": p_test[i],
            "predicted_probability": q[i],
            "KL_pq_contribution_bits": (
                p_test[i] * np.log2(
                    max(p_test[i], EPSILON) /
                    max(q[i], EPSILON)
                )
            )
        })


# ------------------------------------------------------------
# 8. SAVE COMPARISON RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "task3_information_metrics.csv",
    index=False
)

classwise_df = pd.DataFrame(classwise_results)

classwise_df.to_csv(
    OUTPUT_DIR / "task3_classwise_analysis.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))


# ------------------------------------------------------------
# 9. ILLUSTRATE KL DIVERGENCE ASYMMETRY
# ------------------------------------------------------------

# Independent numerical example for explaining asymmetry.
p_example = np.array([0.8, 0.1, 0.1])
q_example = np.array([0.4, 0.3, 0.3])

print("\n========== KL ASYMMETRY EXAMPLE ==========")
print("Illustrative p:", p_example)
print("Illustrative q:", q_example)

print(
    "KL(p || q):",
    round(kl_divergence(p_example, q_example), 6),
    "bits"
)

print(
    "KL(q || p):",
    round(kl_divergence(q_example, p_example), 6),
    "bits"
)

print(
    "Cross-entropy H(p, q):",
    round(cross_entropy(p_example, q_example), 6),
    "bits"
)

print(
    "Entropy H(p):",
    round(entropy(p_example), 6),
    "bits"
)


# ------------------------------------------------------------
# 10. SAVE SUMMARY
# ------------------------------------------------------------

summary = {
    "total_images": len(y),
    "number_of_classes": len(CLASS_NAMES),
    "test_images": len(y_test),
    "test_entropy_bits": H_test,
    "maximum_entropy_bits": maximum_entropy
}

for _, row in results_df.iterrows():
    key = row["model"].lower().replace(" ", "_")
    summary[f"{key}_accuracy"] = row["test_accuracy"]
    summary[f"{key}_cross_entropy_bits"] = (
        row["cross_entropy_H_pq_bits"]
    )
    summary[f"{key}_KL_pq_bits"] = row["KL_pq_bits"]
    summary[f"{key}_KL_qp_bits"] = row["KL_qp_bits"]

pd.Series(summary).to_csv(
    OUTPUT_DIR / "task3_summary.csv",
    header=["value"]
)

print("\n========== TASK 3 COMPLETE ==========")
print("Output folder:", OUTPUT_DIR.resolve())
print("Saved files:")
print("- task3_class_distribution.csv")
print("- task3_class_distribution.png")
print("- task3_information_metrics.csv")
print("- task3_classwise_analysis.csv")
print("- task3_summary.csv")
