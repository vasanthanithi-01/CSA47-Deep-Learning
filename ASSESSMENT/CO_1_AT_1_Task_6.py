
# ============================================================
# CSA47 - DEEP LEARNING
# ASSESSMENT 1 - TASK 6 (4 MARKS)
# Capacity, Overfitting and Underfitting
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\VASANTH\Downloads\raw\color"
)

OUTPUT_DIR = Path("task6_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
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

            features.append(pixels.flatten())
            labels.append(class_name)

        except Exception as error:
            print("Skipped:", image_path, error)

X = np.asarray(features, dtype=np.float32)
y = np.asarray(labels)

if len(X) < 20:
    raise ValueError("Not enough images were loaded.")

print("\n========== DATASET SUMMARY ==========")
print("Feature matrix shape:", X.shape)
print("Total images:", len(y))


# ------------------------------------------------------------
# 3. TRAIN, VALIDATION AND TEST SPLITS
# ------------------------------------------------------------

# 20% is reserved for the final test set.
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# Of the remaining 80%, reserve 20% for validation.
# Final proportions: 64% train, 16% validation, 20% test.
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_trainval
)

print("\n========== DATA SPLIT ==========")
print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Test images:", len(X_test))

# The test set is deliberately not used for model selection.


# ------------------------------------------------------------
# 4. DEFINE MODELS WITH DIFFERENT CAPACITIES
# ------------------------------------------------------------

experiments = []

# Logistic Regression:
# Smaller C means stronger regularisation.
for C in [0.01, 0.1, 1.0, 10.0]:
    experiments.append({
        "model_name": "Logistic Regression",
        "setting": f"C={C}",
        "model": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    })

# KNN:
# Small k is more flexible; larger k smooths predictions.
for k in [1, 5, 15, 31]:
    experiments.append({
        "model_name": "KNN",
        "setting": f"k={k}",
        "model": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=k,
                weights="uniform",
                n_jobs=-1
            )
        )
    })

# Random Forest:
# Deeper trees can represent more complex decision boundaries.
for depth in [3, 8, 15, None]:
    experiments.append({
        "model_name": "Random Forest",
        "setting": f"max_depth={depth}",
        "model": RandomForestClassifier(
            n_estimators=60,
            max_depth=depth,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    })


# ------------------------------------------------------------
# 5. TRAIN MODELS AND MEASURE CAPACITY BEHAVIOUR
# ------------------------------------------------------------

results = []

for experiment in experiments:

    model_name = experiment["model_name"]
    setting = experiment["setting"]
    model = experiment["model"]

    print(f"\nTraining {model_name} ({setting})...")

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time

    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    train_error = 1 - train_accuracy
    val_error = 1 - val_accuracy

    generalisation_gap = train_accuracy - val_accuracy

    results.append({
        "model": model_name,
        "setting": setting,
        "training_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "training_error": train_error,
        "validation_error": val_error,
        "generalisation_gap": generalisation_gap,
        "training_time_seconds": training_time
    })

    print("Training accuracy:", round(train_accuracy, 4))
    print("Validation accuracy:", round(val_accuracy, 4))
    print("Training error:", round(train_error, 4))
    print("Validation error:", round(val_error, 4))
    print("Generalisation gap:", round(generalisation_gap, 4))
    print("Training time:", round(training_time, 3), "seconds")


# ------------------------------------------------------------
# 6. SAVE RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "task6_capacity_results.csv",
    index=False
)

print("\n========== CAPACITY RESULTS ==========")
print(
    results_df[
        [
            "model",
            "setting",
            "training_accuracy",
            "validation_accuracy",
            "generalisation_gap"
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# 7. PLOT TRAINING AND VALIDATION ACCURACY
# ------------------------------------------------------------

for model_name in results_df["model"].unique():

    subset = results_df[
        results_df["model"] == model_name
    ].copy()

    positions = np.arange(len(subset))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        positions - width / 2,
        subset["training_accuracy"],
        width,
        label="Training accuracy"
    )

    plt.bar(
        positions + width / 2,
        subset["validation_accuracy"],
        width,
        label="Validation accuracy"
    )

    plt.xticks(
        positions,
        subset["setting"],
        rotation=20,
        ha="right"
    )

    plt.title(f"{model_name}: Capacity and Accuracy")
    plt.xlabel("Model setting")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    filename = (
        model_name.lower().replace(" ", "_")
        + "_capacity_accuracy.png"
    )

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=200
    )

    plt.show()


# ------------------------------------------------------------
# 8. PLOT GENERALISATION GAP
# ------------------------------------------------------------

for model_name in results_df["model"].unique():

    subset = results_df[
        results_df["model"] == model_name
    ].copy()

    plt.figure(figsize=(8, 5))

    plt.bar(
        subset["setting"],
        subset["generalisation_gap"]
    )

    plt.axhline(0, linewidth=1)

    plt.title(f"{model_name}: Generalisation Gap")
    plt.xlabel("Model setting")
    plt.ylabel("Training Accuracy - Validation Accuracy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    filename = (
        model_name.lower().replace(" ", "_")
        + "_generalisation_gap.png"
    )

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=200
    )

    plt.show()


# ------------------------------------------------------------
# 9. IDENTIFY BEST SETTINGS USING VALIDATION ACCURACY
# ------------------------------------------------------------

print("\n========== BEST VALIDATION SETTINGS ==========")

best_settings = (
    results_df.sort_values(
        "validation_accuracy",
        ascending=False
    )
    .groupby("model", sort=False)
    .head(1)
)

print(
    best_settings[
        [
            "model",
            "setting",
            "training_accuracy",
            "validation_accuracy",
            "generalisation_gap"
        ]
    ].to_string(index=False)
)

best_settings.to_csv(
    OUTPUT_DIR / "task6_best_validation_settings.csv",
    index=False
)


# ------------------------------------------------------------
# 10. SAVE SUMMARY
# ------------------------------------------------------------

summary = {
    "total_images": len(y),
    "training_images": len(y_train),
    "validation_images": len(y_val),
    "reserved_test_images": len(y_test),
    "number_of_experiments": len(results_df)
}

for model_name in results_df["model"].unique():

    subset = results_df[
        results_df["model"] == model_name
    ]

    best_row = subset.loc[
        subset["validation_accuracy"].idxmax()
    ]

    key = model_name.lower().replace(" ", "_")

    summary[f"{key}_best_setting"] = best_row["setting"]
    summary[f"{key}_best_validation_accuracy"] = (
        best_row["validation_accuracy"]
    )
    summary[f"{key}_best_generalisation_gap"] = (
        best_row["generalisation_gap"]
    )

pd.Series(summary).to_csv(
    OUTPUT_DIR / "task6_summary.csv",
    header=["value"]
)

print("\n========== TASK 6 COMPLETE ==========")
print("Output folder:", OUTPUT_DIR.resolve())
print("Saved:")
print("- task6_capacity_results.csv")
print("- task6_best_validation_settings.csv")
print("- task6_summary.csv")
print("- Capacity and accuracy plots")
print("- Generalisation gap plots")
