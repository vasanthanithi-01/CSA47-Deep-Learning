
# ============================================================
# CSA47 - DEEP LEARNING
# ASSESSMENT 1 - TASK 5 (4 MARKS)
# Baseline Learning Strategies
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
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\VASANTH\Downloads\raw\color"
)

OUTPUT_DIR = Path("task5_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)
PCA_COMPONENTS = 50
TEST_SIZE = 0.20
RANDOM_STATE = 42

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

if len(X) < 10:
    raise ValueError("Not enough images were loaded.")

print("\n========== DATASET SUMMARY ==========")
print("Feature matrix:", X.shape)
print("Number of classes:", len(np.unique(y)))
print(pd.Series(y).value_counts().sort_index())


# ------------------------------------------------------------
# 3. SPLIT DATA INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 4. PCA COMPRESSION
# ------------------------------------------------------------

n_components = min(
    PCA_COMPONENTS,
    X_train.shape[0] - 1,
    X_train.shape[1]
)

pca = PCA(
    n_components=n_components,
    svd_solver="randomized",
    random_state=RANDOM_STATE
)

start = time.perf_counter()

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

pca_time = time.perf_counter() - start

variance_retained = (
    np.sum(pca.explained_variance_ratio_) * 100
)

print("\n========== PCA RESULTS ==========")
print("Original feature count:", X_train.shape[1])
print("Compressed feature count:", n_components)
print("Variance retained:", round(variance_retained, 4), "%")
print("PCA fitting and transformation time:",
      round(pca_time, 4), "seconds")

pd.DataFrame({
    "component": np.arange(1, n_components + 1),
    "explained_variance_ratio": pca.explained_variance_ratio_
}).to_csv(
    OUTPUT_DIR / "task5_pca_variance.csv",
    index=False
)


# ------------------------------------------------------------
# 5. DEFINE THREE DIFFERENT LEARNING ALGORITHMS
# ------------------------------------------------------------

def create_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        ),

        "K-Nearest Neighbours": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=-1
            )
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=80,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }


# ------------------------------------------------------------
# 6. TRAIN AND EVALUATE ON RAW AND PCA FEATURES
# ------------------------------------------------------------

experiments = [
    ("Raw Features", X_train, X_test),
    ("PCA Features", X_train_pca, X_test_pca)
]

results = []
models_for_evaluation = {}

for feature_type, train_features, test_features in experiments:

    print("\n")
    print("=" * 60)
    print("FEATURE REPRESENTATION:", feature_type)
    print("Training shape:", train_features.shape)
    print("Testing shape:", test_features.shape)
    print("=" * 60)

    models = create_models()

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        start_train = time.perf_counter()
        model.fit(train_features, y_train)
        train_time = time.perf_counter() - start_train

        start_predict = time.perf_counter()
        predictions = model.predict(test_features)
        predict_time = time.perf_counter() - start_predict

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            labels=CLASS_NAMES,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            labels=CLASS_NAMES,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            labels=CLASS_NAMES,
            average="macro",
            zero_division=0
        )

        result = {
            "feature_type": feature_type,
            "model": model_name,
            "feature_count": train_features.shape[1],
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
            "training_time_seconds": train_time,
            "prediction_time_seconds": predict_time,
            "prediction_time_ms_per_image": (
                predict_time / len(test_features) * 1000
            )
        }

        results.append(result)

        models_for_evaluation[(feature_type, model_name)] = (
            model,
            test_features,
            predictions
        )

        print("Accuracy:", round(accuracy, 4))
        print("Macro precision:", round(precision, 4))
        print("Macro recall:", round(recall, 4))
        print("Macro F1-score:", round(f1, 4))
        print("Training time:", round(train_time, 4), "seconds")
        print("Prediction time:", round(predict_time, 4), "seconds")
        print(
            "Average prediction time:",
            round(result["prediction_time_ms_per_image"], 4),
            "ms/image"
        )


# ------------------------------------------------------------
# 7. SAVE PERFORMANCE COMPARISON
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "task5_model_comparison.csv",
    index=False
)

print("\n========== FINAL MODEL COMPARISON ==========")
print(
    results_df.sort_values(
        by="macro_f1",
        ascending=False
    ).to_string(index=False)
)


# ------------------------------------------------------------
# 8. GENERATE CLASSIFICATION REPORTS AND CONFUSION MATRICES
# ------------------------------------------------------------

for key, (_, test_features, predictions) in (
    models_for_evaluation.items()
):

    feature_type, model_name = key

    safe_name = (
        feature_type.lower().replace(" ", "_")
        + "_"
        + model_name.lower().replace(" ", "_")
    )

    print(f"\n========== {feature_type}: {model_name} ==========")

    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print(report)

    with open(
        OUTPUT_DIR / f"{safe_name}_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=CLASS_NAMES,
        display_labels=[
            "Healthy",
            "Early Blight",
            "Late Blight"
        ],
        xticks_rotation=25,
        cmap="Blues"
    )

    display.ax_.set_title(
        f"{feature_type}: {model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{safe_name}_confusion_matrix.png",
        dpi=200
    )

    plt.show()


# ------------------------------------------------------------
# 9. PLOT MACRO F1-SCORE COMPARISON
# ------------------------------------------------------------

plot_df = results_df.copy()

plot_df["experiment"] = (
    plot_df["feature_type"] + "\n" + plot_df["model"]
)

plt.figure(figsize=(12, 6))

plt.bar(
    plot_df["experiment"],
    plot_df["macro_f1"]
)

plt.title("Model Comparison: Macro F1-Score")
plt.xlabel("Feature Representation and Model")
plt.ylabel("Macro F1-Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "task5_f1_comparison.png",
    dpi=200
)

plt.show()


# ------------------------------------------------------------
# 10. PLOT TRAINING TIME COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.bar(
    plot_df["experiment"],
    plot_df["training_time_seconds"]
)

plt.title("Model Training Time Comparison")
plt.xlabel("Feature Representation and Model")
plt.ylabel("Training Time (seconds)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "task5_training_time.png",
    dpi=200
)

plt.show()


# ------------------------------------------------------------
# 11. SAVE SUMMARY
# ------------------------------------------------------------

summary = {
    "total_images": len(y),
    "number_of_classes": len(np.unique(y)),
    "raw_feature_count": X.shape[1],
    "pca_feature_count": n_components,
    "pca_variance_retained_percent": variance_retained,
    "pca_time_seconds": pca_time
}

best_row = results_df.loc[results_df["macro_f1"].idxmax()]

summary["best_model_by_macro_f1"] = (
    best_row["feature_type"] + " - " + best_row["model"]
)

summary["best_macro_f1"] = best_row["macro_f1"]

pd.Series(summary).to_csv(
    OUTPUT_DIR / "task5_summary.csv",
    header=["value"]
)

print("\n========== TASK 5 COMPLETE ==========")
print("Output folder:", OUTPUT_DIR.resolve())
print("Files include:")
print("- task5_model_comparison.csv")
print("- task5_pca_variance.csv")
print("- task5_summary.csv")
print("- Classification reports")
print("- Confusion matrix plots")
print("- F1-score comparison plot")
print("- Training time comparison plot")
