
# ============================================================
# CSA47 - DEEP LEARNING
# ASSESSMENT 1 - TASK 7 (4 MARKS)
# Bias-Variance and Estimation Analysis
# Dataset: PlantVillage - Tomato Healthy, Early Blight,
#          and Late Blight
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\VASANTH\Downloads\raw\color"
)

OUTPUT_DIR = Path("task7_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)
MAX_IMAGES_PER_CLASS = 300
PCA_COMPONENTS = 50
BOOTSTRAP_REPLICATES = 10
RANDOM_STATE = 42

CLASS_NAMES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ------------------------------------------------------------
# 2. LOAD THE ACTUAL DATASET
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

    # Limit images per class to make repeated bootstrap
    # experiments more manageable on a typical computer.
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

X = np.asarray(features, dtype=np.float32)
y = np.asarray(labels)

if len(X) < 30:
    raise ValueError("Not enough images were loaded.")

print("\n========== DATASET ==========")
print("Feature matrix:", X.shape)
print("Total images:", len(y))
print("Class counts:")
print(pd.Series(y).value_counts().sort_index())


# ------------------------------------------------------------
# 3. TRAIN, VALIDATION AND TEST SPLITS
# ------------------------------------------------------------

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_trainval
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Reserved test samples:", len(X_test))


# ------------------------------------------------------------
# 4. APPLY PCA USING TRAINING DATA ONLY
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

X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)

print("\n========== PCA ==========")
print("Original features:", X_train.shape[1])
print("Retained components:", n_components)
print(
    "Variance retained (%):",
    round(np.sum(pca.explained_variance_ratio_) * 100, 4)
)


# ------------------------------------------------------------
# 5. DEFINE LOW- AND HIGH-CAPACITY MODELS
# ------------------------------------------------------------

models = {
    "Low Capacity - Logistic Regression":
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.01,
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        ),

    "High Capacity - Random Forest":
        RandomForestClassifier(
            n_estimators=60,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
}


# ------------------------------------------------------------
# 6. BOOTSTRAP BIAS-VARIANCE ANALYSIS
# ------------------------------------------------------------

rng = np.random.default_rng(RANDOM_STATE)

class_to_index = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

y_val_encoded = np.array([
    class_to_index[label] for label in y_val
])

Y_true = np.eye(len(CLASS_NAMES))[y_val_encoded]

all_results = []
prediction_variances = {}
prediction_biases = {}

for model_name, model_template in models.items():

    print(f"\n========== {model_name} ==========")

    probability_predictions = []
    bootstrap_accuracies = []

    for replicate in range(BOOTSTRAP_REPLICATES):

        # Sample training examples with replacement.
        bootstrap_indices = rng.integers(
            0,
            len(X_train_pca),
            size=len(X_train_pca)
        )

        X_boot = X_train_pca[bootstrap_indices]
        y_boot = y_train[bootstrap_indices]

        # Create a fresh model for each replicate.
        if "Logistic Regression" in model_name:

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.01,
                    max_iter=1000,
                    random_state=RANDOM_STATE + replicate
                )
            )

        else:

            model = RandomForestClassifier(
                n_estimators=60,
                max_depth=None,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=RANDOM_STATE + replicate,
                n_jobs=-1
            )

        model.fit(X_boot, y_boot)

        probabilities = model.predict_proba(X_val_pca)

        # Align output columns with the predefined class order.
        aligned = np.zeros(
            (len(X_val_pca), len(CLASS_NAMES)),
            dtype=float
        )

        for column, class_name in enumerate(model.classes_):
            aligned[:, class_to_index[class_name]] = (
                probabilities[:, column]
            )

        probability_predictions.append(aligned)

        predictions = model.predict(X_val_pca)

        bootstrap_accuracies.append(
            accuracy_score(y_val, predictions)
        )

        print(
            f"Replicate {replicate + 1}: "
            f"validation accuracy = "
            f"{bootstrap_accuracies[-1]:.4f}"
        )

    # Shape: (replicates, validation samples, classes)
    predictions_array = np.asarray(probability_predictions)

    mean_prediction = predictions_array.mean(axis=0)

    # Average across samples and classes.
    # This is a Brier-style squared-probability analysis.
    bias_squared = np.mean(
        (mean_prediction - Y_true) ** 2
    )

    variance = np.mean(
        np.var(predictions_array, axis=0, ddof=0)
    )

    mean_validation_accuracy = np.mean(bootstrap_accuracies)
    std_validation_accuracy = np.std(
        bootstrap_accuracies,
        ddof=0
    )

    prediction_variances[model_name] = variance
    prediction_biases[model_name] = bias_squared

    all_results.append({
        "model": model_name,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "mean_validation_accuracy": mean_validation_accuracy,
        "std_validation_accuracy": std_validation_accuracy,
        "mean_squared_probability_bias": bias_squared,
        "mean_prediction_variance": variance
    })

    print("Mean validation accuracy:",
          round(mean_validation_accuracy, 4))
    print("Accuracy standard deviation:",
          round(std_validation_accuracy, 4))
    print("Mean squared probability bias:",
          round(bias_squared, 6))
    print("Mean prediction variance:",
          round(variance, 6))


# ------------------------------------------------------------
# 7. SAVE BIAS-VARIANCE RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    OUTPUT_DIR / "task7_bias_variance_results.csv",
    index=False
)

print("\n========== BIAS-VARIANCE COMPARISON ==========")
print(results_df.to_string(index=False))


# ------------------------------------------------------------
# 8. PLOT BIAS AND VARIANCE
# ------------------------------------------------------------

plot_df = results_df.set_index("model")[
    [
        "mean_squared_probability_bias",
        "mean_prediction_variance"
    ]
]

plot_df.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Bootstrap Bias and Prediction Variance")
plt.xlabel("Model")
plt.ylabel("Mean Squared Probability")
plt.xticks(rotation=20, ha="right")
plt.legend(["Squared bias", "Prediction variance"])
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "task7_bias_variance.png",
    dpi=200
)

plt.show()


# ------------------------------------------------------------
# 9. MLE AND MAP ESTIMATION EXAMPLE
# ------------------------------------------------------------

# Illustrative Bernoulli experiment:
# 7 successes in 10 trials.

successes = 7
trials = 10

# Maximum Likelihood Estimate
mle = successes / trials

# Beta prior parameters
alpha = 2
beta = 2

# Beta posterior parameters
posterior_alpha = alpha + successes
posterior_beta = beta + trials - successes

# MAP estimate for alpha > 1 and beta > 1
map_estimate = (
    posterior_alpha - 1
) / (
    posterior_alpha + posterior_beta - 2
)

print("\n========== MLE AND MAP ==========")
print("Successes:", successes)
print("Trials:", trials)
print("MLE:", round(mle, 4))
print("Prior: Beta(", alpha, ",", beta, ")")
print(
    "Posterior: Beta(",
    posterior_alpha,
    ",",
    posterior_beta,
    ")"
)
print("MAP estimate:", round(map_estimate, 4))

estimation_df = pd.DataFrame([
    {
        "method": "MLE",
        "estimate": mle
    },
    {
        "method": "MAP",
        "estimate": map_estimate
    }
])

estimation_df.to_csv(
    OUTPUT_DIR / "task7_mle_map_results.csv",
    index=False
)

plt.figure(figsize=(7, 5))

plt.bar(
    estimation_df["method"],
    estimation_df["estimate"]
)

plt.ylim(0, 1)
plt.title("MLE and MAP Estimation")
plt.ylabel("Estimated Probability")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "task7_mle_map.png",
    dpi=200
)

plt.show()


# ------------------------------------------------------------
# 10. SAVE SUMMARY
# ------------------------------------------------------------

summary = {
    "total_images_used": len(y),
    "training_samples": len(y_train),
    "validation_samples": len(y_val),
    "pca_components": n_components,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "Bernoulli_MLE": mle,
    "Bernoulli_MAP": map_estimate
}

pd.Series(summary).to_csv(
    OUTPUT_DIR / "task7_summary.csv",
    header=["value"]
)

print("\n========== TASK 7 COMPLETE ==========")
print("Output folder:", OUTPUT_DIR.resolve())
print("Saved files:")
print("- task7_bias_variance_results.csv")
print("- task7_bias_variance.png")
print("- task7_mle_map_results.csv")
print("- task7_mle_map.png")
print("- task7_summary.csv")
