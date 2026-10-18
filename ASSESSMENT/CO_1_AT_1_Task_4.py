
# ============================================================
# CSA47 - DEEP LEARNING
# ASSESSMENT 1 - TASK 4 (3 MARKS)
# Numerical Computation Issues
# Dataset: PlantVillage - Tomato Healthy, Early Blight,
#          and Late Blight
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from PIL import Image


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\VASANTH\Downloads\raw\color"
)

OUTPUT_DIR = Path("task4_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)

CLASS_NAMES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ------------------------------------------------------------
# 2. DEMONSTRATE SOFTMAX OVERFLOW
# ------------------------------------------------------------

def naive_softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def stable_softmax(z):
    # Subtracting the maximum leaves softmax unchanged
    # but prevents unnecessarily large exponentials.
    shifted_z = z - np.max(z)
    exp_z = np.exp(shifted_z)
    return exp_z / np.sum(exp_z)


print("\n========== SOFTMAX NUMERICAL STABILITY ==========")

z = np.array([1000.0, 999.0, 995.0])

print("Input logits:", z)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    naive_result = naive_softmax(z)

print("Naive softmax:", naive_result)
print("Naive result contains non-finite values:",
      not np.all(np.isfinite(naive_result)))

stable_result = stable_softmax(z)

print("Stable softmax:", stable_result)
print("Stable probabilities sum:", stable_result.sum())

assert np.all(np.isfinite(stable_result))
assert np.isclose(stable_result.sum(), 1.0)

# Expected approximately [0.7275, 0.2680, 0.0049]


# ------------------------------------------------------------
# 3. DEMONSTRATE THE LOG-SUM-EXP TRICK
# ------------------------------------------------------------

def log_sum_exp(z):
    m = np.max(z)
    return m + np.log(np.sum(np.exp(z - m)))


print("\n========== LOG-SUM-EXP ==========")

naive_log_sum_exp = None

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    naive_log_sum_exp = np.log(np.sum(np.exp(z)))

stable_log_sum_exp = log_sum_exp(z)

print("Naive log-sum-exp:", naive_log_sum_exp)
print("Stable log-sum-exp:", stable_log_sum_exp)

# The stable result should be finite.
assert np.isfinite(stable_log_sum_exp)


# ------------------------------------------------------------
# 4. LOAD A SAMPLE OF THE REAL DATASET
# ------------------------------------------------------------

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR}"
    )

feature_rows = []
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

    print(f"{class_name}: {len(image_paths)} images found")

    # Limit the diagnostic sample per class to control memory.
    for image_path in image_paths[:200]:
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

            feature_rows.append(pixels.flatten())
            labels.append(class_name)

        except Exception as error:
            print("Skipped:", image_path, error)

X = np.asarray(feature_rows, dtype=np.float64)

if X.shape[0] < 2:
    raise ValueError("Not enough valid images for covariance analysis.")

print("\n========== FEATURE MATRIX ==========")
print("Sample images:", X.shape[0])
print("Features per image:", X.shape[1])
print("Feature matrix shape:", X.shape)


# ------------------------------------------------------------
# 5. ESTIMATE CONDITIONING USING A FEATURE SUBSET
# ------------------------------------------------------------

# Using all 4096 features to form and diagonalise a covariance
# matrix can be expensive. This diagnostic uses the first 256
# pixel features and clearly reports that limitation.

NUM_FEATURES = min(256, X.shape[1])

X_subset = X[:, :NUM_FEATURES]

# Centre each feature before calculating covariance.
X_centered = X_subset - X_subset.mean(axis=0)

covariance = np.cov(X_centered, rowvar=False)

eigenvalues = np.linalg.eigvalsh(covariance)

largest_eigenvalue = float(np.max(eigenvalues))
smallest_eigenvalue = float(np.min(eigenvalues))

# Small negative eigenvalues can occur from floating-point error.
# A zero or near-zero eigenvalue makes the unregularised
# condition number infinite or extremely large.
tolerance = (
    np.finfo(float).eps
    * max(covariance.shape)
    * max(abs(largest_eigenvalue), 1.0)
)

positive_eigenvalues = eigenvalues[eigenvalues > tolerance]

print("\n========== COVARIANCE CONDITIONING ==========")
print("Features analysed:", NUM_FEATURES)
print("Covariance matrix shape:", covariance.shape)
print("Largest eigenvalue:", largest_eigenvalue)
print("Smallest eigenvalue:", smallest_eigenvalue)

if len(positive_eigenvalues) == 0:
    condition_number = np.inf
    print("No numerically positive eigenvalues were found.")
else:
    smallest_positive = float(np.min(positive_eigenvalues))
    condition_number = largest_eigenvalue / smallest_positive

    print("Smallest positive eigenvalue:", smallest_positive)
    print("Estimated condition number:", condition_number)

print(
    "Note: this is a diagnostic for the selected feature subset, "
    "not the condition number of the full 4096-feature covariance."
)


# ------------------------------------------------------------
# 6. EXAMINE THE EFFECT OF REGULARISATION
# ------------------------------------------------------------

# Adding lambda*I increases the eigenvalues by lambda and
# can improve conditioning.

regularisation_values = [0.0, 1e-6, 1e-4, 1e-2]
regularisation_results = []

for lam in regularisation_values:
    regularised_covariance = (
        covariance + lam * np.eye(NUM_FEATURES)
    )

    reg_eigenvalues = np.linalg.eigvalsh(
        regularised_covariance
    )

    reg_max = float(np.max(reg_eigenvalues))
    reg_min = float(np.min(reg_eigenvalues))

    if reg_min <= 0:
        reg_condition = np.inf
    else:
        reg_condition = reg_max / reg_min

    regularisation_results.append({
        "lambda": lam,
        "minimum_eigenvalue": reg_min,
        "maximum_eigenvalue": reg_max,
        "condition_number": reg_condition
    })

regularisation_df = pd.DataFrame(regularisation_results)

print("\n========== REGULARISATION COMPARISON ==========")
print(regularisation_df.to_string(index=False))

regularisation_df.to_csv(
    OUTPUT_DIR / "task4_conditioning_results.csv",
    index=False
)


# ------------------------------------------------------------
# 7. PLOT CONDITION NUMBER
# ------------------------------------------------------------

plot_df = regularisation_df.copy()

# Avoid attempting to plot infinity directly.
plot_df["condition_number_plot"] = (
    plot_df["condition_number"].replace(
        [np.inf, -np.inf], np.nan
    )
)

plt.figure(figsize=(8, 5))

plt.plot(
    plot_df["lambda"].astype(str),
    plot_df["condition_number_plot"],
    marker="o"
)

plt.title("Covariance Conditioning with Regularisation")
plt.xlabel("Regularisation Strength (lambda)")
plt.ylabel("Condition Number")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "task4_condition_number.png",
    dpi=200
)
plt.show()


# ------------------------------------------------------------
# 8. SAVE SUMMARY
# ------------------------------------------------------------

summary = {
    "input_logits": str(z.tolist()),
    "stable_softmax": str(stable_result.tolist()),
    "stable_probabilities_sum": stable_result.sum(),
    "stable_log_sum_exp": stable_log_sum_exp,
    "sample_images": X.shape[0],
    "features_in_conditioning_diagnostic": NUM_FEATURES,
    "largest_covariance_eigenvalue": largest_eigenvalue,
    "smallest_covariance_eigenvalue": smallest_eigenvalue,
    "estimated_condition_number": condition_number
}

pd.Series(summary).to_csv(
    OUTPUT_DIR / "task4_summary.csv",
    header=["value"]
)

print("\n========== TASK 4 COMPLETE ==========")
print("Output folder:", OUTPUT_DIR.resolve())
print("Saved files:")
print("- task4_conditioning_results.csv")
print("- task4_condition_number.png")
print("- task4_summary.csv")
