
# ============================================================
# CSA47 - DEEP LEARNING
# ASSESSMENT 1 - TASK 2 (5 MARKS)
# Feature-Space Analysis using Eigenvalues and SVD
# Dataset: PlantVillage - Tomato Healthy, Early Blight,
#          and Late Blight
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from PIL import Image
from sklearn.decomposition import PCA


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\VASANTH\Downloads\raw\color"
)

OUTPUT_DIR = Path("task2_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)
RANDOM_STATE = 42

CLASS_NAMES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ------------------------------------------------------------
# 2. VERIFY DATASET AND LOAD IMAGES
# ------------------------------------------------------------

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR}\n"
        "Check the extraction path."
    )

images = []
labels = []

for class_name in CLASS_NAMES:
    class_dir = DATA_DIR / class_name

    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Class folder not found: {class_dir}\n"
            "Check the folder names inside raw\\color."
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

if len(X) < 2:
    raise ValueError(
        "Fewer than two images were loaded. Check the dataset."
    )

print("\n========== DATASET INFORMATION ==========")
print("Total images:", X.shape[0])
print("Features per image:", X.shape[1])
print("Feature matrix shape:", X.shape)
print("Number of classes:", len(np.unique(y)))

pd.Series(y).value_counts().sort_index().rename_axis(
    "class_name"
).reset_index(name="image_count").to_csv(
    OUTPUT_DIR / "task2_class_counts.csv",
    index=False
)


# ------------------------------------------------------------
# 3. CENTRE THE FEATURE MATRIX
# ------------------------------------------------------------

mean_vector = X.mean(axis=0)
X_centered = X - mean_vector

print("\n========== CENTRING ==========")
print("Mean vector shape:", mean_vector.shape)
print("Centred matrix shape:", X_centered.shape)


# ------------------------------------------------------------
# 4. PCA USING RANDOMISED SVD
# ------------------------------------------------------------

# Randomised PCA is more practical than full SVD for large
# image datasets. It estimates the leading components.

MAX_COMPONENTS = min(
    200,
    X_centered.shape[0] - 1,
    X_centered.shape[1]
)

if MAX_COMPONENTS < 1:
    raise ValueError("Not enough data to perform PCA.")

start_time = time.perf_counter()

pca = PCA(
    n_components=MAX_COMPONENTS,
    svd_solver="randomized",
    random_state=RANDOM_STATE
)

pca.fit(X_centered)

pca_time = time.perf_counter() - start_time

singular_values = pca.singular_values_
explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
components = pca.components_

print("\n========== PCA RESULTS ==========")
print("Components computed:", MAX_COMPONENTS)
print("PCA fitting time (seconds):", round(pca_time, 4))

print("\nLeading singular values:")
for i, value in enumerate(singular_values[:5], start=1):
    print(f"Component {i}: {value:.6f}")

print("\nLeading eigenvalues:")
for i, value in enumerate(explained_variance[:5], start=1):
    print(f"Component {i}: {value:.8f}")

print(
    "\nFirst component explained variance:",
    f"{explained_variance_ratio[0] * 100:.4f}%"
)


# ------------------------------------------------------------
# 5. SAVE SINGULAR VALUES AND EXPLAINED VARIANCE
# ------------------------------------------------------------

variance_df = pd.DataFrame({
    "component": np.arange(1, MAX_COMPONENTS + 1),
    "singular_value": singular_values,
    "eigenvalue": explained_variance,
    "explained_variance_ratio": explained_variance_ratio,
    "cumulative_variance_ratio": cumulative_variance
})

variance_df.to_csv(
    OUTPUT_DIR / "task2_variance_analysis.csv",
    index=False
)

print("\n========== CUMULATIVE EXPLAINED VARIANCE ==========")

for k in [5, 20, 50, 100, 200]:
    if k <= MAX_COMPONENTS:
        print(
            f"k={k}: "
            f"{cumulative_variance[k - 1] * 100:.4f}%"
        )


# ------------------------------------------------------------
# 6. PLOT SINGULAR VALUE SPECTRUM
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    np.arange(1, MAX_COMPONENTS + 1),
    singular_values
)

plt.title("Singular Value Spectrum")
plt.xlabel("Principal Component")
plt.ylabel("Singular Value")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "task2_singular_value_spectrum.png",
    dpi=200
)
plt.show()


# ------------------------------------------------------------
# 7. PLOT CUMULATIVE EXPLAINED VARIANCE
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    np.arange(1, MAX_COMPONENTS + 1),
    cumulative_variance * 100
)

plt.axhline(
    80,
    linestyle="--",
    label="80% variance"
)

plt.axhline(
    90,
    linestyle="--",
    label="90% variance"
)

plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Explained Variance (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "task2_cumulative_variance.png",
    dpi=200
)
plt.show()


# ------------------------------------------------------------
# 8. RECONSTRUCTION ERROR FOR DIFFERENT k VALUES
# ------------------------------------------------------------

component_counts = [5, 20, 50, 100, 200]
component_counts = [
    k for k in component_counts
    if k <= MAX_COMPONENTS
]

reconstruction_results = []

for k in component_counts:

    # Project the centred data onto the first k components.
    scores = X_centered @ components[:k].T

    # Reconstruct the original pixel features.
    reconstructed = scores @ components[:k] + mean_vector

    mse = np.mean((X - reconstructed) ** 2)
    variance_percent = cumulative_variance[k - 1] * 100
    feature_reduction = (1 - k / X.shape[1]) * 100

    reconstruction_results.append({
        "components_k": k,
        "original_features": X.shape[1],
        "retained_features": k,
        "variance_retained_percent": variance_percent,
        "reconstruction_mse": mse,
        "feature_reduction_percent": feature_reduction
    })

    print(
        f"k={k:3d} | "
        f"Variance={variance_percent:.4f}% | "
        f"MSE={mse:.8f} | "
        f"Feature reduction={feature_reduction:.2f}%"
    )

reconstruction_df = pd.DataFrame(reconstruction_results)

reconstruction_df.to_csv(
    OUTPUT_DIR / "task2_reconstruction_results.csv",
    index=False
)


# ------------------------------------------------------------
# 9. PLOT RECONSTRUCTION ERROR
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    reconstruction_df["components_k"],
    reconstruction_df["reconstruction_mse"],
    marker="o"
)

plt.title("PCA Reconstruction Error")
plt.xlabel("Number of Components (k)")
plt.ylabel("Mean Squared Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "task2_reconstruction_error.png",
    dpi=200
)
plt.show()


# ------------------------------------------------------------
# 10. VISUALISE LEADING PRINCIPAL COMPONENTS
# ------------------------------------------------------------

number_to_show = min(6, MAX_COMPONENTS)

fig, axes = plt.subplots(
    2, 3,
    figsize=(10, 6)
)

for i, ax in enumerate(axes.flat):

    if i < number_to_show:
        component_image = components[i].reshape(IMAGE_SIZE)

        ax.imshow(component_image, cmap="gray")
        ax.set_title(f"Principal Component {i + 1}")

    ax.axis("off")

plt.suptitle("Leading Principal Components")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "task2_principal_components.png",
    dpi=200
)
plt.show()


# ------------------------------------------------------------
# 11. HAND-CALCULATED 2 x 2 EIGENVALUE EXAMPLE
# ------------------------------------------------------------

A = np.array([
    [2.0, 1.0],
    [1.0, 2.0]
])

eigenvalues, eigenvectors = np.linalg.eigh(A)

print("\n========== 2 x 2 EIGENVALUE EXAMPLE ==========")
print("Matrix A:\n", A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Expected eigenvalues: 1 and 3
assert np.allclose(eigenvalues, [1, 3])


# ------------------------------------------------------------
# 12. SAVE SUMMARY
# ------------------------------------------------------------

summary = {
    "total_images": X.shape[0],
    "features_per_image": X.shape[1],
    "number_of_classes": len(np.unique(y)),
    "components_computed": MAX_COMPONENTS,
    "pca_fitting_time_seconds": pca_time,
    "first_singular_value": singular_values[0],
    "first_eigenvalue": explained_variance[0],
    "first_component_variance_percent":
        explained_variance_ratio[0] * 100
}

for k in component_counts:
    summary[f"variance_at_k{k}_percent"] = (
        cumulative_variance[k - 1] * 100
    )

pd.Series(summary).to_csv(
    OUTPUT_DIR / "task2_summary.csv",
    header=["value"]
)

print("\n========== TASK 2 COMPLETE ==========")
print("Output folder:", OUTPUT_DIR.resolve())
print("Saved results:")
print("- task2_class_counts.csv")
print("- task2_variance_analysis.csv")
print("- task2_reconstruction_results.csv")
print("- task2_summary.csv")
print("- task2_singular_value_spectrum.png")
print("- task2_cumulative_variance.png")
print("- task2_reconstruction_error.png")
print("- task2_principal_components.png")
