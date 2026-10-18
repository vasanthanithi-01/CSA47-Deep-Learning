
# ============================================================
# CSA47 - DEEP LEARNING
# ASSESSMENT 1 - TASK 1 (4 MARKS)
# Problem Structure and Data Representation
# Dataset: PlantVillage - Tomato Healthy, Early Blight,
#          and Late Blight
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\VASANTH\Downloads\raw\color"
)

OUTPUT_DIR = Path("task1_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
RANDOM_STATE = 42

CLASS_NAMES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ------------------------------------------------------------
# 2. VERIFY AND INSPECT THE DATASET
# ------------------------------------------------------------

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR}\n"
        "Check the dataset extraction path."
    )

records = []

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

    for path in image_paths:
        records.append({
            "path": str(path),
            "class_name": class_name
        })

df = pd.DataFrame(records)

if df.empty:
    raise ValueError("No valid images were found.")

print("\n========== DATASET SUMMARY ==========")
print("Dataset path:", DATA_DIR)
print("Total images:", len(df))
print("Number of classes:", df["class_name"].nunique())
print("\nImages per class:")
print(df["class_name"].value_counts().sort_index())

class_counts = (
    df["class_name"]
    .value_counts()
    .sort_index()
    .rename_axis("class_name")
    .reset_index(name="image_count")
)

class_counts.to_csv(
    OUTPUT_DIR / "task1_class_counts.csv",
    index=False
)


# ------------------------------------------------------------
# 3. VISUALISE CLASS DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    class_counts["class_name"],
    class_counts["image_count"]
)

plt.title("PlantVillage Tomato Dataset: Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "task1_class_distribution.png",
    dpi=200
)
plt.show()


# ------------------------------------------------------------
# 4. INSPECT AN ORIGINAL IMAGE
# ------------------------------------------------------------

sample_path = Path(df.iloc[0]["path"])
sample_label = df.iloc[0]["class_name"]

with Image.open(sample_path) as image:
    original = image.copy()
    original_width, original_height = image.size
    original_mode = image.mode

print("\n========== ORIGINAL IMAGE ==========")
print("Sample image:", sample_path.name)
print("Class label:", sample_label)
print("Original dimensions:", (original_width, original_height))
print("Original image mode:", original_mode)

plt.figure(figsize=(5, 5))
plt.imshow(original)
plt.title(f"Original Image\n{sample_label}")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "task1_original_image.png",
    dpi=200
)
plt.show()


# ------------------------------------------------------------
# 5. PREPROCESSING: GRAYSCALE, RESIZE AND NORMALISE
# ------------------------------------------------------------

def preprocess_image(image_path):
    """
    Convert image to grayscale,
    resize to 64 x 64,
    and normalise pixel values to [0, 1].
    """

    with Image.open(image_path) as image:
        image = image.convert("L")

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS
        )

        pixels = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

    return pixels


processed_image = preprocess_image(sample_path)

print("\n========== PREPROCESSING ==========")
print("Processed image shape:", processed_image.shape)
print("Number of pixel features:", processed_image.size)
print("Minimum pixel value:", processed_image.min())
print("Maximum pixel value:", processed_image.max())

plt.figure(figsize=(5, 5))
plt.imshow(
    processed_image,
    cmap="gray",
    vmin=0,
    vmax=1
)
plt.title("Preprocessed Grayscale Image (64 x 64)")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "task1_preprocessed_image.png",
    dpi=200
)
plt.show()


# ------------------------------------------------------------
# 6. ENCODE LABELS AND SPLIT THE DATA
# ------------------------------------------------------------

class_to_index = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

df["label"] = df["class_name"].map(class_to_index)

# Reserve 20% of the dataset for testing.
train_val_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

# Split the remaining 80% into training and validation.
# Final proportions: 64% train, 16% validation, 20% test.
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=train_val_df["label"]
)

print("\n========== DATA SPLIT ==========")
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))

print("\nTraining class counts:")
print(train_df["class_name"].value_counts().sort_index())

train_df.to_csv(
    OUTPUT_DIR / "task1_train_split.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "task1_validation_split.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "task1_test_split.csv",
    index=False
)


# ------------------------------------------------------------
# 7. CREATE A MINI-BATCH WITHOUT PYTORCH
# ------------------------------------------------------------

batch_size = min(BATCH_SIZE, len(train_df))

batch_images = []
batch_labels = []

for _, row in train_df.head(batch_size).iterrows():

    pixels = preprocess_image(row["path"])

    # Add channel dimension: (1, 64, 64)
    image_array = pixels[np.newaxis, :, :]

    batch_images.append(image_array)
    batch_labels.append(int(row["label"]))

images = np.asarray(
    batch_images,
    dtype=np.float32
)

labels = np.asarray(
    batch_labels,
    dtype=np.int64
)

print("\n========== BATCH SHAPES ==========")
print("One image shape:", images[0].shape)
print("Mini-batch shape:", images.shape)
print("Batch labels shape:", labels.shape)
print("Number of output classes:", len(CLASS_NAMES))

assert images.shape[1:] == (1, 64, 64)
assert images.shape[0] <= BATCH_SIZE
assert labels.shape[0] == images.shape[0]

print("Batch shape checks: PASSED")


# ------------------------------------------------------------
# 8. SCALAR, MATRIX, VECTOR AND TENSOR REPRESENTATIONS
# ------------------------------------------------------------

# Scalar: one pixel value
scalar_pixel = float(processed_image[0, 0])

# Matrix: one grayscale image
image_matrix = processed_image

# Vector: flattened image
feature_vector = image_matrix.flatten()

# Dataset feature matrix dimensions
dataset_matrix_shape = (
    len(df),
    IMAGE_SIZE[0] * IMAGE_SIZE[1]
)

print("\n========== REPRESENTATIONS ==========")
print("Scalar pixel value:", scalar_pixel)
print("Image matrix shape:", image_matrix.shape)
print("Flattened vector shape:", feature_vector.shape)
print("Dataset feature matrix shape:", dataset_matrix_shape)
print("Batch tensor shape:", images.shape)


# ------------------------------------------------------------
# 9. L1, L2 AND L-INFINITY NORMS
# ------------------------------------------------------------

# Illustrative vector used for hand calculations
x = np.array([3, -4, 12], dtype=float)

l1_norm = np.linalg.norm(x, ord=1)
l2_norm = np.linalg.norm(x, ord=2)
linf_norm = np.linalg.norm(x, ord=np.inf)

print("\n========== ILLUSTRATIVE VECTOR NORMS ==========")
print("Vector:", x)
print("L1 norm:", l1_norm)
print("L2 norm:", l2_norm)
print("L-infinity norm:", linf_norm)

assert np.isclose(l1_norm, 19)
assert np.isclose(l2_norm, 13)
assert np.isclose(linf_norm, 12)

# Norms of the actual image feature vector
print("\n========== ACTUAL IMAGE VECTOR NORMS ==========")
print("L1 norm:", np.linalg.norm(feature_vector, ord=1))
print("L2 norm:", np.linalg.norm(feature_vector, ord=2))
print("L-infinity norm:", np.linalg.norm(feature_vector, ord=np.inf))


# ------------------------------------------------------------
# 10. NORMALISATION EXAMPLE
# ------------------------------------------------------------

pixel_value = 200
normalised_value = pixel_value / 255.0

print("\n========== NORMALISATION EXAMPLE ==========")
print("Original pixel value:", pixel_value)
print("Normalised pixel value:", normalised_value)


# ------------------------------------------------------------
# 11. SAVE TASK 1 SUMMARY
# ------------------------------------------------------------

summary = {
    "dataset_path": str(DATA_DIR.resolve()),
    "total_images": len(df),
    "number_of_classes": df["class_name"].nunique(),
    "image_height": IMAGE_SIZE[0],
    "image_width": IMAGE_SIZE[1],
    "features_per_image": IMAGE_SIZE[0] * IMAGE_SIZE[1],
    "batch_size_requested": BATCH_SIZE,
    "actual_batch_size": len(images),
    "train_images": len(train_df),
    "validation_images": len(val_df),
    "test_images": len(test_df),
    "original_image_width": original_width,
    "original_image_height": original_height,
    "original_image_mode": original_mode,
    "illustrative_L1": l1_norm,
    "illustrative_L2": l2_norm,
    "illustrative_Linf": linf_norm
}

pd.Series(summary).to_csv(
    OUTPUT_DIR / "task1_summary.csv",
    header=["value"]
)

print("\n========== TASK 1 COMPLETE ==========")
print("Output folder:", OUTPUT_DIR.resolve())
print("Saved files:")
print("- task1_class_counts.csv")
print("- task1_class_distribution.png")
print("- task1_original_image.png")
print("- task1_preprocessed_image.png")
print("- task1_train_split.csv")
print("- task1_validation_split.csv")
print("- task1_test_split.csv")
print("- task1_summary.csv")
