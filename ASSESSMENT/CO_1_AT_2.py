
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ==================================================
# TASK 1: BREAK DOWN THE AUDIT PROBLEM
# ==================================================
print("\nTASK 1: AUDIT PROBLEM")
print("1. Check feature compression using SVD.")
print("2. Analyse loss curvature and optimisation.")
print("3. Evaluate prediction probabilities and screening performance.")
print("4. Study bias, variance, and clinical error costs.")


# ==================================================
# TASK 2: SVD AND MATRIX COMPRESSION
# ==================================================
print("\nTASK 2: SVD AND COMPRESSION")

A = np.array([[3, 0],
              [4, 5]], dtype=float)

U, S, VT = np.linalg.svd(A)

A1 = S[0] * np.outer(U[:, 0], VT[0, :])
error = np.linalg.norm(A - A1, 'fro')
total_energy = np.sum(A ** 2)
retained = (S[0] ** 2 / total_energy) * 100

print("A transpose A:\n", A.T @ A)
print("Singular values:", S)
print("U:\n", U)
print("V transpose:\n", VT)
print("Rank-1 approximation:\n", A1)
print("Reconstruction error:", error)
print("Energy retained (%):", retained)
print("Energy discarded (%):", 100 - retained)


# ==================================================
# TASK 3: LOSS CURVATURE AND CONDITION NUMBER
# ==================================================
print("\nTASK 3: LOSS CURVATURE")

H = np.array([[4, 1],
              [1, 4]], dtype=float)

b = np.array([5, 5], dtype=float)

eigenvalues, eigenvectors = np.linalg.eigh(H)
condition_number = max(eigenvalues) / min(eigenvalues)
w_star = np.linalg.solve(H, b)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors (columns):\n", eigenvectors)
print("Condition number:", condition_number)
print("Positive definite:", np.all(eigenvalues > 0))
print("Optimal weights:", w_star)


# ==================================================
# TASK 4: INFORMATION THEORY
# ==================================================
print("\nTASK 4: INFORMATION THEORY")

p = np.array([0.5, 0.25, 0.125, 0.125])
q1 = np.array([0.25, 0.25, 0.25, 0.25])
q2 = np.array([0.4, 0.3, 0.2, 0.1])


def entropy(probabilities):
    return -np.sum(probabilities * np.log2(probabilities))


def cross_entropy(p_true, q_pred):
    return -np.sum(p_true * np.log2(q_pred))


def kl_divergence(p_true, q_pred):
    return np.sum(p_true * np.log2(p_true / q_pred))


print("H(p):", entropy(p))
print("H(q1):", entropy(q1))
print("H(q2):", entropy(q2))
print("Cross-entropy H(p,q1):", cross_entropy(p, q1))
print("Cross-entropy H(p,q2):", cross_entropy(p, q2))
print("KL divergence for q1:", kl_divergence(p, q1))
print("KL divergence for q2:", kl_divergence(p, q2))


# ==================================================
# TASK 5: BAYES' THEOREM
# ==================================================
print("\nTASK 5: BAYES' THEOREM")

prevalence = 0.10
sensitivity = 0.90
specificity = 0.95

p_positive = (sensitivity * prevalence +
              (1 - specificity) * (1 - prevalence))

ppv = sensitivity * prevalence / p_positive

p_negative = 1 - p_positive
p_disease_given_negative = (
    (1 - sensitivity) * prevalence / p_negative
)

print("Probability of positive result:", p_positive)
print("Probability of disease given positive:", ppv)
print("Probability of disease given negative:",
      p_disease_given_negative)


# ==================================================
# TASK 6: GRADIENT DESCENT FOR THREE ITERATIONS
# ==================================================
print("\nTASK 6: GRADIENT DESCENT")

eta = 0.1
w = np.array([0.0, 0.0])


def loss(w):
    return 0.5 * w.T @ H @ w - b.T @ w


def gradient(w):
    return H @ w - b


print("Initial weights:", w, "Loss:", loss(w))

for iteration in range(1, 4):
    w = w - eta * gradient(w)
    distance = np.linalg.norm(w - w_star)

    print(
        f"Iteration {iteration}: weights={w}, "
        f"loss={loss(w):.4f}, "
        f"distance from optimum={distance:.4f}"
    )


# ==================================================
# TASK 7: MOMENTUM AND LEARNING RATE
# ==================================================
print("\nTASK 7: MOMENTUM")

beta = 0.9
eta = 0.1
w = np.array([0.0, 0.0])
velocity = np.array([0.0, 0.0])

for iteration in range(1, 3):
    velocity = beta * velocity + gradient(w)
    w = w - eta * velocity

    print(
        f"Momentum iteration {iteration}: "
        f"weights={w}, loss={loss(w):.4f}"
    )

eta_max = 2 / max(eigenvalues)
print("Gradient descent learning-rate limit:", eta_max)
print("Learning rate 0.5 is stable:",
      0.5 < eta_max)

w = np.array([0.0, 0.0])
for iteration in range(1, 4):
    w = w - 0.5 * gradient(w)
    print(f"Learning rate 0.5, iteration {iteration}: {w}")


# ==================================================
# TASK 8: BIAS, VARIANCE AND EXPECTED SQUARED ERROR
# ==================================================
print("\nTASK 8: BIAS-VARIANCE DECOMPOSITION")

true_value = 10
noise_variance = 1

models = {
    "Model P": np.array([8, 9, 8, 9], dtype=float),
    "Model Q": np.array([12, 7, 11, 8], dtype=float)
}

for name, predictions in models.items():
    mean_prediction = np.mean(predictions)
    bias_squared = (mean_prediction - true_value) ** 2
    variance = np.mean((predictions - mean_prediction) ** 2)
    expected_error = bias_squared + variance + noise_variance

    print(name)
    print("  Mean prediction:", mean_prediction)
    print("  Bias squared:", bias_squared)
    print("  Variance:", variance)
    print("  Expected squared error:", expected_error)


# ==================================================
# TASK 9: CONFUSION MATRIX METRICS
# ==================================================
print("\nTASK 9: CONFUSION MATRIX")

TP, FN, FP, TN = 40, 10, 5, 45
total = TP + FN + FP + TN

accuracy = (TP + TN) / total
precision = TP / (TP + FP)
recall = TP / (TP + FN)
specificity_metric = TN / (TN + FP)
f1 = 2 * precision * recall / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity_metric)
print("F1 score:", f1)
print("False negatives:", FN)


# ==================================================
# TASK 10: INTEGRATED AUDIT CONCLUSION
# ==================================================
print("\nTASK 10: INTEGRATED AUDIT")
print("Compression retains 90% of matrix energy.")
print("The loss function is strictly convex.")
print("Gradient descent with learning rate 0.1 converges.")
print("The partly trained distribution has lower KL divergence.")
print("Screening recall is 80%, so missed cases need attention.")


# ==================================================
# TASK 11: LOW-PREVALENCE SCREENING
# ==================================================
print("\nTASK 11: LOW-PREVALENCE SCREENING")

prevalence = 0.02
sensitivity = 0.90
specificity = 0.95
population = 1000

diseased = population * prevalence
healthy = population - diseased

TP11 = diseased * sensitivity
FN11 = diseased - TP11
TN11 = healthy * specificity
FP11 = healthy - TN11

ppv11 = TP11 / (TP11 + FP11)
npv11 = TN11 / (TN11 + FN11)
missed_disease_probability = FN11 / (FN11 + TN11)

cost_fn = 10
cost_fp = 1
total_cost = FN11 * cost_fn + FP11 * cost_fp

required_specificity = (
    1 - sensitivity * prevalence / (1 - prevalence)
)

print("True positives:", TP11)
print("False negatives:", FN11)
print("True negatives:", TN11)
print("False positives:", FP11)
print("PPV:", ppv11)
print("NPV:", npv11)
print("Disease probability after negative:", missed_disease_probability)
print("Total cost per 1000:", total_cost)
print("Specificity required for PPV >= 50%:",
      required_specificity)

print("\nALL 11 TASKS COMPLETED.")
