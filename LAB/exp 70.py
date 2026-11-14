from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load financial-style classification dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification before PCA
model1 = LogisticRegression(max_iter=5000)
model1.fit(X_train_scaled, y_train)
pred1 = model1.predict(X_test_scaled)

accuracy1 = accuracy_score(y_test, pred1)

# Apply PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Classification after PCA
model2 = LogisticRegression(max_iter=5000)
model2.fit(X_train_pca, y_train)
pred2 = model2.predict(X_test_pca)

accuracy2 = accuracy_score(y_test, pred2)

# Display results
print("Original Number of Features:", X.shape[1])
print("Features After PCA:", X_train_pca.shape[1])

print("Accuracy Before PCA:",
      round(accuracy1 * 100, 2), "%")

print("Accuracy After PCA:",
      round(accuracy2 * 100, 2), "%")