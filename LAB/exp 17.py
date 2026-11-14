import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X = np.arange(20).reshape(-1, 1)
y = np.array([0, 1] * 10)

model = DecisionTreeClassifier()
model.fit(X, y)

train = model.predict(X)
test = model.predict([[20], [21], [22]])

print("Training Accuracy:", accuracy_score(y, train))
print("Testing Predictions:", test)
print("Overfitting occurs when training performance is high but testing performance is poor.")
