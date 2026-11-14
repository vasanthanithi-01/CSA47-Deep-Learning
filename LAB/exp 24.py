import numpy as np
from sklearn.linear_model import Perceptron

X = np.array([
    [20000, 600],
    [30000, 650],
    [40000, 700],
    [15000, 450]
])

y = np.array([0, 1, 1, 0])

model = Perceptron()
model.fit(X, y)

prediction = model.predict([[35000, 680]])

print("Loan Eligibility:", prediction[0])
