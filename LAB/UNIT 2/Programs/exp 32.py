import numpy as np

actual = np.array([1, 0, 1, 0])

predicted = np.array([0.9, 0.2, 0.8, 0.1])

loss = -np.mean(
    actual * np.log(predicted) +
    (1 - actual) * np.log(1 - predicted)
)

print("Binary Cross Entropy Loss:",
      round(loss, 4))
