import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(30, 5, 1000)

print("Mean:", round(np.mean(data), 2))
print("Standard Deviation:", round(np.std(data), 2))

plt.hist(data, bins=20, density=True)
plt.xlabel("Temperature")
plt.ylabel("Probability Density")
plt.title("Normal Distribution")
plt.show()
