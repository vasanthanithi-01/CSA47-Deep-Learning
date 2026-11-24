import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Generate two independent signals
time = np.linspace(0, 8, 1000)

signal1 = np.sin(2 * time)
signal2 = np.sign(np.sin(3 * time))

# Combine the signals
S = np.c_[signal1, signal2]
S += 0.2 * np.random.normal(size=S.shape)

# Mix the signals
A = np.array([[1, 1], [0.5, 2]])
mixed_signals = S @ A.T

# Apply ICA to separate the mixed signals
ica = FastICA(n_components=2, random_state=42)
separated_signals = ica.fit_transform(mixed_signals)

# Display the results
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(time, S)
plt.title("Original Independent Signals")

plt.subplot(3, 1, 2)
plt.plot(time, mixed_signals)
plt.title("Mixed Signals")

plt.subplot(3, 1, 3)
plt.plot(time, separated_signals)
plt.title("Separated Signals Using ICA")

plt.tight_layout()
plt.show()

print("ICA Signal Separation Completed")
print("Separated Signal Shape:", separated_signals.shape)