import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Generate two voice-like signals
time = np.linspace(0, 10, 1000)

voice1 = np.sin(2 * np.pi * 2 * time)
voice2 = np.sin(2 * np.pi * 5 * time) + 0.5 * np.sin(2 * np.pi * 8 * time)

# Combine the original signals
S = np.c_[voice1, voice2]

# Mix the signals using two microphones
A = np.array([[1, 0.5], [0.5, 1]])
mixed = S @ A.T

# Apply ICA for blind source separation
ica = FastICA(n_components=2, random_state=42)
separated = ica.fit_transform(mixed)

# Calculate correlation with original signals
corr1 = abs(np.corrcoef(voice1, separated[:, 0])[0, 1])
corr2 = abs(np.corrcoef(voice2, separated[:, 1])[0, 1])

# Display results
print("Signal Separation Completed")
print("Correlation of Voice 1:", round(corr1, 3))
print("Correlation of Voice 2:", round(corr2, 3))

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(time, S)
plt.title("Original Voice Signals")

plt.subplot(3, 1, 2)
plt.plot(time, mixed)
plt.title("Mixed Microphone Signals")

plt.subplot(3, 1, 3)
plt.plot(time, separated)
plt.title("Separated Signals Using ICA")

plt.tight_layout()
plt.show()