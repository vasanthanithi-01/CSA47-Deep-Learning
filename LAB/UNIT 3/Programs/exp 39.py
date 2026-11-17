import numpy as np
from scipy.signal import convolve2d

image = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

kernel = np.array([[1, 0],
                   [0, -1]])

feature_map = convolve2d(image, kernel, mode='valid')

print("Original Image:")
print(image)

print("\nFeature Map after Convolution:")
print(feature_map)

print("\nMax Pooling:")
print(np.max(feature_map))

print("\nAverage Pooling:")
print(np.mean(feature_map))