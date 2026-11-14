import numpy as np

P = np.array([0.4, 0.6])
Q = np.array([0.5, 0.5])

kl = np.sum(P * np.log(P / Q))

print("KL Divergence:", round(kl, 4))
