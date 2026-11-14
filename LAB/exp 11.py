import math

p1 = 0.6
p2 = 0.4

entropy = -(p1 * math.log2(p1) +
            p2 * math.log2(p2))

print("Entropy:", round(entropy, 4))
