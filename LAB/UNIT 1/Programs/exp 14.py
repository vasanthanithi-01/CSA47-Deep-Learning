x = 0
lr = 0.1

for i in range(50):
    gradient = 2 * (x - 3)
    x = x - lr * gradient

print("Minimum point:", round(x, 4))
print("Minimum cost:", round((x-3)**2, 6))
