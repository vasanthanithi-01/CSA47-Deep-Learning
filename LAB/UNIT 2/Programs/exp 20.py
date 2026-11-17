for lr in [0.01, 0.1, 0.5]:
    x = 0

    for i in range(50):
        gradient = 2 * (x - 5)
        x = x - lr * gradient

    loss = (x - 5) ** 2

    print("Learning Rate:", lr)
    print("Final Value:", round(x, 4))
    print("Final Loss:", round(loss, 6))
