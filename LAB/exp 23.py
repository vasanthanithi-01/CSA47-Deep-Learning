import numpy as np

Q = np.zeros((2, 2))

alpha = 0.1
gamma = 0.9

rewards = [[10, -5], [-5, 10]]

for episode in range(100):
    state = episode % 2

    action = np.argmax(Q[state])

    reward = rewards[state][action]

    next_state = 1 - state

    Q[state, action] += alpha * (
        reward + gamma * np.max(Q[next_state])
        - Q[state, action]
    )

print("Q Table:")
print(Q)

print("Best Actions:", np.argmax(Q, axis=1))
