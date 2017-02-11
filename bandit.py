import numpy as np
import matplotlib.pyplot as plt

num_actions = 10
epsilon = 0.
num_iter = 1000

q_star_a = np.random.normal(size=[num_actions])
R_t_a = np.zeros([num_iter, num_actions])
Q_t_a = np.zeros([num_iter, num_actions])
K_a = np.zeros(num_actions, dtype=np.int32)

# First action selected at random
a = np.random.randint(num_actions)
Q_t_a[0, a] = q_star_a[a]

for t in range(1, num_iter):
    # Action Selection
    is_greedy = np.random.random() < (1 - epsilon)
    if is_greedy:
        a = np.argmax(Q_t_a[t - 1])
    else:
        a = np.random.randint(num_actions)

    # Value Update
    noise_term = np.random.normal(scale=0.1)
    K_a[a] += 1
    Q_t_a[t] = Q_t_a[t - 1]
    R_t_a[t, a] = q_star_a[a] + noise_term
    Q_t_a[t, a] = np.sum(R_t_a[:, a]) / K_a[a]

print("Q values", Q_t_a[-1])
print("Q star", q_star_a)

R_t = np.sum(R_t_a, axis=1)
plt.plot(R_t)
plt.show()