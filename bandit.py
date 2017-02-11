import numpy as np
import matplotlib.pyplot as plt

num_actions = 10
epsilons = np.array([0., 0.1, 0.01])
num_iter = 10000

q_star_a = np.random.normal(size=[num_actions])
optimal_action = np.argmax(q_star_a)
optimal_actions = np.zeros([num_iter, len(epsilons)], dtype=np.int32)
R_t_a = np.zeros([num_iter, num_actions, len(epsilons)])
Q_t_a = np.zeros([num_iter, num_actions, len(epsilons)])
K_a = np.zeros([num_actions, len(epsilons)], dtype=np.int32)

# The first action is always assumed to be the action at index 0
# Absent prior knowledge, this is equivalent to a random choice

for t in range(1, num_iter):
    # Action Selection
    is_greedy = np.random.random() < (1 - epsilons)
    greedy_actions = np.argmax(Q_t_a[t - 1], axis=0)
    random_actions = np.random.randint(num_actions, size=len(epsilons))
    actions = np.where(is_greedy, greedy_actions, random_actions)
    optimal_actions[t, actions == np.argmax(q_star_a)] += 1

    # Value Update
    noise_term = np.random.normal(scale=1., size=len(epsilons))
    K_a[actions, np.arange(len(epsilons))] += 1
    Q_t_a[t] = Q_t_a[t - 1]
    R_t_a[t, actions, np.arange(len(epsilons))] = q_star_a[actions] + noise_term
    Q_t_a[t, K_a > 0] = np.sum(R_t_a, axis=0)[K_a > 0] / K_a[K_a > 0]

print("Q values", Q_t_a[-1])
print("Q star", q_star_a)

R_t = np.sum(R_t_a, axis=1)
plt.subplot(211)
plt.plot(R_t)
plt.subplot(212)
plt.plot(optimal_actions)
plt.show()