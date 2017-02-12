import numpy as np
import matplotlib.pyplot as plt

# Experiment Setup
np.random.seed(19680801)
num_actions = 10
num_trials = 2000
num_iter = 10000
epsilon_values = [0., 0.1, 0.01]
epsilons = np.array(epsilon_values * num_trials)

q_star_a = np.repeat(np.random.normal(size=[num_actions, num_trials]), len(epsilon_values), axis=1)
optimal_action = np.argmax(q_star_a, axis=0)
optimal_actions = np.zeros([num_iter, len(epsilons)], dtype=np.int32)
R_t_a = np.zeros([num_iter, num_actions, len(epsilons)])
Q_a = np.zeros([num_actions, len(epsilons)])
K_a = np.zeros([num_actions, len(epsilons)], dtype=np.int32)

# The first action is always assumed to be the action at index 0
# Absent prior knowledge, this is equivalent to a random choice

for t in range(1, num_iter):
    # Select Action
    is_greedy = np.random.random(len(epsilons)) < (1 - epsilons)
    greedy_actions = np.argmax(Q_a, axis=0)
    random_actions = np.random.randint(num_actions, size=len(epsilons))
    actions = np.where(is_greedy, greedy_actions, random_actions)
    action_idx = actions, np.arange(len(epsilons))
    optimal_actions[t, actions == optimal_action] += 1

    # Sample Environment
    noise_term = np.random.normal(scale=1., size=len(epsilons))
    R_t_a[t][action_idx] = q_star_a[action_idx] + noise_term

    # Update Estimate
    K_a[action_idx] += 1
    step_size = 1 / K_a[action_idx]
    target = R_t_a[t][action_idx]
    old_estimate = Q_a[action_idx]

    Q_a[action_idx] = old_estimate + step_size * (target - old_estimate)


R_t = np.mean(np.sum(R_t_a, axis=1).reshape([num_iter, num_trials, -1]), axis=1)
plt.subplot(211)
plt.plot(R_t)
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.subplot(212)
plt.plot(np.mean(optimal_actions.reshape([num_iter, num_trials, -1]), axis=1))
plt.xlabel('Steps')
plt.ylabel('Optimal action')
plt.show()
