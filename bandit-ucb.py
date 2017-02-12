import numpy as np
import matplotlib.pyplot as plt

# Experiment Setup
np.random.seed(19680801)
num_actions = 10
num_trials = 2000
num_iter = 1000
epsilon_values = [1e-1, 0]
epsilons = np.array(epsilon_values * num_trials)
num_samples = len(epsilons)

q_star_a = np.repeat(np.random.normal(size=[num_actions, num_trials]), len(epsilon_values), axis=1)
optimal_action = np.argmax(q_star_a, axis=0)
optimal_actions = np.zeros([num_iter, num_samples], dtype=np.int32)
R_t_a = np.zeros([num_iter, num_actions, num_samples])
Q_a = np.full([num_actions, num_samples], 5.)
Q_a[:, ::2] = 0.
K_a = np.zeros([num_actions, num_samples], dtype=np.int32)

# The first action is always assumed to be the action at index 0
# Absent prior knowledge, this is equivalent to a random choice

for t in range(1, num_iter):
    # Select Action
    is_greedy = np.random.random(num_samples) < (1 - epsilons)
    greedy_actions = np.argmax(Q_a, axis=0)
    random_actions = np.random.randint(num_actions, size=num_samples)
    actions = np.where(is_greedy, greedy_actions, random_actions)
    action_idx = actions, np.arange(num_samples)
    optimal_actions[t, actions == optimal_action] += 1

    # Sample Environment
    noise_term = np.random.normal(scale=1., size=num_samples)
    R_t_a[t][action_idx] = q_star_a[action_idx] + noise_term

    # Update Estimate
    K_a[action_idx] += 1
    # Exponential Average - Constant Step Size
    step_size = np.full([num_samples], 0.1)
    target = R_t_a[t][action_idx]
    old_estimate = Q_a[action_idx]

    Q_a[action_idx] = old_estimate + step_size * (target - old_estimate)


R_t = np.mean(np.sum(R_t_a, axis=1).reshape([num_iter, num_trials, -1]), axis=1)
plt.subplot(211)
plt.plot(R_t[:, 0], label='realistic, e = 0.1')
plt.plot(R_t[:, 1], label='optimistic, e = 0')
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
plt.subplot(212)
plt.plot(np.mean(optimal_actions.reshape([num_iter, num_trials, -1]), axis=1))
plt.xlabel('Steps')
plt.ylabel('Optimal action')
plt.show()
