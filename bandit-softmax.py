import numpy as np
import matplotlib.pyplot as plt

# Experiment Setup
np.random.seed(19680801)
num_actions = 10
num_trials = 2000
num_iter = 10000
tau_values = [np.exp(0), np.exp(-2), np.exp(-4)]
taus = np.array(tau_values * num_trials)

q_star_a = np.repeat(np.random.normal(size=[num_actions, num_trials]), len(tau_values), axis=1)
optimal_action = np.argmax(q_star_a, axis=0)
optimal_actions = np.zeros([num_iter, len(taus)], dtype=np.int32)
R_t_a = np.zeros([num_iter, num_actions, len(taus)])
Q_a = np.zeros([num_actions, len(taus)])
K_a = np.zeros([num_actions, len(taus)], dtype=np.int32)

# The first action is always assumed to be the action at index 0
# Absent prior knowledge, this is equivalent to a random choice

for t in range(1, num_iter):
    # Action Selection
    energy = np.exp(Q_a / taus)
    sum_energy = np.sum(energy, axis=0, keepdims=True)
    softmax = energy / sum_energy
    cdf = np.cumsum(softmax, axis=0)
    actions = np.argmin(np.random.random_sample(len(taus)) > cdf, axis=0)
    action_idx = actions, np.arange(len(taus))
    optimal_actions[t, actions == optimal_action] += 1

    # Sample Environment
    noise_term = np.random.normal(scale=1., size=len(taus))
    R_t_a[t][action_idx] = q_star_a[action_idx] + noise_term

    # Update Estimate
    K_a[action_idx] += 1
    step_size = 1 / K_a[action_idx]
    target = R_t_a[t][action_idx]
    old_estimate = Q_a[action_idx]

    Q_a[action_idx] = old_estimate + step_size * (target - old_estimate)

R_t = np.mean(np.sum(R_t_a, axis=1).reshape([num_iter, num_trials, -1]), axis=1)
A_t = np.mean(optimal_actions.reshape([num_iter, num_trials, -1]), axis=1)
plt.subplot(211)
plt.plot(R_t[:, 0], 'r')
plt.plot(R_t[:, 1], 'g')
plt.plot(R_t[:, 2], 'b')
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.subplot(212)
plt.plot(A_t[:, 0], 'r')
plt.plot(A_t[:, 1], 'g')
plt.plot(A_t[:, 2], 'b')
plt.xlabel('Steps')
plt.ylabel('Optimal action')
plt.show()
