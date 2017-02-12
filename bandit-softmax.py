import numpy as np
import matplotlib.pyplot as plt

# Experiment Setup
np.random.seed(19680801)
num_actions = 10
num_trials = 2000
num_iter = 10000
tau_values = [np.exp(0), np.exp(-1), np.exp(-2), np.exp(-3)]
taus = np.array(tau_values * num_trials)
num_samples = len(taus)

q_star_a = np.repeat(np.random.normal(size=[num_actions, num_trials]), len(tau_values), axis=1)
optimal_action = np.argmax(q_star_a, axis=0)
optimal_actions = np.zeros([num_iter, num_samples], dtype=np.int32)
R_t_a = np.zeros([num_iter, num_actions, num_samples])
Q_a = np.zeros([num_actions, num_samples])
K_a = np.zeros([num_actions, num_samples], dtype=np.int32)

# The first action is always assumed to be the action at index 0
# Absent prior knowledge, this is equivalent to a random choice

for t in range(1, num_iter):
    # Action Selection
    energy = np.exp(Q_a / taus)
    sum_energy = np.sum(energy, axis=0, keepdims=True)
    softmax = energy / sum_energy
    cdf = np.cumsum(softmax, axis=0)
    actions = np.argmin(np.random.random_sample(num_samples) > cdf, axis=0)
    action_idx = actions, np.arange(num_samples)
    optimal_actions[t, actions == optimal_action] += 1

    # Sample Environment
    noise_term = np.random.normal(scale=1., size=num_samples)
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
for eps in range(len(tau_values)):
    plt.plot(R_t[:, eps], label="tau = %f" % tau_values[eps])
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
plt.subplot(212)
plt.plot(np.mean(optimal_actions.reshape([num_iter, num_trials, -1]), axis=1))
plt.xlabel('Steps')
plt.ylabel('Optimal action')
plt.show()
