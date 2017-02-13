import numpy as np
import matplotlib.pyplot as plt

# Experiment Setup
np.random.seed(19680801)
num_actions = 10
num_trials = 2000
num_iter = 1000
# Every other alpha has baseline
alpha_values = [0.1, 0.1, 0.4, 0.4]
alphas = np.array(alpha_values * num_trials)
num_samples = len(alphas)

q_star_a = np.repeat(np.random.normal(loc=4., size=[num_actions, num_trials]), len(alpha_values), axis=1)
optimal_action = np.argmax(q_star_a, axis=0)
optimal_actions = np.zeros([num_iter, num_samples], dtype=np.int32)
R_t_a = np.zeros([num_iter, num_actions, num_samples])
baseline = np.zeros(num_samples)
H_a = np.zeros([num_actions, num_samples])
K_a = np.zeros([num_actions, num_samples], dtype=np.int32)

# The first action is always assumed to be the action at index 0
# Absent prior knowledge, this is equivalent to a random choice

for t in range(1, num_iter):
    # Action Selection
    energy = np.exp(H_a)
    sum_energy = np.sum(energy, axis=0, keepdims=True)
    pi_a = energy / sum_energy
    cdf = np.cumsum(pi_a, axis=0)
    actions = np.argmin(np.random.random_sample(num_samples) > cdf, axis=0)
    action_idx = actions, np.arange(num_samples)
    optimal_actions[t, actions == optimal_action] += 1

    # Sample Environment
    noise_term = np.random.normal(scale=1., size=num_samples)
    R_t_a[t][action_idx] = q_star_a[action_idx] + noise_term

    K_a[action_idx] += 1

    # Update baseline incrementally
    # Define as average reward (across all actions)
    # Doubling up on experiments so only run baseline every other trial
    step_size = 1 / t
    reward_target = np.sum(R_t_a[t], axis=0)
    baseline[::2] += step_size * (reward_target[::2] - baseline[::2])

    # Apply gradient update
    one_hot_action = np.zeros([num_actions, num_samples])
    one_hot_action[action_idx] = 1
    gradient = (reward_target - baseline) * (one_hot_action - pi_a)
    H_a += alphas * gradient

R_t = np.mean(np.sum(R_t_a, axis=1).reshape([num_iter, num_trials, -1]), axis=1)
A_t = np.mean(optimal_actions.reshape([num_iter, num_trials, -1]), axis=1)
plt.subplot(211)
for idx, eps in enumerate(alpha_values):
    plt.plot(
        R_t[:, idx],
        label="a = %.1f %s baseline " % (
            eps,
            "with" if idx % 2 == 0 else "without"
        )
    )
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
plt.subplot(212)
plt.plot(np.mean(optimal_actions.reshape([num_iter, num_trials, -1]), axis=1))
plt.xlabel('Steps')
plt.ylabel('Optimal action')
plt.show()
