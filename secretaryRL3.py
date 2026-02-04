# Tau-based Secretary Problem RL Simulation (Stabilized Version)

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Hyperparameters
# =========================
N = 1000
EPOCHS = 50000
lr = 0.01
sigma = 0.05
BLOCK = 1000

TARGET = 1 / np.e
ERROR_LEVELS = [1e-2, 1e-3, 1e-4, 1e-5]

# =========================
# Environment
# =========================

def generate_candidates(N):
    abilities = np.random.rand(N)
    order = np.random.permutation(N)
    return abilities[order]

def run_episode_tau(abilities, tau):
    cutoff = int(tau * N)
    best_so_far = -1
    chosen_index = None

    for i in range(N):
        is_best = abilities[i] > best_so_far
        if i < cutoff:
            if is_best:
                best_so_far = abilities[i]
        else:
            if is_best:
                chosen_index = i
                break

    if chosen_index is None:
        chosen_index = N - 1

    reward = int(abilities[chosen_index] == np.max(abilities))
    return chosen_index, reward

# =========================
# Training (Stabilized)
# =========================

theta = 0.0  # unconstrained parameter
baseline = 0.0

taus = []
chosen_indices = []
rewards = []

tau_error_hit_epoch = {e: None for e in ERROR_LEVELS}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for epoch in range(EPOCHS):
    mu = sigmoid(theta)
    tau = np.clip(np.random.normal(mu, sigma), 0.01, 0.99)

    abilities = generate_candidates(N)
    chosen_idx, reward = run_episode_tau(abilities, tau)

    # baseline update
    baseline = 0.99 * baseline + 0.01 * reward
    advantage = reward - baseline

    # REINFORCE gradient (chain rule applied)
    grad_logp = (tau - mu) / (sigma ** 2)
    grad_mu = grad_logp * mu * (1 - mu)
    theta += lr * advantage * grad_mu

    taus.append(mu)
    chosen_indices.append(chosen_idx)
    rewards.append(reward)

    for e in ERROR_LEVELS:
        if tau_error_hit_epoch[e] is None:
            if abs(mu - TARGET) / TARGET < e:
                tau_error_hit_epoch[e] = epoch

# =========================
# Plot 1: chosen index vs epoch
# =========================

plt.figure(figsize=(10, 4))
plt.plot(range(EPOCHS), chosen_indices, alpha=0.4)
plt.axhline(TARGET * N, color='red', linestyle='--', label='N / e')
plt.xlabel("Epoch")
plt.ylabel("Chosen candidate index")
plt.title("Chosen Candidate Index over Epochs")
plt.legend()
plt.show()

# =========================
# Plot 2: tau vs epoch
# =========================

plt.figure(figsize=(10, 4))
plt.plot(range(EPOCHS), taus, label='Learned tau')
plt.axhline(TARGET, color='red', linestyle='--', label='1 / e')
plt.xlabel("Epoch")
plt.ylabel("Tau")
plt.title("Tau Convergence over Epochs (Stabilized)")
plt.legend()
plt.show()

# =========================
# Tau convergence report
# =========================

print("=== Tau Convergence Epochs ===")
for e, ep in tau_error_hit_epoch.items():
    print(f"Relative error < {e*100:.4f}% : epoch {ep}")

# =========================
# Post-training block analysis
# =========================

print("\n=== Success count per 100-epoch block (from end) ===")
blocks = EPOCHS // BLOCK
for i in range(blocks):
    start = EPOCHS - (i + 1) * BLOCK
    end = EPOCHS - i * BLOCK
    success_count = sum(rewards[start:end])
    print(f"Epoch {start} ~ {end} : {success_count} successes")
