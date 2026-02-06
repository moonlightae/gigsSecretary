# Tau-based Secretary Problem RL Simulation
# Outputs saved to datasRL/data as txt and png

import numpy as np
import matplotlib.pyplot as plt
import os

# Directory setup
def make_unique_dir(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return base_path

    idx = 1
    while True:
        new_path = f"{base_path}_{idx}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            return new_path
        idx += 1


BASE_DIR = "datasRL"
BASE_SAVE_DIR = os.path.join(BASE_DIR, "data")

SAVE_DIR = make_unique_dir(BASE_SAVE_DIR)
print(f"Saving results to: {SAVE_DIR}")
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters
N = 10000
EPOCHS = 100000
lr = 0.01
sigma = 0.05
BLOCK = 1000

TARGET = 1 / np.e
ERROR_LEVELS = [1e-2, 1e-3, 1e-4, 1e-5]

# Environment
def generate_candidates(N):
    abilities = np.random.rand(N)
    order = np.random.permutation(N)
    return abilities[order]

def run_episode_tau(abilities, tau):
    cutoff = int(tau * len(abilities))
    best_so_far = -1
    chosen_index = None

    for i in range(len(abilities)):
        is_best = abilities[i] > best_so_far
        if i < cutoff:
            if is_best:
                best_so_far = abilities[i]
            continue
        else:
            if is_best:
                chosen_index = i
                break

    if chosen_index is None:
        chosen_index = len(abilities) - 1

    reward = int(abilities[chosen_index] == np.max(abilities))
    return chosen_index, reward

# Training
theta = 0.0
baseline = 0.0

chosen_indices = []
taus = []
rewards = []

error_hit_epoch = {e: None for e in ERROR_LEVELS}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for epoch in range(EPOCHS):
    mu = sigmoid(theta)
    tau = np.clip(np.random.normal(mu, sigma), 0.01, 0.99)

    abilities = generate_candidates(N)
    chosen_idx, reward = run_episode_tau(abilities, tau)

    baseline = 0.99 * baseline + 0.01 * reward
    advantage = reward - baseline

    grad_logp = (tau - mu) / (sigma ** 2)
    grad_mu = grad_logp * mu * (1 - mu)
    theta += lr * advantage * grad_mu

    taus.append(mu)
    chosen_indices.append(chosen_idx)
    rewards.append(reward)

    for e in ERROR_LEVELS:
        if error_hit_epoch[e] is None:
            if abs(mu - TARGET) / TARGET < e:
                error_hit_epoch[e] = epoch

# Save plots
plt.figure(figsize=(10, 5))
plt.plot(range(EPOCHS), chosen_indices, alpha=0.5)
plt.axhline(TARGET * N, color='red', linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Chosen candidate index")
plt.title("Chosen Candidate Index over Epochs")
png_path = os.path.join(SAVE_DIR, "chosen_candidate_index.png")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(range(EPOCHS), taus, label="Learned tau")
plt.axhline(TARGET, color='red', linestyle='--', label="1/e")
plt.xlabel("Epoch")
plt.ylabel("Tau")
plt.title("Tau Convergence over Epochs")
plt.legend()
png_path = os.path.join(SAVE_DIR, "tau_convergence.png")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()

blocks = EPOCHS // BLOCK
success_rates = []
x_labels = []
for i in range(blocks):
    start = i * BLOCK
    end = (i + 1) * BLOCK
    success_count = sum(rewards[start:end])
    success_rate = success_count / BLOCK

    success_rates.append(success_rate)
    x_labels.append(f"{start}-{end}")
plt.figure(figsize=(10, 5))
plt.plot(range(blocks), success_rates, marker='o', label=f"Success rate per {BLOCK} epochs")
plt.axhline(y=1 / np.e, linestyle='--', label="y = 1/e")
plt.xlabel("100-epoch blocks (from end)")
plt.ylabel("Success rate")
plt.title(f"Success rate per {BLOCK}-epoch block")
plt.legend()
plt.grid(True)
png_path = os.path.join(SAVE_DIR, "success_rate_blocks.png")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()

# Save logs
txt_path = os.path.join(SAVE_DIR, "logs.txt")

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"epoch: {EPOCHS}\nN: {N}\n\n")
    f.write("Tau convergence epochs (relative error 기준)\n")
    for e, ep in error_hit_epoch.items():
        f.write(f"Error < {e*100:.4f}% : epoch {ep}\n")
    f.write("\n")

print("All outputs saved to datasRL/data")
