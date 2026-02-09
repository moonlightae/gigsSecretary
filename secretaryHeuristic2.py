import random
import math
import matplotlib.pyplot as plt

# =========================
# 1회 시행
# =========================
def secretary_trial(N, alpha):
    k = int(alpha * N)
    perm = random.sample(range(N), N)

    best = N
    for i in range(k):
        best = min(best, perm[i])

    for i in range(k, N):
        if perm[i] < best:
            return 1 if perm[i] == 0 else 0
    return 0


# =========================
# 휴리스틱 학습 (epoch 기반)
# =========================
def heuristic_learning(
    N=500,
    epochs=30,
    trials_per_epoch=200
):
    alphas = [i / 100 for i in range(20, 50)]  # 0.20 ~ 0.49
    stats = {a: {"success": 0, "trials": 0} for a in alphas}
    best_alpha_history = []

    for epoch in range(epochs):
        for alpha in alphas:
            for _ in range(trials_per_epoch):
                stats[alpha]["success"] += secretary_trial(N, alpha)
                stats[alpha]["trials"] += 1

        best_alpha = max(
            alphas,
            key=lambda a: stats[a]["success"] / stats[a]["trials"]
        )
        best_alpha_history.append(best_alpha)

        print(f"Epoch {epoch+1:02d} | Best α = {best_alpha:.3f}")

    # =========================
    # 그래프 1: α vs 성공확률
    # =========================
    plt.figure()
    plt.plot(
        alphas,
        [stats[a]["success"] / stats[a]["trials"] for a in alphas],
        marker="o"
    )
    plt.axvline(1 / math.e, linestyle="--", label="1/e")
    plt.xlabel("α")
    plt.ylabel("Success Probability")
    plt.title("α vs Success Probability")
    plt.legend()
    plt.show()

    # =========================
    # 그래프 2: epoch vs α
    # =========================
    plt.figure()
    plt.plot(range(1, epochs + 1), best_alpha_history, marker="o")
    plt.axhline(1 / math.e, linestyle="--", label="1/e")
    plt.xlabel("Epoch")
    plt.ylabel("Best α")
    plt.title("α vs Epoch (Learning Dynamics)")
    plt.legend()
    plt.show()


# =========================
# 실행 (이거 없으면 아무것도 안 뜸)
# =========================
if __name__ == "__main__":
    heuristic_learning()
