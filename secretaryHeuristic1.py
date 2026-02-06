# [단일 시행 코드]

import random
import math
import time
import csv
import matplotlib.pyplot as plt

# 비서 문제 정의
def secretary_trial(N, k):
    perm = random.sample(range(N), N)

    best = N
    for i in range(k):
        if perm[i] < best:
            best = perm[i]

    for i in range(k, N):
        if perm[i] < best:
            return 1 if perm[i] == 0 else 0

    return 0

# 전체 시뮬레이션
def run_simulation(
    N=10_000,
    max_trials=5_000_000,
    log_interval=100_000,
    csv_path="secretary_permutation.csv"
):

    alpha = 1 / math.e
    k = int(alpha * N)

    thresholds = [0.01, 0.001, 0.0001]
    hit = {th: None for th in thresholds}

    success = 0
    history = []

    start = time.time()

    for t in range(1, max_trials + 1):
        success += secretary_trial(N, k)
        p_hat = success / t
        rel_error = abs(p_hat - alpha) / alpha
        history.append((t, p_hat))

        for th in thresholds:
            if hit[th] is None and rel_error < th:
                hit[th] = (t, time.time() - start)

        if t % log_interval == 0:
            print(f"[{t:,}] p̂={p_hat:.6f}, rel_error={rel_error:.6%}")

        if all(hit.values()):
            break

    # CSV 저장
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "probability"])
        writer.writerows(history)

    print("\n=== Threshold reached ===")
    for th, (t_hit, sec) in hit.items():
        print(f"{th*100:.3f}% : {t_hit:,} trials, {sec:.2f}s")

    # 그래프
    plt.figure(figsize=(8, 5))
    plt.plot([x[0] for x in history], [x[1] for x in history], label="estimate")
    plt.axhline(alpha, linestyle="--", label="1/e")
    plt.xlabel("Trials")
    plt.ylabel("Success Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 실행
if __name__ == "__main__":
    run_simulation()

