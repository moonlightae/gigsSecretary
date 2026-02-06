# [10회 시행용 코드]

import random
import math
import time
import csv
import matplotlib.pyplot as plt

# 1회 시행 (정확한 비서 문제)
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


# 전체 시뮬레이션 (1회)
def run_simulation(
    N=10_000,
    max_trials=5_000_000,
    log_interval=100_000,
    plot=True
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

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot([x[0] for x in history], [x[1] for x in history], label="estimate")
        plt.axhline(alpha, linestyle="--", label="1/e")
        plt.xlabel("Trials")
        plt.ylabel("Success Probability")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return history, hit

# 10회 반복 실험
def run_10_experiments():
    thresholds = [0.01, 0.001, 0.0001]

    all_histories = []
    time_records = {th: [] for th in thresholds}

    for i in range(1, 11):
        print(f"\n===== Experiment {i} =====")
        history, hit = run_simulation(plot=False)

        all_histories.append(history)

        for th in thresholds:
            t_hit, sec = hit[th]
            time_records[th].append(sec)
            print(f"{th*100:.3f}% → {t_hit:,} trials, {sec:.2f}s")

    # 평균 시간 출력
    print("\n=== Average Time to Reach Threshold ===")
    for th in thresholds:
        avg_time = sum(time_records[th]) / len(time_records[th])
        print(f"{th*100:.3f}% : {avg_time:.2f}s")

    # 전체 평균 그래프
    min_len = min(len(h) for h in all_histories)

    avg_prob = []
    trials = []

    for i in range(min_len):
        trials.append(all_histories[0][i][0])
        avg_prob.append(
            sum(h[i][1] for h in all_histories) / len(all_histories)
        )

    alpha = 1 / math.e

    plt.figure(figsize=(8, 5))
    plt.plot(trials, avg_prob, label="Average estimate (10 runs)")
    plt.axhline(alpha, linestyle="--", label="1/e")
    plt.xlabel("Trials")
    plt.ylabel("Success Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 실행
if __name__ == "__main__":
    run_10_experiments()