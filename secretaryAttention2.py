import argparse
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 실험 재현성
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 유한 N에서의 최적값 p*(N)과 k*
def finite_optimum_p_and_k(N: int) -> Tuple[float, int]:
    H = [0.0] * N
    for i in range(1, N):
        H[i] = H[i - 1] + 1.0 / i  # H[i] = sum_{j=1..i} 1/j

    H_Nm1 = H[N - 1]
    best_p = -1.0
    best_k = 1

    # k in [1, N-1]
    for k in range(1, N):
        p = (k / N) * (H_Nm1 - H[k - 1])
        if p > best_p:
            best_p = p
            best_k = k

    return best_p, best_k

# 비서 문제, 순열 생성
def generate_permutations(N: int, batch_size: int, device: torch.device) -> torch.Tensor:
    perms = np.stack([np.random.permutation(np.arange(1, N + 1)) for _ in range(batch_size)], axis=0)
    return torch.tensor(perms, dtype=torch.long, device=device)

# 정책이 확인할 feature 구성
def build_features(perms: torch.Tensor) -> torch.Tensor:
    """
    perms: (B, N) ranks 1..N
    Features per step (causal):
      - best_so_far / N
      - is_record
      - position t/N
    Output: (B, N, 3)
    """
    B, N = perms.shape
    best_so_far = torch.cummin(perms, dim=1).values
    is_record = (perms == best_so_far).float()
    rel_best = best_so_far.float() / float(N)
    pos = (torch.arange(1, N + 1, device=perms.device).float() / float(N)).unsqueeze(0).repeat(B, 1)
    return torch.stack([rel_best, is_record, pos], dim=-1)

# 미래 정보 못보게 방지
def causal_mask(N: int, device: torch.device) -> torch.Tensor:
    m = torch.full((N, N), float("-inf"), device=device)
    return torch.triu(m, diagonal=1)

# 어텐션 기반 정지 정책
class StopPolicy(nn.Module):
    def __init__(self, N: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.N = N
        self.in_proj = nn.Linear(3, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, N, d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, feats: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(feats) + self.pos_emb
        h = self.enc(h, mask=attn_mask)
        return self.head(h).squeeze(-1)

# 롤아웃 함수
def rollout_batch(
    model: StopPolicy,
    feats: torch.Tensor,
    perms: torch.Tensor,
    attn_mask: torch.Tensor,
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    reward: (B,) 1 if chosen rank == 1 else 0
    logprob_sum: (B,)
    """
    device = feats.device
    B, N, _ = feats.shape
    logits = model(feats, attn_mask)  # (B, N)

    done = torch.zeros(B, dtype=torch.bool, device=device)
    chosen_rank = torch.full((B,), N, dtype=torch.long, device=device)
    logprob_sum = torch.zeros(B, dtype=torch.float, device=device)

    for t in range(N):
        idx = (~done).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            break

        p_stop = torch.sigmoid(logits[idx, t])

        if deterministic:
            action_stop = (p_stop >= 0.5).float()
            logp = torch.zeros_like(action_stop)
        else:
            dist = torch.distributions.Bernoulli(probs=p_stop)
            action_stop = dist.sample()
            logp = dist.log_prob(action_stop)

        logprob_sum[idx] += logp

        stopping = action_stop > 0.5
        if stopping.any():
            stop_idx = idx[stopping]
            done[stop_idx] = True
            chosen_rank[stop_idx] = perms[stop_idx, t]

        if t == N - 1:
            remain = (~done).nonzero(as_tuple=False).squeeze(-1)
            if remain.numel() > 0:
                done[remain] = True
                chosen_rank[remain] = perms[remain, N - 1]

    reward = (chosen_rank == 1).float()
    return reward, logprob_sum


@torch.no_grad()
def estimate_success_probability(
    model: StopPolicy,
    N: int,
    device: torch.device,
    eval_episodes: int,
    batch_size: int,
) -> float:
    model.eval()
    m = causal_mask(N, device)

    total = 0
    succ = 0.0
    while total < eval_episodes:
        b = min(batch_size, eval_episodes - total)
        perms = generate_permutations(N, b, device)
        feats = build_features(perms)
        reward, _ = rollout_batch(model, feats, perms, m, deterministic=True)
        succ += float(reward.sum().item())
        total += b

    return succ / float(eval_episodes)

# threshold 달성 추적 및 기록
@dataclass
class Hit:
    epoch: Optional[int] = None
    time_sec: Optional[float] = None
    p_hat: Optional[float] = None


def relative_error(p_hat: float, target: float) -> float:
    return abs(p_hat - target) / (abs(target) + 1e-12)

# 단일 실행
def run_one(
    N: int,
    seed: int,
    device: torch.device,
    max_epochs: int,
    episodes_per_epoch: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    eval_episodes: int,
    thresholds: List[float],
    plot: bool,
    train_log_every: int,
    batch_log_every: int,
    target: float,
) -> Tuple[List[Tuple[int, float]], Dict[float, Hit]]:
    set_seed(seed)

    model = StopPolicy(N=N).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    m = causal_mask(N, device)

    baseline = 0.0
    beta = 0.95

    history: List[Tuple[int, float]] = []
    hits: Dict[float, Hit] = {th: Hit() for th in thresholds}
    start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        done_eps = 0
        epoch_reward_sum = 0.0
        epoch_batches = 0

        while done_eps < episodes_per_epoch:
            b = min(batch_size, episodes_per_epoch - done_eps)
            perms = generate_permutations(N, b, device)
            feats = build_features(perms)

            reward, logprob_sum = rollout_batch(model, feats, perms, m, deterministic=False)

            r_mean = float(reward.mean().item())
            epoch_reward_sum += r_mean
            epoch_batches += 1

            baseline = beta * baseline + (1 - beta) * r_mean
            adv = reward - baseline
            loss = -(adv.detach() * logprob_sum).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            done_eps += b

            if batch_log_every > 0 and (epoch_batches % batch_log_every == 0):
                elapsed = time.time() - start
                avg_r = epoch_reward_sum / max(1, epoch_batches)
                print(
                    f"[Train] epoch={epoch}, batch={epoch_batches}, avg_reward={avg_r:.6f}, "
                    f"baseline={baseline:.6f}, elapsed={elapsed:.1f}s",
                    flush=True
                )

        if train_log_every > 0 and (epoch % train_log_every == 0):
            elapsed = time.time() - start
            avg_r = epoch_reward_sum / max(1, epoch_batches)
            print(
                f"[Train] epoch={epoch}, avg_reward={avg_r:.6f}, baseline={baseline:.6f}, elapsed={elapsed:.1f}s",
                flush=True
            )

        if epoch == 1 or (epoch % eval_every == 0):
            p_hat = estimate_success_probability(model, N, device, eval_episodes, batch_size)
            history.append((epoch, p_hat))

            err = relative_error(p_hat, target)
            elapsed = time.time() - start
            print(
                f"[Eval] epoch={epoch}, p_hat={p_hat:.6f}, rel_error_to_p*(N)={err:.6%}, elapsed={elapsed:.1f}s",
                flush=True
            )

            for th in thresholds:
                if hits[th].epoch is None and err < th:
                    hits[th].epoch = epoch
                    hits[th].time_sec = elapsed
                    hits[th].p_hat = p_hat

            # 모든 threshold 도착 시 종료
            if all(hits[th].epoch is not None for th in thresholds):
                break

    if history:
        last_p = history[-1][1]
        for th in thresholds:
            if hits[th].p_hat is None:
                hits[th].p_hat = last_p

    if plot and history:
        xs = [x for x, _ in history]
        ys = [y for _, y in history]
        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys, label=f"Estimate (seed={seed})")
        plt.axhline(target, linestyle="--", label="p*(N) target")
        plt.axhline(1.0 / math.e, linestyle="--", label="1/e")
        plt.xlabel("Epoch (evaluation checkpoints)")
        plt.ylabel("Success Probability")
        plt.title(f"Secretary (Attention) - Run (N={N})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return history, hits

# 전체 실험(여러 run)
def run_experiment(args):
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if args.device == "auto" else args.device)
    )

    p_star, k_star = finite_optimum_p_and_k(args.N)
    asym = 1.0 / math.e
    th_str = ", ".join([f"{t*100:.3g}%" for t in args.thresholds])

    print(f"[Config] N={args.N}, runs={args.runs}, device={device}", flush=True)
    print(f"[Target] p*(N) = {p_star:.6f}  (k*={k_star}, k*/N={k_star/args.N:.6f})", flush=True)
    print(f"[Asymptote] 1/e = {asym:.6f}  (gap |p*(N)-1/e| = {abs(p_star-asym):.6f})", flush=True)
    print(f"[Thresholds] relative error to p*(N): {th_str}", flush=True)

    all_histories: List[List[Tuple[int, float]]] = []
    epochs_by_th: Dict[float, List[Optional[int]]] = {th: [] for th in args.thresholds}
    times_by_th: Dict[float, List[Optional[float]]] = {th: [] for th in args.thresholds}

    base_seed = args.seed

    for i in range(args.runs):
        seed = base_seed + i * 111
        print(f"\n===== Run {i+1}/{args.runs} (seed={seed}) =====", flush=True)

        history, hits = run_one(
            N=args.N,
            seed=seed,
            device=device,
            max_epochs=args.max_epochs,
            episodes_per_epoch=args.episodes_per_epoch,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            thresholds=args.thresholds,
            plot=args.plot_each,
            train_log_every=args.train_log_every,
            batch_log_every=args.batch_log_every,
            target=p_star,
        )

        all_histories.append(history)

        for th in args.thresholds:
            h = hits[th]
            epochs_by_th[th].append(h.epoch)
            times_by_th[th].append(h.time_sec)

            th_pct = th * 100
            if h.epoch is None:
                print(f"{th_pct:.3f}% -> NOT reached (last p_hat ~ {h.p_hat:.6f})", flush=True)
            else:
                err_pct = relative_error(h.p_hat, p_star) * 100
                print(
                    f"{th_pct:.3f}% -> reached: epoch={h.epoch}, time={h.time_sec:.2f}s, "
                    f"p_hat={h.p_hat:.6f}, rel_error={err_pct:.6f}%",
                    flush=True
                )

    # 평균 학습 곡선
    valid = [h for h in all_histories if h]
    if valid:
        min_len = min(len(h) for h in valid)
        xs = [valid[0][j][0] for j in range(min_len)]
        avg_ps = [sum(h[j][1] for h in valid) / len(valid) for j in range(min_len)]

        plt.figure(figsize=(8, 5))
        plt.plot(xs, avg_ps, label=f"Average estimate ({args.runs} runs)")
        plt.axhline(p_star, linestyle="--", label="p*(N)")
        plt.axhline(asym, linestyle="--", label="1/e")
        plt.xlabel("Epoch (evaluation checkpoints)")
        plt.ylabel("Success Probability")
        plt.title(f"Secretary (Attention) - Average Curve (N={args.N})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 실험 요약
    print("\n=== Average over reached runs only ===", flush=True)
    for th in args.thresholds:
        ep_list = [e for e in epochs_by_th[th] if e is not None]
        tm_list = [t for t in times_by_th[th] if t is not None]
        reached = len(ep_list)
        mean_ep = float(np.mean(ep_list)) if ep_list else float("nan")
        mean_tm = float(np.mean(tm_list)) if tm_list else float("nan")
        print(f"{th*100:.3g}% : reached {reached}/{args.runs}, avg_epoch={mean_ep:.2f}, avg_time={mean_tm:.2f}s", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--episodes_per_epoch", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_episodes", type=int, default=5000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--plot_each", action="store_true")

    p.add_argument("--thresholds", nargs="+", type=float, default=[0.01, 0.001, 0.0001],
                   help="Relative error thresholds to p*(N). Example: 0.01 0.001 0.0001")

    p.add_argument("--train_log_every", type=int, default=1)
    p.add_argument("--batch_log_every", type=int, default=0)

    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()
    if args.N < 10:
        raise ValueError("Use N >= 10.")
    if args.runs < 1:
        raise ValueError("runs must be >= 1.")
    run_experiment(args)


if __name__ == "__main__":
    main()