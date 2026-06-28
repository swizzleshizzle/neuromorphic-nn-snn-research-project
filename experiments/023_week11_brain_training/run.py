"""EXP-023 — first grid-world training run (Week-11 S3 hands-on, ADR-0001).

Trains the five-region Brain on the 5x5 grid world with surrogate-gradient REINFORCE
(memory bypassed, recall=False). Prints an untrained baseline, trains for EPISODES,
writes a training curve, and records one trained episode as a dashboard trace.

Run (repo root, venv active):
    .venv/Scripts/python.exe experiments/023_week11_brain_training/run.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from neuromorphic.brain import Brain  # noqa: E402
from neuromorphic.envs import GridWorldEnv  # noqa: E402
from neuromorphic.monitor import FileSink, record_policy_episode  # noqa: E402
from neuromorphic.training.reinforce import (  # noqa: E402
    ema,
    greedy_action,
    make_policy_head,
    policy_parameters,
    train_episode,
)

# Tiny tensors (4-neuron regions) — multi-threading is pure overhead here. One thread
# is markedly faster than letting torch thrash every core.
torch.set_num_threads(1)

GAMMA = 0.99
BASELINE_BETA = 0.1

CURVE = Path("outputs/week11_training_curve.png")
TRACE = Path("outputs/week11_trained_trace.jsonl")


def eval_avg_reward(brain, head, env, n, gen, max_steps) -> tuple[float, float]:
    """Average reward and goal-reached fraction over n greedy episodes (head policy, no learning)."""
    total = 0.0
    goals = 0
    for _ in range(n):
        obs, _ = env.reset()
        steps, reached = 0, False
        while steps < max_steps:
            with torch.no_grad():
                a = greedy_action(brain, head, obs, generator=gen)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            steps += 1
            if term:
                reached = True
                break
            if trunc:
                break
        goals += 1 if reached else 0
    return total / n, goals / n


def moving_avg(xs: list[float], k: int = 20) -> list[float]:
    out = []
    for i in range(len(xs)):
        window = xs[max(0, i - k + 1):i + 1]
        out.append(sum(window) / len(window))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-023 REINFORCE grid-world training run")
    p.add_argument("--episodes", type=int, default=200, help="training episodes")
    p.add_argument("--max-steps", type=int, default=100, help="max env steps per episode")
    p.add_argument("--eval-episodes", type=int, default=20, help="greedy eval episodes (pre/post)")
    p.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    env = GridWorldEnv()
    brain = Brain(grid_n=env.size, seed=args.seed)
    head = make_policy_head(brain)
    gen = torch.Generator().manual_seed(args.seed)

    print(f"EXP-023 · {args.episodes} episodes · max_steps {args.max_steps} · lr {args.lr}", flush=True)
    pre_reward, pre_goals = eval_avg_reward(brain, head, env, args.eval_episodes, gen, args.max_steps)
    print(f"untrained: avg reward {pre_reward:.1f} · reached goal {pre_goals:.0%}", flush=True)

    opt = torch.optim.Adam(policy_parameters(head), lr=args.lr)
    baseline = 0.0
    rewards: list[float] = []
    for ep in range(args.episodes):
        stats = train_episode(
            brain, head, env, opt, gamma=GAMMA, baseline=baseline,
            generator=gen, max_steps=args.max_steps,
        )
        baseline = ema(baseline, stats["mean_return"], BASELINE_BETA)
        rewards.append(stats["total_reward"])
        recent = sum(rewards[-20:]) / min(20, len(rewards))
        goal = "GOAL" if stats["reached_goal"] else "    "
        print(f"  ep {ep + 1:4d} · reward {stats['total_reward']:6.1f} · "
              f"avg20 {recent:6.1f} · steps {stats['steps']:3d} · {goal}", flush=True)

    post_reward, post_goals = eval_avg_reward(brain, head, env, args.eval_episodes, gen, args.max_steps)
    print(f"trained:   avg reward {post_reward:.1f} · reached goal {post_goals:.0%}", flush=True)
    print(f"delta:     {post_reward - pre_reward:+.1f} reward", flush=True)

    CURVE.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.3, label="episode reward")
    plt.plot(moving_avg(rewards), label="moving avg (20)")
    plt.axhline(pre_reward, ls="--", c="gray", label="untrained baseline")
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.title("EXP-023 — REINFORCE on 5x5 grid world")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE, dpi=110)
    print(f"training curve -> {CURVE}")

    # Record one trained episode for the dashboard, driven by the TRAINED HEAD
    # (recall=False, the actual policy). Bypassed regions render silent; only the
    # sensory region is on the policy path in v1.
    sink = FileSink(TRACE)
    summary = record_policy_episode(
        brain, head, env, sink, seed=args.seed, recall=False,
        policy_regions=("sensory",), generator=gen,
    )
    print(f"trained trace  -> {TRACE} (reached_goal={summary['reached_goal']})", flush=True)


if __name__ == "__main__":
    main()
