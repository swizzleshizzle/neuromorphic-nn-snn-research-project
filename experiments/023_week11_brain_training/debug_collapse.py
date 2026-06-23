"""DEBUG (Week-12) — instrument the REINFORCE training loop to find the root cause
of the learn-then-collapse failure observed in EXP-023.

Phase-1 evidence gathering (systematic-debugging): no fixes. Per episode, log the
policy entropy, the greedy (argmax-logit) action the eval would pick, the sampled
action histogram, the mean motor spike-counts (the logits), the gradient norm, and
the mean firing rate of every region — so we can see WHERE/HOW it breaks before
proposing anything.

Run:
    .venv/Scripts/python.exe -u experiments/023_week11_brain_training/debug_collapse.py --episodes 80 --max-steps 60
"""

from __future__ import annotations

import argparse
import math

import torch

torch.set_num_threads(1)

from torch.distributions import Categorical  # noqa: E402

from neuromorphic.brain import Brain  # noqa: E402
from neuromorphic.envs import GridWorldEnv  # noqa: E402
import itertools  # noqa: E402

from neuromorphic.training.reinforce import discounted_returns, ema  # noqa: E402


def old_policy_params(brain):
    """The ORIGINAL (pre-fix) policy params — sensory + PFC + motor — so this script
    still reproduces the original motor-spike-count collapse the fix removed."""
    return itertools.chain(
        brain.sensory.parameters(), brain.pfc.parameters(), brain.motor.parameters()
    )


def make_dist(brain, obs, mode, temp, gen):
    """Build the action policy with a switchable logit parameterization (the variable under test).

    sum  = raw summed spike-counts over the window (current trainer — saturates).
    mean = mean firing rate over the window, in [0,1] (bounded — the hypothesis).
    temp = summed counts divided by a temperature.
    """
    out = brain.step(obs, store=False, recall=False, record=False, generator=gen)
    spikes = out["action_spikes"]  # [T, B, A]
    if mode == "mean":
        logits = spikes.mean(dim=0)[0]
    elif mode == "temp":
        logits = spikes.sum(dim=0)[0] / temp
    else:  # sum
        logits = spikes.sum(dim=0)[0]
    return Categorical(logits=logits), logits

GAMMA = 0.99
BASELINE_BETA = 0.1
REGIONS = ("sensory", "hippocampus", "prefrontal", "router", "motor")
PRIMARY = {"sensory": "concept", "hippocampus": "population", "prefrontal": "utility",
           "router": "select", "motor": "action"}
ACT = ("up", "rgt", "dwn", "lft")


def debug_episode(brain, env, opt, baseline, gen, max_steps, mode, temp):
    obs, _ = env.reset()
    log_probs, rewards, entropies = [], [], []
    act_counts = [0, 0, 0, 0]
    logit_sum = torch.zeros(4)
    reached, steps = False, 0
    while steps < max_steps:
        dist, logits = make_dist(brain, obs, mode, temp, gen)
        a = dist.sample()
        log_probs.append(dist.log_prob(a))
        entropies.append(dist.entropy().item())
        act_counts[int(a)] += 1
        logit_sum += logits.detach()
        obs, r, term, trunc, _ = env.step(int(a))
        rewards.append(float(r))
        steps += 1
        if term:
            reached = True
            break
        if trunc:
            break

    returns = torch.tensor(discounted_returns(rewards, GAMMA), dtype=torch.float32)
    adv = returns - baseline
    loss = -(torch.stack(log_probs) * adv).sum()
    opt.zero_grad()
    loss.backward()
    gnorm = math.sqrt(sum(p.grad.norm().item() ** 2 for p in old_policy_params(brain) if p.grad is not None))
    opt.step()

    mean_logits = (logit_sum / max(1, steps))
    return {
        "reward": sum(rewards),
        "steps": steps,
        "reached": reached,
        "entropy": sum(entropies) / len(entropies),
        "greedy": int(mean_logits.argmax()),
        "acts": act_counts,
        "mean_logits": mean_logits.tolist(),
        "mean_return": float(returns.mean()),
        "gnorm": gnorm,
        "adv_abs": float(adv.abs().mean()),
    }


def region_rates(brain, gen):
    """Mean firing rate of each region's primary output train on a fixed probe obs."""
    out = brain.step([0, 0, 4, 4], store=True, recall=True, record=True, generator=gen)
    rates = {}
    for rid in REGIONS:
        train = out["recordings"][rid].get(PRIMARY[rid])
        rates[rid] = float(train.float().mean()) if train is not None else float("nan")
    return rates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["sum", "mean", "temp"], default="sum")
    p.add_argument("--temp", type=float, default=8.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    env = GridWorldEnv()
    brain = Brain(grid_n=env.size, seed=args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    opt = torch.optim.Adam(old_policy_params(brain), lr=args.lr)

    print(f"DEBUG EXP-023 collapse · {args.episodes} ep · max_steps {args.max_steps} · lr {args.lr} · mode {args.mode}")
    print("ep  | rew  stp G | entropy greedy acts[u,r,d,l]      | mean_logits[u,r,d,l]        | gnorm  adv | region rates s/h/p/r/m")
    baseline = 0.0
    for ep in range(args.episodes):
        d = debug_episode(brain, env, opt, baseline, gen, args.max_steps, args.mode, args.temp)
        baseline = ema(baseline, d["mean_return"], BASELINE_BETA)
        rr = region_rates(brain, gen)
        ml = " ".join(f"{x:5.1f}" for x in d["mean_logits"])
        rates = "/".join(f"{rr[r]:.2f}" for r in REGIONS)
        g = "G" if d["reached"] else "."
        print(f"{ep + 1:3d} | {d['reward']:5.0f} {d['steps']:3d} {g} | "
              f"{d['entropy']:.3f}  {ACT[d['greedy']]:>3}  {d['acts']} | "
              f"{ml} | {d['gnorm']:5.2f} {d['adv_abs']:4.1f} | {rates}", flush=True)


if __name__ == "__main__":
    main()
