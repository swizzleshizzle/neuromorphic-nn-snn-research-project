"""Approximate value iteration for a "moves to solved" judge (EXP-073).

The judge learns from its OWN look-ahead through the simulator: a state's target is one move plus
the best child's value under a frozen copy of the judge. No distance table is read here; probe and
exclusion lists arrive as plain arguments. Training builds a graph through the spiking encoder in
arm A, so nothing in this module may route through Brain.step or the look-ahead helpers, which run
under no_grad.
"""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuromorphic.envs.cube import SOLVED, apply_move, is_solved


class Judge(nn.Module):
    def __init__(self, encoder_fn, sensory: nn.Module, T: int, content: int, hidden: int = 128):
        super().__init__()
        self.encoder_fn = encoder_fn
        self.sensory = sensory
        self.T = T
        self.head = nn.Sequential(nn.Linear(content, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def concept(self, states, generator):
        obs = torch.as_tensor(np.array(states), dtype=torch.long)
        spikes = self.encoder_fn(obs, T=self.T, generator=generator)
        return self.sensory(spikes).mean(dim=0)

    def forward(self, states, generator):
        return F.softplus(self.head(self.concept(states, generator))).squeeze(-1)


def judge_from_brain(brain, hidden: int = 128) -> Judge:
    return Judge(brain._encoder, copy.deepcopy(brain.sensory), brain.T, brain.content, hidden)


def random_walk_states(n, max_len, rng, n_actions, exclude, *, start=SOLVED,
                       apply_fn=apply_move):
    out = []
    while len(out) < n:
        s = start
        for _ in range(rng.randint(1, max_len)):
            s = apply_fn(s, rng.randrange(n_actions))
        if s not in exclude:
            out.append(s)
    return out


def bellman_targets(states, jt, n_actions, generator, draws=1, *, apply_fn=apply_move,
                    goal_fn=is_solved):
    with torch.no_grad():
        children = [apply_fn(s, a) for s in states for a in range(n_actions)]
        vals = torch.stack([jt(children, generator) for _ in range(draws)]).mean(dim=0)
        solved_child = torch.tensor([goal_fn(c) for c in children])
        vals = torch.where(solved_child, torch.zeros_like(vals), vals)
        best = vals.view(len(states), n_actions).min(dim=1).values
        y = 1.0 + best
        solved = torch.tensor([goal_fn(s) for s in states])
        return torch.where(solved, torch.zeros_like(y), y)


def sync_target(judge):
    # `judge.sensory` may already have run a live-graph forward (train_step in arm A), which
    # leaves `mem1`/`mem2` as non-leaf tensors attached to that graph. torch's deepcopy protocol
    # raises on a non-leaf tensor, so reset the membrane state to fresh leaf zeros first. This is
    # a strict no-op on the next real forward, which calls `reset()` itself regardless.
    judge.sensory.reset()
    jt = copy.deepcopy(judge)
    for p in jt.parameters():
        p.requires_grad_(False)
    return jt


def make_optimizer(judge, arm):
    if arm == "A":
        return torch.optim.Adam([{"params": judge.head.parameters(), "lr": 1e-3},
                                 {"params": judge.sensory.parameters(), "lr": 1e-4}])
    if arm == "B":
        for p in judge.sensory.parameters():
            p.requires_grad_(False)
        return torch.optim.Adam(judge.head.parameters(), lr=1e-3)
    raise ValueError(f"unknown arm {arm!r}")


def train_step(judge, jt, optimizer, states, n_actions, generator, draws=1):
    y = bellman_targets(states, jt, n_actions, generator, draws)
    loss = F.mse_loss(judge(states, generator), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss)


def _rank(values):
    """Average ranks (1-indexed), ties sharing the mean of the ranks they span."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rank correlation: Pearson correlation of average ranks (ties averaged)."""
    rx = _rank(list(x))
    ry = _rank(list(y))
    n = len(rx)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    dx = [a - mean_rx for a in rx]
    dy = [a - mean_ry for a in ry]
    num = sum(a * b for a, b in zip(dx, dy))
    den = (sum(a * a for a in dx) * sum(b * b for b in dy)) ** 0.5
    return num / den if den > 0 else 0.0


def probe_judge(judge, probe, generator):
    """Score `judge` against a fixed (state, true_distance) probe set.

    All probe states are encoded in ONE batched call under torch.no_grad(), matching the
    batching discipline `bellman_targets` already uses for children.
    """
    states = [s for s, _ in probe]
    dists = [d for _, d in probe]
    with torch.no_grad():
        vals = judge(states, generator).tolist()

    spearman_all = spearman(dists, vals)

    by_dist = {}
    for d, v in zip(dists, vals):
        by_dist.setdefault(d, []).append(v)
    mean_j_by_distance = {str(d): sum(vs) / len(vs) for d, vs in by_dist.items()}

    pairs_7_11 = [(d, v) for d, v in zip(dists, vals) if 7 <= d <= 11]
    if len(pairs_7_11) >= 2:
        d711, v711 = zip(*pairs_7_11)
        spearman_7_11 = spearman(list(d711), list(v711))
    else:
        # Not enough depth-7-to-11 entries in this probe to correlate; NaN so a probe set with
        # no deep entries cannot silently read as "measured, uncorrelated".
        spearman_7_11 = float("nan")

    means_7_11 = [mean_j_by_distance[str(d)] for d in range(7, 12) if str(d) in mean_j_by_distance]
    # Vacuously True with fewer than two distances present: there is no adjacent pair to violate
    # monotonicity, so there is nothing for this flag to catch at that point.
    monotone_7_11 = all(means_7_11[k] < means_7_11[k + 1] for k in range(len(means_7_11) - 1))

    return {
        "spearman_all": spearman_all,
        "spearman_7_11": spearman_7_11,
        "mean_j_by_distance": mean_j_by_distance,
        "monotone_7_11": monotone_7_11,
    }


def _save_checkpoint(ckpt_dir, judge, jt, opt, next_update, rng, generator, probes):
    torch.save(judge.state_dict(), ckpt_dir / "judge.pt")
    torch.save(jt.state_dict(), ckpt_dir / "jt.pt")
    torch.save(opt.state_dict(), ckpt_dir / "opt.pt")
    rstate = rng.getstate()
    state = {
        "update": next_update,
        "random_state": [rstate[0], list(rstate[1]), rstate[2]],
        "generator_state": generator.get_state().tolist(),
        "probes": probes,
    }
    (ckpt_dir / "state.json").write_text(json.dumps(state))


def train_judge(judge, arm, *, n_updates, batch, max_len, n_actions, exclude, probe, seed,
                sync_every, probe_every, draws, ckpt_dir, log=print):
    """Resumable approximate value iteration. Deterministic given `seed`: one
    `random.Random(seed)` drives the walks and one `torch.Generator().manual_seed(seed)` drives
    the stochastic encoding. Each probe pass uses its OWN `torch.Generator().manual_seed(seed)`
    (not the training generator): a probe never advances the training RNG, so how many times a
    call happens to probe (including the unconditional probe at the end of a short, resumed call)
    cannot change the training trajectory. It also means every probe in the history is scored
    under identical encoding noise, so a change in spearman_7_11 across checkpoints reflects the
    judge, not encoder randomness.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_path = ckpt_dir / "state.json"

    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    opt = make_optimizer(judge, arm)
    probes = []
    start = 0
    jt = sync_target(judge)

    if state_path.exists():
        state = json.loads(state_path.read_text())
        judge.load_state_dict(torch.load(ckpt_dir / "judge.pt"))
        jt = sync_target(judge)
        jt.load_state_dict(torch.load(ckpt_dir / "jt.pt"))
        opt = make_optimizer(judge, arm)
        opt.load_state_dict(torch.load(ckpt_dir / "opt.pt"))
        rs = state["random_state"]
        rng.setstate((rs[0], tuple(rs[1]), rs[2]))
        generator.set_state(torch.tensor(state["generator_state"], dtype=torch.uint8))
        start = state["update"]
        probes = state["probes"]

    t0 = time.time()
    loss_last = None
    for update in range(start, n_updates):
        if update % sync_every == 0:
            jt = sync_target(judge)
        states = random_walk_states(batch, max_len, rng, n_actions, exclude)
        loss_last = train_step(judge, jt, opt, states, n_actions, generator, draws)
        if update % probe_every == 0 or update == n_updates - 1:
            probe_result = probe_judge(judge, probe, torch.Generator().manual_seed(seed))
            probes.append(probe_result)
            _save_checkpoint(ckpt_dir, judge, jt, opt, update + 1, rng, generator, probes)
            log(f"update {update}: loss {loss_last:.6f} "
                f"spearman_7_11 {probe_result['spearman_7_11']}")

    wall_s = time.time() - t0
    final_probe = probes[-1] if probes else probe_judge(judge, probe,
                                                         torch.Generator().manual_seed(seed))
    return {"updates": n_updates, "wall_s": wall_s, "loss_last": loss_last,
            "probes": probes, "final_probe": final_probe}
