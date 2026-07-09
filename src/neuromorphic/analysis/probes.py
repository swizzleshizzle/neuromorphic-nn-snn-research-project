"""Linear-probe + representation-geometry primitives for encoder characterization (EXP-027)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam

from neuromorphic.envs.gridworld import manhattan
from neuromorphic.training.pretrain import displacement_target

_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # up, right, down, left (matches GridWorldEnv)


def optimal_action_targets(states: torch.Tensor, grid_n: int) -> torch.Tensor:
    """[M,4] float: (i,a)=1 iff action a strictly reduces Manhattan distance to the goal."""
    M = states.shape[0]
    out = torch.zeros(M, 4)
    for i in range(M):
        ax, ay, gx, gy = (int(v) for v in states[i])
        d0 = manhattan((ax, ay), (gx, gy))
        for a, (dx, dy) in enumerate(_DELTAS):
            nx = min(max(ax + dx, 0), grid_n - 1)
            ny = min(max(ay + dy, 0), grid_n - 1)
            if manhattan((nx, ny), (gx, gy)) < d0:
                out[i, a] = 1.0
    return out


def _r2(pred: torch.Tensor, y: torch.Tensor) -> float:
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean(dim=0)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def ridge_probe(X_tr, Y_tr, X_te, Y_te, lam: float) -> dict:
    """Explicit L2-penalized linear probe (normal equations); held-out R2 + weights."""
    N = X_tr.shape[1]
    A = X_tr.T @ X_tr + lam * torch.eye(N)
    W = torch.linalg.solve(A, X_tr.T @ Y_tr)   # [N, Ytarget]
    return {"r2": _r2(X_te @ W, Y_te), "weights": W}


def peraction_probe(X_tr, Y_tr, X_te, Y_te, *, epochs: int = 200, lr: float = 1e-2) -> dict:
    """Four independent linear+sigmoid probes; per-action held-out accuracy."""
    torch.manual_seed(0)
    probe = nn.Linear(X_tr.shape[1], 4)
    opt = Adam(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(probe(X_tr), Y_tr).backward()
        opt.step()
    with torch.no_grad():
        pred = (probe(X_te) > 0).float()
        acc = (pred == Y_te).float().mean(dim=0)   # [4]
    return {"acc": acc, "mean_acc": float(acc.mean())}


def shuffle_null(fit_fn, X_tr, Y_tr, X_te, Y_te, *, n: int, seed: int) -> dict:
    """Permute train labels, refit n times; empirical chance band (mean, hi=max)."""
    vals = []
    for i in range(n):
        g = torch.Generator().manual_seed(seed + i)
        perm = torch.randperm(Y_tr.shape[0], generator=g)
        vals.append(float(fit_fn(X_tr, Y_tr[perm], X_te, Y_te)))
    t = torch.tensor(vals)
    return {"mean": float(t.mean()), "hi": float(t.max())}


def pca_reduce(X_tr, X_te, k: int):
    """Fit PCA on train (center + top-k right singular vectors), apply to both."""
    mu = X_tr.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X_tr - mu, full_matrices=False)
    comp = Vh[:k].T                       # [N, k]
    return (X_tr - mu) @ comp, (X_te - mu) @ comp


def participation_ratio(X) -> float:
    """(sum lam)^2 / sum(lam^2) of the covariance spectrum; ~effective dimensionality."""
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    lam = torch.linalg.eigvalsh(cov).clamp(min=0)
    return float((lam.sum() ** 2) / ((lam ** 2).sum() + 1e-12))


def unit_importance(X, Y, *, lam: float = 1e-2) -> torch.Tensor:
    """Rank units by summed abs ridge weight; returns unit indices, most important first."""
    W = ridge_probe(X, Y, X, Y, lam=lam)["weights"]   # [N, Yt]
    score = W.abs().sum(dim=1)
    return torch.argsort(score, descending=True)


def keepk_curve(X_tr, Y_tr, X_te, Y_te, order, ks, lam: float) -> list:
    """Held-out R2 keeping only the top-k units (by `order`) for each k in ks."""
    out = []
    for k in ks:
        idx = order[:k]
        out.append({"k": int(k), "r2": ridge_probe(X_tr[:, idx], Y_tr, X_te[:, idx], Y_te, lam=lam)["r2"]})
    return out


def dropk_curve(X_tr, Y_tr, X_te, Y_te, order, ks, lam: float) -> list:
    """Held-out R2 after DROPPING the top-k units (by `order`), keeping the remaining N-k."""
    out = []
    for k in ks:
        idx = order[k:]
        if idx.numel() == 0:
            r2 = _r2(torch.zeros_like(Y_te), Y_te)   # no units left -> predict-zero endpoint
        else:
            r2 = ridge_probe(X_tr[:, idx], Y_tr, X_te[:, idx], Y_te, lam=lam)["r2"]
        out.append({"k": int(k), "r2": r2})
    return out


def fraction_for_r2(keep_curve, full_r2, *, n_units, frac: float = 0.9) -> float:
    """Fraction of units (from a keep-k curve) needed to reach `frac` of the full-population R2."""
    if full_r2 <= 0:
        return 1.0
    target = frac * full_r2
    for c in sorted(keep_curve, key=lambda c: c["k"]):
        if c["r2"] >= target:
            return c["k"] / n_units
    return 1.0


def single_unit_r2(X_tr, Y_tr, X_te, Y_te, *, lam: float) -> torch.Tensor:
    """Per-unit held-out R2 (each unit probed alone) -> `[N]` tensor; the selectivity spread."""
    N = X_tr.shape[1]
    out = torch.empty(N)
    for j in range(N):
        out[j] = ridge_probe(X_tr[:, j:j + 1], Y_tr, X_te[:, j:j + 1], Y_te, lam=lam)["r2"]
    return out


REGION_SIGNALS = {
    "sensory": ("concept", 64),
    "sensory_hidden": ("hidden", 128),
    "prefrontal": ("utility", 4),
    "prefrontal_state": ("state", 100),
    "router": ("gate", 4),
    "motor": ("action", 4),
    "hippocampus": ("population", 150),
}


def region_rate_matrix(brain, states, *, region_key, signal_key, width,
                       recall=False, T=None, generator=None) -> torch.Tensor:
    """[M, width] mean-over-T rate for one region signal; zero-filled if the region is bypassed.

    Uses a single batched brain.step(record=True); the region name is the Brain._regions key
    (sensory/hippocampus/prefrontal/router/motor), NOT a dashboard/output alias. A bypassed
    region (hippocampus under recall=False) has an empty recordings dict -> return zeros instead
    of index-crashing. `T` (when given) temporarily narrows `brain.T` around the encode so
    reduced-window smoke runs are actually faster; `None` leaves `brain.T` untouched (default).
    """
    obs = states if torch.is_tensor(states) else torch.tensor(states)
    prev_T = brain.T
    if T is not None:
        brain.T = int(T)
    try:
        out = brain.step(obs, store=False, recall=recall, record=True, generator=generator)
    finally:
        brain.T = prev_T
    rec = out["recordings"].get(region_key, {})           # {} for a bypassed region
    train = rec.get(signal_key) if isinstance(rec, dict) else None
    if train is None:
        return torch.zeros(obs.shape[0], width)
    return train.mean(dim=0)                               # [T,M,N] -> [M,N]


def all_region_rates(brain, states, *, recall=False, T=None, generator=None) -> dict:
    """One `brain.step` forward pass -> `{region_key: [M, width]}` for every REGION_SIGNALS entry.

    Slices every region signal (sensory concept + hidden, PFC utility + state, router gate, motor
    action, hippocampus population) out of the SAME recordings dict, so the cross-region contrast
    and the concept geometry share a single Poisson draw instead of one independent draw per region
    (and one `brain.step` instead of ~8). Bypassed regions (empty recording, e.g. hippocampus under
    recall=False) are zero-filled. Honors a reduced `T` by temporarily narrowing `brain.T`
    (restored after); `None` leaves `brain.T` untouched.
    """
    obs = states if torch.is_tensor(states) else torch.tensor(states)
    M = obs.shape[0]
    prev_T = brain.T
    if T is not None:
        brain.T = int(T)
    try:
        out = brain.step(obs, store=False, recall=recall, record=True, generator=generator)
    finally:
        brain.T = prev_T
    recordings = out["recordings"]
    rates = {}
    for region_key, (signal_key, width) in REGION_SIGNALS.items():
        rec = recordings.get(region_key.split("_")[0], {})   # sensory_hidden -> sensory, etc.
        signal = rec.get(signal_key) if isinstance(rec, dict) else None
        rates[region_key] = torch.zeros(M, width) if signal is None else signal.mean(dim=0)
    return rates


def task_targets(states, grid_n) -> dict:
    return {
        "displacement": displacement_target(states, grid_n),
        "optimal_action": optimal_action_targets(states, grid_n),
    }
