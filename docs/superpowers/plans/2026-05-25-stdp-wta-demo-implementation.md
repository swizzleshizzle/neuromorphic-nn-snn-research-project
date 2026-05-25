# STDP-WTA Demo (EXP-012) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended for this small demo) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small unsupervised SNN (12 Poisson inputs → 4 LIF outputs, hard WTA + pair-STDP) that develops emergent selectivity for 3 spatial input patterns, plus 4 visualizations and a selectivity report.

**Architecture:** Single experiment folder `experiments/012_week7_stdp_demo/` with `patterns.py` (Poisson generator), `models.py` (`STDPLayer` doing forward + manual STDP in `torch.no_grad`), `run.py` (main training/viz/report), and a hardcoded `config.yaml`. No reuse of `ExperimentConfig` or `ExperimentTracker` — this is a one-off demo, not part of the tracked grid.

**Tech Stack:** PyTorch, snnTorch 0.9.x (`snn.Leaky` for LIF dynamics — but plasticity is custom), matplotlib, PyYAML.

**Parent spec:** `docs/design/2026-05-25-stdp-wta-demo.md`

**Discipline:**
- Inline smoke gates per task, NO pytest (matches EXP-011 discipline for non-pure code).
- 60-minute budget. Cut order if running long: drop spike raster first, drop weight-evolution snapshots second, NEVER drop final weight matrix + selectivity report.
- `outputs/` already gitignored. Only source committed.

---

## Task 0: Pre-flight + scaffold

- [ ] **Step 1: Clean tree check**

```powershell
git status
```

Expected: only `.claude/` untracked.

- [ ] **Step 2: Create folder + __init__.py**

```powershell
New-Item -ItemType Directory -Force -Path experiments\012_week7_stdp_demo | Out-Null
New-Item -ItemType File -Force -Path experiments\012_week7_stdp_demo\__init__.py | Out-Null
```

---

## Task 1: `patterns.py` — Poisson pattern generator

**Files:**
- Create: `experiments/012_week7_stdp_demo/patterns.py`

**Predict before executing:**
- `poisson_pattern_spikes(pattern_idx=0, num_steps=400)` (= 1000 ms at dt=2.5ms) — inputs {0,1,2,3} fire ~80–120 spikes each; inputs {4..11} fire 0–15 spikes each.

- [ ] **Step 1: Write `patterns.py`**

```python
"""Spatial Poisson pattern generator for EXP-012 STDP-WTA demo.

Three fixed spatial patterns over 12 input neurons; each pattern designates
4 'active' inputs that fire at active_rate_hz, others at background_rate_hz.
"""

from __future__ import annotations

import torch

PATTERNS: dict[str, list[int]] = {
    "A": [0, 1, 2, 3],
    "B": [4, 5, 6, 7],
    "C": [8, 9, 10, 11],
}
PATTERN_NAMES = list(PATTERNS.keys())  # ["A", "B", "C"]


def poisson_pattern_spikes(
    pattern_idx: int,
    num_steps: int,
    n_inputs: int = 12,
    active_rate_hz: float = 100.0,
    background_rate_hz: float = 5.0,
    dt_s: float = 0.0025,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Bernoulli-sample a [num_steps, n_inputs] spike tensor for one pattern.

    Args:
        pattern_idx: 0, 1, or 2 → patterns A, B, C
        num_steps: number of timesteps to generate
        n_inputs: 12 (locked by spec)
        active_rate_hz / background_rate_hz: firing rates
        dt_s: timestep in seconds (0.0025 = 2.5 ms)
        generator: optional torch.Generator for reproducibility
    """
    name = PATTERN_NAMES[pattern_idx]
    active = PATTERNS[name]

    p = torch.full((n_inputs,), background_rate_hz * dt_s)
    for i in active:
        p[i] = active_rate_hz * dt_s

    p_grid = p.unsqueeze(0).expand(num_steps, n_inputs)
    if generator is None:
        return torch.bernoulli(p_grid)
    return torch.bernoulli(p_grid, generator=generator)


def background_only_spikes(
    num_steps: int,
    n_inputs: int = 12,
    background_rate_hz: float = 5.0,
    dt_s: float = 0.0025,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Inter-trial gap: all inputs at background rate."""
    p = torch.full((num_steps, n_inputs), background_rate_hz * dt_s)
    if generator is None:
        return torch.bernoulli(p)
    return torch.bernoulli(p, generator=generator)
```

- [ ] **Step 2: Inline rate-check gate (gate 2 from spec §8)**

```powershell
.\.venv\Scripts\python.exe -c "
import sys
sys.path.insert(0, 'experiments/012_week7_stdp_demo')
import torch
from patterns import poisson_pattern_spikes

torch.manual_seed(0)
# 1000 ms of pattern A at dt=2.5ms → 400 steps
spk = poisson_pattern_spikes(pattern_idx=0, num_steps=400)
counts = spk.sum(dim=0)
print('per-input spike counts over 1s of pattern A:')
print(counts.tolist())
# active: indices 0-3 at 100Hz → expect ~100 spikes (range 80-120)
for i in [0, 1, 2, 3]:
    assert 60 <= counts[i] <= 140, f'active input {i}: {counts[i]} not in [60, 140]'
# background: indices 4-11 at 5Hz → expect ~5 spikes (range 0-15)
for i in range(4, 12):
    assert 0 <= counts[i] <= 18, f'background input {i}: {counts[i]} not in [0, 18]'
print('rate gate PASS')
"
```

Expected: active inputs ~100 spikes, background ~5 spikes, gate PASS.

---

## Task 2: `models.py` — `STDPLayer` (LIF + WTA + manual STDP)

**Files:**
- Create: `experiments/012_week7_stdp_demo/models.py`

**Predict before executing:**
- Init: `layer.W.shape == (12, 4)`, values in `[0, 1]`.
- Pre-training response: feed 100 steps of pattern A, get a `[100, 4]` output spike tensor. With random init weights and threshold 1.0, expect non-zero spikes from at least some outputs (i.e. the layer isn't silent).

- [ ] **Step 1: Write `models.py`**

```python
"""STDP-WTA layer for EXP-012.

A single feedforward layer (12 → 4) with:
  - LIF dynamics on the 4 output neurons (beta decay + threshold)
  - Hard winner-take-all: at most 1 output spikes per timestep
  - Manual pair-based STDP plasticity applied under torch.no_grad()

NOT a generic STDP library. Demo-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class STDPState:
    mem: torch.Tensor       # [n_post]  membrane potential
    x_pre: torch.Tensor     # [n_pre]   pre-synaptic trace
    y_post: torch.Tensor    # [n_post]  post-synaptic trace


class STDPLayer(nn.Module):
    """Feedforward layer with LIF + hard WTA + manual pair-STDP."""

    def __init__(
        self,
        n_pre: int = 12,
        n_post: int = 4,
        beta: float = 0.9,
        threshold: float = 1.0,
        a_plus: float = 0.005,
        a_minus: float = -0.0055,
        tau_pre_ms: float = 20.0,
        tau_post_ms: float = 20.0,
        dt_ms: float = 2.5,
        w_max: float = 1.0,
        w_init_low: float = 0.0,
        w_init_high: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.beta = beta
        self.threshold = threshold
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_max = w_max
        self.dt_ms = dt_ms
        # Trace decay factors per timestep
        self.alpha_pre = math.exp(-dt_ms / tau_pre_ms)
        self.alpha_post = math.exp(-dt_ms / tau_post_ms)

        gen = torch.Generator().manual_seed(seed)
        W = torch.empty(n_pre, n_post).uniform_(w_init_low, w_init_high, generator=gen)
        # Weights are parameters but never see gradient — kept as a buffer for clarity.
        self.register_buffer("W", W)

    def init_state(self, device: torch.device | str = "cpu") -> STDPState:
        return STDPState(
            mem=torch.zeros(self.n_post, device=device),
            x_pre=torch.zeros(self.n_pre, device=device),
            y_post=torch.zeros(self.n_post, device=device),
        )

    @torch.no_grad()
    def forward_step(
        self,
        spk_pre: torch.Tensor,           # [n_pre], 0/1
        state: STDPState,
        learn: bool = True,
    ) -> tuple[torch.Tensor, STDPState]:
        # 1) Integrate membrane
        cur = spk_pre @ self.W                                 # [n_post]
        mem = self.beta * state.mem + cur

        # 2) Hard WTA: at most one spike per step
        crossed = mem >= self.threshold                        # [n_post] bool
        spk_post = torch.zeros_like(mem)
        if crossed.any():
            # Pick highest-membrane crosser; zero others
            mem_for_pick = mem.masked_fill(~crossed, float("-inf"))
            winner = int(mem_for_pick.argmax().item())
            spk_post[winner] = 1.0
            # Subtract-reset only the winner; force losers' mem to 0 this step
            mem_new = torch.where(
                torch.arange(self.n_post, device=mem.device) == winner,
                mem - self.threshold,
                torch.zeros_like(mem),
            )
        else:
            mem_new = mem

        # 3) Decay traces, then apply spikes
        x_pre = state.x_pre * self.alpha_pre
        y_post = state.y_post * self.alpha_post

        if learn:
            # LTD: pre arrives AFTER post — every pre spike now triggers W += A_minus * y_post
            # outer product: dW[i,j] = A_minus * spk_pre[i] * y_post[j]
            if spk_pre.any():
                dW_ltd = self.a_minus * torch.outer(spk_pre, y_post)
                self.W += dW_ltd

            # LTP: pre arrived BEFORE post — every post spike now triggers W += A_plus * x_pre
            if spk_post.any():
                dW_ltp = self.a_plus * torch.outer(x_pre, spk_post)
                self.W += dW_ltp

            # Clip
            self.W.clamp_(0.0, self.w_max)

        # 4) Increment traces for THIS step's spikes (after STDP applied)
        x_pre = x_pre + spk_pre
        y_post = y_post + spk_post

        return spk_post, STDPState(mem=mem_new, x_pre=x_pre, y_post=y_post)
```

- [ ] **Step 2: Inline init-shape + non-silence gate (gates 1 + 3 from spec §8)**

```powershell
.\.venv\Scripts\python.exe -c "
import sys
sys.path.insert(0, 'experiments/012_week7_stdp_demo')
import torch
from models import STDPLayer
from patterns import poisson_pattern_spikes

torch.manual_seed(0)
layer = STDPLayer(seed=0)
assert layer.W.shape == (12, 4), layer.W.shape
assert (layer.W >= 0).all() and (layer.W <= 1.0).all()
print(f'init: W.shape={tuple(layer.W.shape)}, min={layer.W.min():.3f}, max={layer.W.max():.3f}, mean={layer.W.mean():.3f}')

# Pre-training: feed 400 steps of pattern A, no learning, check at least some outputs fire
torch.manual_seed(0)
spikes_in = poisson_pattern_spikes(pattern_idx=0, num_steps=400)
state = layer.init_state()
out_counts = torch.zeros(4)
for t in range(400):
    spk, state = layer.forward_step(spikes_in[t], state, learn=False)
    out_counts += spk
print(f'pre-training output spike counts on 1s pattern A: {out_counts.tolist()}')
assert out_counts.sum() > 0, 'silent network at init — every output never crossed threshold'
print('init + non-silence gates PASS')
"
```

Expected: shapes correct, network produces some output spikes pre-training.

---

## Task 3: `config.yaml` + `run.py` (training + selectivity gates)

**Files:**
- Create: `experiments/012_week7_stdp_demo/config.yaml`
- Create: `experiments/012_week7_stdp_demo/run.py`

**Predict before executing:**
- Full run takes <2 min on CPU (small network, 28k timesteps).
- Post-training: at least 2 of 4 outputs have a preferred pattern with ≥ 2× ratio over any other pattern.
- Weight std after training > 2× weight std at init.

- [ ] **Step 1: Write `config.yaml`**

```yaml
# EXP-012 STDP-WTA demo — locked hyperparameters per spec.
seed: 0

# Network
n_inputs: 12
n_outputs: 4
beta: 0.9
threshold: 1.0
w_max: 1.0

# STDP
a_plus: 0.005
a_minus: -0.0055
tau_pre_ms: 20.0
tau_post_ms: 20.0

# Simulation
dt_ms: 2.5
pattern_steps: 20         # 50 ms presentation
gap_steps: 8              # 20 ms gap
n_trials: 1000

# Input patterns (rates)
active_rate_hz: 100.0
background_rate_hz: 5.0

# Eval
n_eval_trials_per_pattern: 50    # post-training selectivity check
```

- [ ] **Step 2: Write `run.py` (training + gates; visualizations added in Task 4)**

```python
"""EXP-012 STDP-WTA demo — main training and selectivity report.

Visualizations are added in run.py (this file) as separate functions called
from main() — see Task 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from models import STDPLayer
from patterns import (
    PATTERN_NAMES,
    background_only_spikes,
    poisson_pattern_spikes,
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_trial(
    layer: STDPLayer,
    state,
    pattern_idx: int,
    cfg: dict,
    learn: bool,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, "STDPState"]:
    """Run a single trial: pattern_steps of one pattern + gap_steps of background.

    Returns (input_spikes [T, n_inputs], output_spikes [T, n_outputs], end_state).
    """
    pat = poisson_pattern_spikes(
        pattern_idx=pattern_idx,
        num_steps=cfg["pattern_steps"],
        n_inputs=cfg["n_inputs"],
        active_rate_hz=cfg["active_rate_hz"],
        background_rate_hz=cfg["background_rate_hz"],
        dt_s=cfg["dt_ms"] / 1000.0,
        generator=generator,
    )
    gap = background_only_spikes(
        num_steps=cfg["gap_steps"],
        n_inputs=cfg["n_inputs"],
        background_rate_hz=cfg["background_rate_hz"],
        dt_s=cfg["dt_ms"] / 1000.0,
        generator=generator,
    )
    spikes_in = torch.cat([pat, gap], dim=0)         # [pattern+gap, n_inputs]

    out_rec = []
    for t in range(spikes_in.shape[0]):
        spk_out, state = layer.forward_step(spikes_in[t], state, learn=learn)
        out_rec.append(spk_out)
    spikes_out = torch.stack(out_rec, dim=0)         # [T, n_outputs]
    return spikes_in, spikes_out, state


def measure_selectivity(layer: STDPLayer, cfg: dict, generator: torch.Generator) -> torch.Tensor:
    """Return a [n_patterns, n_outputs] matrix of mean firing rates (spikes / pattern_steps)
    measured with learning frozen.
    """
    n_patterns = len(PATTERN_NAMES)
    n_outputs = cfg["n_outputs"]
    counts = torch.zeros(n_patterns, n_outputs)
    trials_per = cfg["n_eval_trials_per_pattern"]
    for p_idx in range(n_patterns):
        for _ in range(trials_per):
            state = layer.init_state()
            _, spikes_out, _ = run_trial(layer, state, p_idx, cfg, learn=False, generator=generator)
            # Only count spikes during pattern window
            counts[p_idx] += spikes_out[: cfg["pattern_steps"]].sum(dim=0)
    return counts / float(trials_per * cfg["pattern_steps"])


def main():
    cfg = load_config(HERE / "config.yaml")
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 1)

    layer = STDPLayer(
        n_pre=cfg["n_inputs"],
        n_post=cfg["n_outputs"],
        beta=cfg["beta"],
        threshold=cfg["threshold"],
        a_plus=cfg["a_plus"],
        a_minus=cfg["a_minus"],
        tau_pre_ms=cfg["tau_pre_ms"],
        tau_post_ms=cfg["tau_post_ms"],
        dt_ms=cfg["dt_ms"],
        w_max=cfg["w_max"],
        seed=cfg["seed"],
    )

    W_init = layer.W.clone()
    print(f"[init] W mean={W_init.mean():.3f} std={W_init.std():.3f}")

    # Pre-training selectivity (gate 3 — should be roughly uniform)
    pre_resp = measure_selectivity(layer, cfg, gen)
    print("[pre-training] mean firing rate per (pattern, output):")
    print(pre_resp)

    # Training loop
    snapshots = {0: W_init.clone()}
    snapshot_at = {250, 500, 1000}
    state = layer.init_state()
    n_patterns = len(PATTERN_NAMES)
    for trial in range(1, cfg["n_trials"] + 1):
        pat_idx = int(torch.randint(0, n_patterns, (1,), generator=gen).item())
        _, _, state = run_trial(layer, state, pat_idx, cfg, learn=True, generator=gen)
        if trial in snapshot_at:
            snapshots[trial] = layer.W.clone()

    W_final = layer.W.clone()
    print(f"[final] W mean={W_final.mean():.3f} std={W_final.std():.3f}")

    # Post-training selectivity
    post_resp = measure_selectivity(layer, cfg, gen)
    print("[post-training] mean firing rate per (pattern, output):")
    print(post_resp)

    # Gates 4 + 5
    n_selective = 0
    selectivity_lines = []
    for j in range(cfg["n_outputs"]):
        rates = post_resp[:, j]
        if rates.max() <= 1e-6:
            selectivity_lines.append(f"Output {j}: silent")
            continue
        best = int(rates.argmax().item())
        best_rate = float(rates[best])
        other_rates = torch.cat([rates[:best], rates[best + 1:]])
        max_other = float(other_rates.max()) if other_rates.numel() > 0 else 0.0
        ratio = best_rate / max(max_other, 1e-9)
        is_selective = ratio >= 2.0
        if is_selective:
            n_selective += 1
        selectivity_lines.append(
            f"Output {j}: prefers pattern {PATTERN_NAMES[best]} "
            f"(rate {best_rate:.4f} vs next-best {max_other:.4f}, ratio {ratio:.2f}x) "
            f"{'[SELECTIVE]' if is_selective else '[non-selective]'}"
        )

    print()
    for line in selectivity_lines:
        print(line)
    print(f"\nSelective outputs: {n_selective}/{cfg['n_outputs']}")
    print(f"Weight std ratio (post/init): {(W_final.std() / W_init.std()).item():.2f}")

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "selectivity_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# EXP-012 STDP-WTA selectivity report\n\n")
        for line in selectivity_lines:
            f.write(line + "\n")
        f.write(f"\nSelective outputs (ratio >= 2x): {n_selective}/{cfg['n_outputs']}\n")
        f.write(f"Weight std ratio (post/init): {(W_final.std() / W_init.std()).item():.2f}\n")
    print(f"\nWrote {report_path}")

    assert n_selective >= 2, f"gate 4 FAILED: only {n_selective}/4 outputs selective"
    std_ratio = (W_final.std() / W_init.std()).item()
    assert std_ratio >= 2.0, f"gate 5 FAILED: weight std ratio {std_ratio:.2f} < 2.0"
    print("\nGates 4 + 5 PASS")

    # Return for the viz step (Task 4 will extend main() to use these)
    return {
        "snapshots": snapshots,
        "W_init": W_init,
        "W_final": W_final,
        "pre_resp": pre_resp,
        "post_resp": post_resp,
        "selectivity_lines": selectivity_lines,
        "n_selective": n_selective,
        "cfg": cfg,
    }


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it end-to-end**

```powershell
.\.venv\Scripts\python.exe experiments/012_week7_stdp_demo/run.py
```

Expected: prints W stats, pre-training response matrix (~uniform), post-training response matrix (peaked), per-output selectivity lines, "Selective outputs: ≥ 2/4", "Gates 4 + 5 PASS".

If gate 4 fails, triage in this order:
1. Try `a_plus: 0.01` and `a_minus: -0.011` (faster learning)
2. Try `threshold: 0.5` (lower → more output spikes → more STDP triggers)
3. Try `n_trials: 2000`

If gate 5 fails but gate 4 passes, that's odd — selectivity without weight diversification — surface it.

- [ ] **Step 4: Commit**

```powershell
git add experiments/012_week7_stdp_demo/__init__.py experiments/012_week7_stdp_demo/patterns.py experiments/012_week7_stdp_demo/models.py experiments/012_week7_stdp_demo/config.yaml experiments/012_week7_stdp_demo/run.py
git commit -m "exp 012: STDP-WTA demo — patterns + STDPLayer + training loop"
```

---

## Task 4: Visualizations

**Files:**
- Modify: `experiments/012_week7_stdp_demo/run.py` — append viz functions, extend `main()`

**Predict before executing:**
- 4 PNGs in `outputs/`: `weight_matrix_evolution.png`, `weight_matrix_final.png`, `tuning_curves.png`, `spike_raster_late.png`.
- `weight_matrix_final.png` should visibly show ~3 strong column-blocks (one per selective output) with row groupings 0–3 / 4–7 / 8–11.

- [ ] **Step 1: Add viz functions at the bottom of `run.py` (above `if __name__ == "__main__":`)**

```python
# ---- Visualizations ----

def _plot_weight_heatmap(ax, W: torch.Tensor, title: str, vmin: float = 0.0, vmax: float = 1.0):
    im = ax.imshow(W.cpu().numpy(), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xlabel("output neuron")
    ax.set_ylabel("input neuron")
    ax.set_title(title)
    ax.set_xticks(range(W.shape[1]))
    ax.set_yticks(range(W.shape[0]))
    # Group separators between input blocks 0-3, 4-7, 8-11
    for y in (3.5, 7.5):
        ax.axhline(y, color="white", linewidth=1.0, alpha=0.6)
    return im


def viz_weight_evolution(snapshots: dict[int, torch.Tensor], save_path: Path) -> None:
    import matplotlib.pyplot as plt
    keys = sorted(snapshots.keys())
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, t in zip(axes.flat, keys[:4]):
        im = _plot_weight_heatmap(ax, snapshots[t], title=f"trial {t}")
    fig.suptitle("Weight matrix evolution")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="weight")
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def viz_weight_final(
    W_final: torch.Tensor,
    post_resp: torch.Tensor,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    _plot_weight_heatmap(ax, W_final, title="Final weight matrix (post-training)")
    # Label each column with preferred pattern
    preferred = post_resp.argmax(dim=0).cpu().tolist()  # [n_post]
    labels = [f"out {j}\n→ {PATTERN_NAMES[preferred[j]]}" for j in range(W_final.shape[1])]
    ax.set_xticks(range(W_final.shape[1]))
    ax.set_xticklabels(labels)
    # Label row groups
    ax.set_yticks([1.5, 5.5, 9.5])
    ax.set_yticklabels(["A inputs\n(0-3)", "B inputs\n(4-7)", "C inputs\n(8-11)"])
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def viz_tuning_curves(post_resp: torch.Tensor, save_path: Path) -> None:
    """post_resp[p, j] = mean rate of output j when pattern p shown."""
    import matplotlib.pyplot as plt
    n_outputs = post_resp.shape[1]
    fig, axes = plt.subplots(1, n_outputs, figsize=(3 * n_outputs, 3), sharey=True)
    if n_outputs == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        rates = post_resp[:, j].cpu().numpy()
        ax.bar(PATTERN_NAMES, rates, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.set_title(f"output {j}")
        ax.set_xlabel("pattern")
        if j == 0:
            ax.set_ylabel("mean firing rate\n(spikes / step)")
    fig.suptitle("Post-training tuning curves")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def viz_spike_raster(layer: STDPLayer, cfg: dict, save_path: Path, generator: torch.Generator) -> None:
    """Render a ~200 ms window of input + output spikes late in training (learning frozen).

    Runs 4 trials with alternating patterns; concatenates and marks pattern onsets.
    """
    import matplotlib.pyplot as plt
    n_show_trials = 4
    pattern_seq = [0, 1, 2, 0]  # A, B, C, A
    all_in, all_out = [], []
    state = layer.init_state()
    for p_idx in pattern_seq:
        spikes_in, spikes_out, state = run_trial(layer, state, p_idx, cfg, learn=False, generator=generator)
        all_in.append(spikes_in)
        all_out.append(spikes_out)
    spikes_in = torch.cat(all_in, dim=0)
    spikes_out = torch.cat(all_out, dim=0)
    trial_len = cfg["pattern_steps"] + cfg["gap_steps"]
    dt = cfg["dt_ms"]

    fig, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})
    t_in, n_in = spikes_in.nonzero(as_tuple=True)
    ax_in.scatter(t_in.cpu() * dt, n_in.cpu(), s=8, c="black", marker="|")
    ax_in.set_ylabel("input #")
    ax_in.set_ylim(-0.5, cfg["n_inputs"] - 0.5)
    ax_in.set_title("Spike raster (post-training, learning frozen)")

    t_out, n_out = spikes_out.nonzero(as_tuple=True)
    ax_out.scatter(t_out.cpu() * dt, n_out.cpu(), s=40, c="red", marker="|")
    ax_out.set_xlabel("time (ms)")
    ax_out.set_ylabel("output #")
    ax_out.set_ylim(-0.5, cfg["n_outputs"] - 0.5)

    for trial_idx, p_idx in enumerate(pattern_seq):
        onset = trial_idx * trial_len * dt
        offset = onset + cfg["pattern_steps"] * dt
        for ax in (ax_in, ax_out):
            ax.axvspan(onset, offset, alpha=0.10, color="gray")
            ax.axvline(onset, color="gray", linewidth=0.8, alpha=0.6)
        ax_in.text(onset + 2, cfg["n_inputs"] - 0.5, PATTERN_NAMES[p_idx], fontsize=10, va="top")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 2: Extend `main()` to call viz functions before returning**

Insert this block in `main()` just BEFORE `return { ... }`:

```python
    # ---- Visualizations ----
    viz_weight_evolution(snapshots, out_dir / "weight_matrix_evolution.png")
    viz_weight_final(W_final, post_resp, out_dir / "weight_matrix_final.png")
    viz_tuning_curves(post_resp, out_dir / "tuning_curves.png")
    viz_spike_raster(layer, cfg, out_dir / "spike_raster_late.png", gen)
    print(f"Wrote 4 viz PNGs to {out_dir}/")
```

- [ ] **Step 3: Re-run end-to-end**

```powershell
.\.venv\Scripts\python.exe experiments/012_week7_stdp_demo/run.py
```

Expected: same gate output as before + "Wrote 4 viz PNGs to ...".

- [ ] **Step 4: Eyeball the 4 PNGs**

```powershell
Start-Process experiments\012_week7_stdp_demo\outputs\weight_matrix_final.png
Start-Process experiments\012_week7_stdp_demo\outputs\weight_matrix_evolution.png
Start-Process experiments\012_week7_stdp_demo\outputs\tuning_curves.png
Start-Process experiments\012_week7_stdp_demo\outputs\spike_raster_late.png
```

Expected:
- Final weight matrix shows block structure: each selective column has high weights for ONE row group, low for the others.
- Evolution: trial 0 is noisy random; by trial 1000 the structure is clean.
- Tuning curves: at least 2 of 4 outputs have one bar much taller than the other two.
- Spike raster: output spikes line up with their preferred pattern's onset; mostly silent for other patterns.

- [ ] **Step 5: Commit viz code**

```powershell
git add experiments/012_week7_stdp_demo/run.py
git commit -m "exp 012: add 4 STDP-WTA visualizations + selectivity report"
```

---

## Task 5: Closeout

- [ ] **Step 1: Write `results.md` in the experiment folder**

Create `experiments/012_week7_stdp_demo/results.md`:

```markdown
# EXP-012 — STDP-WTA demo results

**Run date:** 2026-05-25
**Spec:** `docs/design/2026-05-25-stdp-wta-demo.md`
**Plan:** `docs/superpowers/plans/2026-05-25-stdp-wta-demo-implementation.md`

## Headline

[REPLACE: e.g. "3 of 4 outputs became selective for distinct patterns; 1 stayed silent. Weight std grew Nx, block structure visible from trial ~250."]

## Selectivity

[REPLACE WITH ACTUAL CONTENTS OF outputs/selectivity_report.txt]

## Notes

- Hyperparameters used as locked in the spec; no tuning needed if gate 4 passed on first run.
- If you HAD to tune (e.g. raise A_plus), note that here.

## Artifacts

- `outputs/selectivity_report.txt`
- `outputs/weight_matrix_evolution.png`
- `outputs/weight_matrix_final.png`
- `outputs/tuning_curves.png`
- `outputs/spike_raster_late.png`
```

Replace the `[REPLACE: ...]` sections with the actual values from the run.

- [ ] **Step 2: Closeout commit**

```powershell
git add experiments/012_week7_stdp_demo/results.md
git commit -m "exp 012: complete — STDP-WTA demo, N/4 outputs selective (see results.md)"
```

Replace `N` with the actual number of selective outputs.

---

## Self-review checklist

- [ ] All 5 gates from spec §8 ran (or were waived with explanation)
- [ ] 4 viz PNGs + selectivity_report.txt produced
- [ ] Source committed in scoped commits matching messages
- [ ] `outputs/` is gitignored (existing rule)
- [ ] Actual selectivity result recorded in closeout commit message
- [ ] `results.md` does not still contain `[REPLACE: ...]` placeholders
