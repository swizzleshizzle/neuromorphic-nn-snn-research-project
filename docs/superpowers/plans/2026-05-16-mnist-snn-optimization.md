# MNIST SNN Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train 3 SNN variants on MNIST (tuned MLP, baseline spiking CNN, beta-tuned spiking CNN), produce a comparison table, and ship a best-checkpoint reaching ≥95% test accuracy.

**Architecture:** A new experiment folder `experiments/009_week6_snn_mnist_optimization/` with one `models.py` (encoder + both architectures), one `run.py` exposing `train(config) → metrics_dict`, three YAML variant configs, and a `run_all.py` driver that iterates them and writes `comparison.{md,csv}` + `best_checkpoint.{pt,json}`. Per-variant training curves are rendered by the morning's `neuromorphic.viz.training_curve` (dogfood).

**Tech Stack:** PyTorch, snnTorch 0.9.4, torchvision, matplotlib. Uses the project's existing `ExperimentConfig`, `ExperimentTracker`, and the new `neuromorphic.viz` package.

**Parent spec:** `docs/design/2026-05-16-mnist-snn-optimization.md`
**Hard dependency:** the morning viz toolkit plan (`docs/superpowers/plans/2026-05-16-viz-toolkit-implementation.md`) must be merged and `pytest tests/viz/ -v` must report `24 passed` BEFORE Task 3 runs. `run.py` imports `neuromorphic.viz.training_curve`.

**Discipline (from parent spec):**
- No pytest tests for this experiment — verification is manual via inline smoke commands at the end of each task.
- Three correctness gates: V1 sanity (Task 4), CNN forward shape (Task 2), best-checkpoint reload (Task 8).
- `outputs/` are gitignored; only source files (YAMLs, `models.py`, `run.py`, `run_all.py`, `verify_best.py`) are committed.

---

## File Structure

**Created by this plan:**

| File | Responsibility |
|---|---|
| `experiments/009_week6_snn_mnist_optimization/__init__.py` | Marks folder as importable (empty) |
| `experiments/009_week6_snn_mnist_optimization/models.py` | `encode_rate` (shape-generic), `FeedforwardSNN` (lifted from week 5), `SpikingCNN` (new) |
| `experiments/009_week6_snn_mnist_optimization/run.py` | `train(config) → dict`, MNIST loaders, arch dispatch, per-variant artifacts |
| `experiments/009_week6_snn_mnist_optimization/run_all.py` | Iterates 3 YAMLs, writes comparison table, copies best checkpoint |
| `experiments/009_week6_snn_mnist_optimization/verify_best.py` | Loads `best_checkpoint.pt`, re-evaluates, asserts acc matches within ±0.1% |
| `experiments/009_week6_snn_mnist_optimization/v1_mlp_tuned.yaml` | V1 config: feedforward SNN, 5 epochs |
| `experiments/009_week6_snn_mnist_optimization/v2_cnn_baseline.yaml` | V2 config: spiking CNN, beta=0.95 |
| `experiments/009_week6_snn_mnist_optimization/v3_cnn_tuned.yaml` | V3 config: spiking CNN, beta=0.90 |

**Modified by this plan:**
- `src/neuromorphic/config.py` — extend `arch` field's comment to list `spiking_cnn`.

**NOT created** (per spec §11 non-goals): no pytest tests, no CNN-topology config fields, no alternative encoders/optimizers.

---

## Task 0: Folder scaffold and config update

**Files:**
- Create: `experiments/009_week6_snn_mnist_optimization/__init__.py`
- Modify: `src/neuromorphic/config.py:57`

- [ ] **Step 1: Confirm hard dependency — viz toolkit is ready**

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/ -q
.venv\Scripts\python.exe -c "from neuromorphic.viz import training_curve; print('viz ok')"
```

Expected: pytest reports `24 passed`. Second command prints `viz ok`. If either fails, STOP — go finish the viz toolkit plan first.

- [ ] **Step 2: Create the experiment folder**

```bash
mkdir -p experiments/009_week6_snn_mnist_optimization
```

Write `experiments/009_week6_snn_mnist_optimization/__init__.py` as an empty file.

- [ ] **Step 3: Update `ExperimentConfig.arch` field comment**

Edit `src/neuromorphic/config.py`. The current line 57 reads:

```python
    arch: str = "baseline_mlp"  # baseline_mlp | tiny_mlp | simple_cnn | feedforward_snn
```

Change it to:

```python
    arch: str = "baseline_mlp"  # baseline_mlp | tiny_mlp | simple_cnn | feedforward_snn | spiking_cnn
```

- [ ] **Step 4: Commit**

```bash
git add experiments/009_week6_snn_mnist_optimization/__init__.py src/neuromorphic/config.py
git commit -m "exp 009: scaffold optimization experiment, add spiking_cnn arch value"
```

---

## Task 1: `models.py` — encoder and `FeedforwardSNN`

**Files:**
- Create: `experiments/009_week6_snn_mnist_optimization/models.py`

**Predict before executing:**
- `encode_rate(torch.rand(8, 784), num_steps=10)` → shape `[10, 8, 784]`, values in {0, 1}.
- `encode_rate(torch.rand(8, 1, 28, 28), num_steps=10)` → shape `[10, 8, 1, 28, 28]`. This is the change vs. the week-5 version, which would crash here on a 4D input.
- `FeedforwardSNN(num_steps=10)(torch.rand(10, 4, 784))` → `(spk, mem)` each shape `[10, 4, 10]`.

- [ ] **Step 1: Write `models.py` with the encoder and FeedforwardSNN**

```python
"""Models for week-6 MNIST optimization (experiment 009).

Contains:
- ``encode_rate``: shape-generic rate encoder, works for both ``[B, 784]``
  (MLP input) and ``[B, 1, 28, 28]`` (CNN input).
- ``FeedforwardSNN``: lifted verbatim from week-5 ``run2.py`` so this
  experiment is self-contained.
- ``SpikingCNN``: added in Task 2.
"""

from __future__ import annotations

import snntorch as snn
import torch
import torch.nn as nn


def encode_rate(data: torch.Tensor, num_steps: int, gain: float = 1.0) -> torch.Tensor:
    """Rate-code an input tensor as Bernoulli spike trains.

    Generalized over input rank: returns ``[num_steps, *data.shape]`` with
    values in ``{0.0, 1.0}``. Works for ``[B, 784]`` and ``[B, 1, 28, 28]``
    alike (the week-5 version's ``.expand(num_steps, -1, -1)`` only handled
    2D input).

    Args:
        data: tensor of any rank, values clipped to ``[0, 1]`` after gain.
        num_steps: time steps to simulate.
        gain: multiplier on values before clipping (probability scaling).

    Returns:
        ``[num_steps, *data.shape]`` float tensor with 0/1 entries.
    """
    probs = torch.clamp(data * gain, min=0.0, max=1.0)
    probs_expanded = probs.unsqueeze(0).expand(num_steps, *([-1] * probs.ndim))
    spikes = (torch.rand_like(probs_expanded) < probs_expanded).float()
    return spikes


class FeedforwardSNN(nn.Module):
    """Three-layer feedforward spiking NN. Lifted verbatim from week 5.

    Architecture: ``num_inputs -> hidden_dims[0] -> hidden_dims[1] -> num_outputs``,
    each Linear layer followed by ``snn.Leaky``.
    """

    def __init__(
        self,
        num_inputs: int = 784,
        hidden_dims: tuple[int, int] = (1000, 1000),
        num_outputs: int = 10,
        beta: float = 0.95,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        num_steps: int = 25,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(num_inputs, hidden_dims[0])
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)
        self.fc3 = nn.Linear(hidden_dims[1], num_outputs)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)

    def forward(self, spk_in: torch.Tensor):
        """``spk_in``: ``[num_steps, B, num_inputs]``.
        Returns ``(spk_out, mem_out)`` each ``[num_steps, B, num_outputs]``."""
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk_out_rec, mem_out_rec = [], []
        for step in range(self.num_steps):
            cur1 = self.fc1(spk_in[step]); spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1);          spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2);          spk3, mem3 = self.lif3(cur3, mem3)
            spk_out_rec.append(spk3)
            mem_out_rec.append(mem3)
        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)
```

- [ ] **Step 2: Smoke-check the encoder on both input shapes**

```powershell
.venv\Scripts\python.exe -c "
import sys, torch
sys.path.insert(0, 'experiments/009_week6_snn_mnist_optimization')
from models import encode_rate

mlp_in = encode_rate(torch.rand(8, 784), num_steps=10)
assert mlp_in.shape == (10, 8, 784), mlp_in.shape
assert set(mlp_in.unique().tolist()).issubset({0.0, 1.0})

cnn_in = encode_rate(torch.rand(8, 1, 28, 28), num_steps=10)
assert cnn_in.shape == (10, 8, 1, 28, 28), cnn_in.shape

print('encode_rate OK: MLP', tuple(mlp_in.shape), '| CNN', tuple(cnn_in.shape))
"
```

Expected: `encode_rate OK: MLP (10, 8, 784) | CNN (10, 8, 1, 28, 28)`.

- [ ] **Step 3: Smoke-check FeedforwardSNN forward shape**

```powershell
.venv\Scripts\python.exe -c "
import sys, torch
sys.path.insert(0, 'experiments/009_week6_snn_mnist_optimization')
from models import FeedforwardSNN

net = FeedforwardSNN(num_steps=10)
spk_in = (torch.rand(10, 4, 784) < 0.2).float()
spk_out, mem_out = net(spk_in)
assert spk_out.shape == (10, 4, 10), spk_out.shape
assert mem_out.shape == (10, 4, 10), mem_out.shape
n = sum(p.numel() for p in net.parameters())
print(f'FeedforwardSNN OK, params={n:,}')
"
```

Expected: `FeedforwardSNN OK, params=1,796,010`.

- [ ] **Step 4: Commit**

```bash
git add experiments/009_week6_snn_mnist_optimization/models.py
git commit -m "exp 009: add models.py with encode_rate + FeedforwardSNN"
```

---

## Task 2: Add `SpikingCNN` to `models.py`

**Files:**
- Modify: `experiments/009_week6_snn_mnist_optimization/models.py`

**Predict before executing:**
- Topology produces a `(25, 8, 10)` output for a `(25, 8, 1, 28, 28)` input.
- Param count: 416 (conv1) + 12,832 (conv2) + 15,690 (fc) = 28,938 (snn.Leaky adds 0 with defaults).
- Broken signal: wrong feature-map size (forgot padding) → fc input dim mismatch → RuntimeError on the Linear layer.

- [ ] **Step 1: Append `SpikingCNN` to `models.py`**

Append at the bottom of `experiments/009_week6_snn_mnist_optimization/models.py`:

```python
class SpikingCNN(nn.Module):
    """Three-layer spiking CNN for MNIST.

    Topology (hardcoded — see design doc §3):
        Conv2d(1, 16, k=5, padding=2)  -> MaxPool2d(2) -> snn.Leaky    # 28x28 -> 14x14
        Conv2d(16, 32, k=5, padding=2) -> MaxPool2d(2) -> snn.Leaky    # 14x14 -> 7x7
        Flatten                                                          # -> 32*7*7 = 1568
        Linear(1568, num_outputs)                  -> snn.Leaky          # output
    """

    def __init__(
        self,
        num_outputs: int = 10,
        beta: float = 0.95,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        num_steps: int = 25,
    ):
        super().__init__()
        self.num_steps = num_steps

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)

        self.fc = nn.Linear(32 * 7 * 7, num_outputs)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)

    def forward(self, spk_in: torch.Tensor):
        """``spk_in``: ``[num_steps, B, 1, 28, 28]``.
        Returns ``(spk_out, mem_out)`` each ``[num_steps, B, num_outputs]``."""
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk_out_rec, mem_out_rec = [], []
        for step in range(self.num_steps):
            cur1 = self.pool1(self.conv1(spk_in[step]))     # [B, 16, 14, 14]
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.pool2(self.conv2(spk1))             # [B, 32, 7, 7]
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc(spk2.flatten(start_dim=1))       # [B, num_outputs]
            spk3, mem3 = self.lif3(cur3, mem3)
            spk_out_rec.append(spk3)
            mem_out_rec.append(mem3)
        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)
```

- [ ] **Step 2: Verification gate #2 — CNN forward shape and param count**

```powershell
.venv\Scripts\python.exe -c "
import sys, torch
sys.path.insert(0, 'experiments/009_week6_snn_mnist_optimization')
from models import SpikingCNN, encode_rate

net = SpikingCNN(num_outputs=10, beta=0.95, num_steps=25)
data = torch.rand(8, 1, 28, 28)
spk_in = encode_rate(data, num_steps=25)
print('input shape:', tuple(spk_in.shape))
spk_out, mem_out = net(spk_in)
print('spk_out shape:', tuple(spk_out.shape))
print('mem_out shape:', tuple(mem_out.shape))
assert spk_out.shape == (25, 8, 10), spk_out.shape
assert mem_out.shape == (25, 8, 10), mem_out.shape
n = sum(p.numel() for p in net.parameters())
print(f'params={n:,}')
assert n == 28938, f'expected 28938, got {n}'
print('SpikingCNN OK')
"
```

Expected output:

```
input shape: (25, 8, 1, 28, 28)
spk_out shape: (25, 8, 10)
mem_out shape: (25, 8, 10)
params=28,938
SpikingCNN OK
```

If shapes diverge, you have an indexing/padding/pool bug — fix BEFORE training anything. This is the spec's verification gate §9 #2.

- [ ] **Step 3: Commit**

```bash
git add experiments/009_week6_snn_mnist_optimization/models.py
git commit -m "exp 009: add SpikingCNN (Conv-Pool-LIF x2 + Linear-LIF, 29K params)"
```

---

## Task 3: `run.py` — single-variant trainer

**Files:**
- Create: `experiments/009_week6_snn_mnist_optimization/run.py`

**Predict before executing:**
- A `train(config)` call on a tiny synthetic config (e.g., 1 epoch, batch=16) should return a dict with all 9 keys defined in spec §6, complete without errors, and write `outputs/{run_name}/checkpoint.pt` + `outputs/{run_name}/training_curve.png`.

- [ ] **Step 1: Write `run.py`**

Write `experiments/009_week6_snn_mnist_optimization/run.py`:

```python
"""Single-variant trainer for the MNIST SNN optimization experiment.

Importable as ``train(config) -> dict`` for the run_all driver, or invoke
the CLI for ad-hoc single-variant runs:

    python run.py --config v1_mlp_tuned.yaml
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neuromorphic.config import ExperimentConfig, load_config, parse_cli_overrides
from neuromorphic.tracking import ExperimentTracker
from neuromorphic.utils import get_device, set_seed
from neuromorphic.viz import training_curve

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from models import FeedforwardSNN, SpikingCNN, encode_rate  # noqa: E402


# ---- internal helpers ----------------------------------------------------

def _build_model(config: ExperimentConfig) -> nn.Module:
    if config.arch == "feedforward_snn":
        return FeedforwardSNN(
            num_inputs=config.num_inputs,
            hidden_dims=tuple(config.hidden_dims),
            num_outputs=config.num_outputs,
            beta=config.beta,
            threshold=config.threshold,
            reset_mechanism=config.reset_mechanism,
            num_steps=config.num_steps,
        )
    if config.arch == "spiking_cnn":
        return SpikingCNN(
            num_outputs=config.num_outputs,
            beta=config.beta,
            threshold=config.threshold,
            reset_mechanism=config.reset_mechanism,
            num_steps=config.num_steps,
        )
    raise ValueError(f"Unknown arch: {config.arch!r}")


def _get_dataloaders(batch_size: int, data_root: str):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])
    train_set = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


def _prep_input(data: torch.Tensor, arch: str) -> torch.Tensor:
    """MLP wants ``[B, 784]``; CNN wants ``[B, 1, 28, 28]`` (untouched)."""
    if arch == "feedforward_snn":
        return data.view(data.size(0), -1)
    return data


def _full_test_eval(net, loader, device, num_steps, gain, arch) -> tuple[float, float]:
    """Returns ``(avg_loss, accuracy)`` over the full loader."""
    net.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, targets in loader:
            data = _prep_input(data.to(device), arch)
            targets = targets.to(device)
            spk_in = encode_rate(data, num_steps=num_steps, gain=gain)
            spk_out, mem_out = net(spk_in)
            for step in range(num_steps):
                total_loss += loss_fn(mem_out[step], targets).item()
            _, predicted = spk_out.sum(dim=0).max(1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
    return total_loss / len(loader), total_correct / total_samples


# ---- public API ----------------------------------------------------------

def train(config: ExperimentConfig) -> dict:
    """Train one variant. Returns a metrics dict and writes per-variant artifacts.

    Side effects (in order):
        1. Builds model from ``config.arch``.
        2. Starts an ExperimentTracker (W&B + TensorBoard per the config).
        3. Trains for ``config.epochs``, timed with ``time.perf_counter``.
        4. Full-test-set eval.
        5. Saves ``checkpoint.pt`` to ``config.viz_output_dir``.
        6. Renders ``training_curve.png`` via ``neuromorphic.viz.training_curve``.
        7. Closes the tracker and returns.

    Returns:
        Dict with keys: variant, arch, num_params, num_steps, beta, epochs,
        train_seconds, final_test_acc, checkpoint_path.
    """
    set_seed(config.seed)
    device = get_device()

    out_dir = Path(config.viz_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = _get_dataloaders(config.batch_size, config.data_root)

    net = _build_model(config).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[{config.run_name}] arch={config.arch} | params={n_params:,} | "
          f"device={device} | epochs={config.epochs} | beta={config.beta} | "
          f"num_steps={config.num_steps}")

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    tracker = ExperimentTracker(config).start()

    loss_hist: list[float] = []
    test_loss_hist: list[float] = []
    test_acc_hist: list[float] = []
    global_step = 0
    start_time = time.perf_counter()

    for epoch in range(config.epochs):
        net.train()
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = _prep_input(data.to(device), config.arch)
            targets = targets.to(device)

            spk_in = encode_rate(data, num_steps=config.num_steps, gain=config.gain)
            spk_out, mem_out = net(spk_in)

            loss_val = torch.zeros(1, device=device)
            for step in range(config.num_steps):
                loss_val = loss_val + loss_fn(mem_out[step], targets)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            loss_hist.append(loss_val.item())
            tracker.log_metric("train/loss", loss_val.item(), step=global_step)

            if global_step % config.log_interval == 0:
                net.eval()
                with torch.no_grad():
                    td, tt = next(iter(test_loader))
                    td = _prep_input(td.to(device), config.arch)
                    tt = tt.to(device)
                    tsk_in = encode_rate(td, num_steps=config.num_steps, gain=config.gain)
                    tsk_out, tm_out = net(tsk_in)
                    test_loss = sum(
                        loss_fn(tm_out[s], tt).item() for s in range(config.num_steps)
                    )
                    _, test_pred = tsk_out.sum(dim=0).max(1)
                    test_acc = (test_pred == tt).float().mean().item()
                test_loss_hist.append(test_loss)
                test_acc_hist.append(test_acc)
                tracker.log_metrics(
                    {"test/loss": test_loss, "test/accuracy": test_acc},
                    step=global_step,
                )
                print(f"  epoch {epoch} iter {batch_idx} (step {global_step}): "
                      f"train_loss={loss_val.item():.2f} | "
                      f"test_acc={test_acc*100:.1f}%")
                net.train()

            global_step += 1

    train_seconds = time.perf_counter() - start_time

    final_loss, final_acc = _full_test_eval(
        net, test_loader, device, config.num_steps, config.gain, config.arch
    )
    print(f"[{config.run_name}] DONE: final_test_acc={final_acc*100:.2f}% | "
          f"train_time={train_seconds:.1f}s")
    tracker.log_metric("test/final_accuracy", final_acc, step=global_step)

    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": net.state_dict(),
            "config": config.to_dict(),
            "loss_hist": loss_hist,
            "test_loss_hist": test_loss_hist,
            "test_acc_hist": test_acc_hist,
            "final_accuracy": final_acc,
            "train_seconds": train_seconds,
        },
        ckpt_path,
    )

    history = {
        "train_loss": loss_hist,
        "test_loss": test_loss_hist,
        "test_acc": test_acc_hist,
    }
    fig, ax = training_curve(history, log_interval=config.log_interval)
    ax.set_title(f"{config.run_name} — final test acc {final_acc*100:.2f}%")
    fig.savefig(out_dir / "training_curve.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    tracker.finish()

    return {
        "variant": config.run_name,
        "arch": config.arch,
        "num_params": n_params,
        "num_steps": config.num_steps,
        "beta": config.beta,
        "epochs": config.epochs,
        "train_seconds": train_seconds,
        "final_test_acc": final_acc,
        "checkpoint_path": str(ckpt_path.resolve()),
    }


def main():
    config_path, overrides = parse_cli_overrides()
    config = load_config(config_path, overrides)
    metrics = train(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Import-smoke `run.py`**

```powershell
.venv\Scripts\python.exe -c "
import sys
sys.path.insert(0, 'experiments/009_week6_snn_mnist_optimization')
from run import train, _build_model, _prep_input
print('run.py imports OK; public train + 2 helpers found')
"
```

Expected: `run.py imports OK; public train + 2 helpers found`.

- [ ] **Step 3: Commit**

```bash
git add experiments/009_week6_snn_mnist_optimization/run.py
git commit -m "exp 009: add run.py train(config) with arch dispatch + viz dogfood"
```

---

## Task 4: V1 — tuned MLP

**Files:**
- Create: `experiments/009_week6_snn_mnist_optimization/v1_mlp_tuned.yaml`

**Predict before executing:**
- V1 = week-5 baseline + 2 extra epochs.
- Expected final test acc: 94–96% (week-5 baseline was 93.5% after 3 epochs).
- Expected train time on RTX 2080: ~5–8 minutes (week 5 finished 3 epochs in ~4 min).
- Broken signal: V1 acc below ~92% means the lifted code (encode_rate generalization, etc.) broke something — investigate before continuing.

- [ ] **Step 1: Write V1 YAML**

Write `experiments/009_week6_snn_mnist_optimization/v1_mlp_tuned.yaml`:

```yaml
# experiments/009_week6_snn_mnist_optimization/v1_mlp_tuned.yaml
# Variant 1: week-5 baseline MLP + 2 extra epochs.

run_name: "v1_mlp_tuned"
experiment_id: "EXP-002-V1"
tags:
  - phase-1
  - snn
  - mnist
  - optimization
  - mlp

seed: 42

arch: "feedforward_snn"
num_inputs: 784
hidden_dims: [1000, 1000]
num_outputs: 10

beta: 0.95
threshold: 1.0
reset_mechanism: "subtract"

num_steps: 25
encoding: "rate"
gain: 1.0

batch_size: 128
epochs: 5
optimizer: "adam"
lr: 5.0e-4

tracker: "both"
log_interval: 50
viz_output_dir: "experiments/009_week6_snn_mnist_optimization/outputs/v1_mlp_tuned"
```

- [ ] **Step 2: Run V1 training (long-running, ~7 minutes)**

```powershell
.venv\Scripts\python.exe experiments/009_week6_snn_mnist_optimization/run.py --config experiments/009_week6_snn_mnist_optimization/v1_mlp_tuned.yaml
```

Expected: per-iteration log lines; final line resembles `[v1_mlp_tuned] DONE: final_test_acc=94.xx% | train_time=400+s`. Then a JSON dump of the metrics dict.

- [ ] **Step 3: Verification gate #1 — V1 sanity**

Open the final accuracy printed in the previous step. Per spec §9 #1, V1 must be within ±0.5% of the week-5 baseline 93.5%, or higher. If V1 is below 93.0%, STOP and diagnose:
- Did `encode_rate` produce the right output? (Re-run Task 1 step 2 smoke.)
- Did the seed/lr/optimizer match the week-5 config? (Compare YAMLs side-by-side.)
- Did W&B log a sensible loss curve? (Open `runs/v1_mlp_tuned/` in TensorBoard or open the W&B run page.)

Do NOT continue to V2 until V1 passes the gate.

- [ ] **Step 4: Verify training_curve.png and checkpoint exist**

```powershell
.venv\Scripts\python.exe -c "
from pathlib import Path
d = Path('experiments/009_week6_snn_mnist_optimization/outputs/v1_mlp_tuned')
for f in ('checkpoint.pt', 'training_curve.png'):
    p = d / f
    assert p.exists(), p
    print(f'{p}: {p.stat().st_size:,} bytes')
"
```

Expected: both files print with non-zero sizes.

- [ ] **Step 5: Commit V1 YAML**

```bash
git add experiments/009_week6_snn_mnist_optimization/v1_mlp_tuned.yaml
git commit -m "exp 009 V1: tuned MLP (5 epochs) config"
```

---

## Task 5: V2 — spiking CNN baseline

**Files:**
- Create: `experiments/009_week6_snn_mnist_optimization/v2_cnn_baseline.yaml`

**Predict before executing:**
- V2 = spiking CNN with the same hyperparams as baseline (beta=0.95, num_steps=25, 3 epochs).
- Expected final test acc: 96–98% (CNN should comfortably beat MLP on MNIST).
- Expected train time on RTX 2080: ~10–15 minutes (CNN per-step compute is heavier than MLP but param count is much smaller; should be similar wall-clock).
- Broken signal: shape mismatch in forward (would crash); or accuracy below 90% (means CNN topology is mis-wired but doesn't crash).

- [ ] **Step 1: Write V2 YAML**

Write `experiments/009_week6_snn_mnist_optimization/v2_cnn_baseline.yaml`:

```yaml
# experiments/009_week6_snn_mnist_optimization/v2_cnn_baseline.yaml
# Variant 2: spiking CNN, baseline hyperparameters.

run_name: "v2_cnn_baseline"
experiment_id: "EXP-002-V2"
tags:
  - phase-1
  - snn
  - mnist
  - optimization
  - cnn

seed: 42

arch: "spiking_cnn"
num_outputs: 10

beta: 0.95
threshold: 1.0
reset_mechanism: "subtract"

num_steps: 25
encoding: "rate"
gain: 1.0

batch_size: 128
epochs: 3
optimizer: "adam"
lr: 5.0e-4

tracker: "both"
log_interval: 50
viz_output_dir: "experiments/009_week6_snn_mnist_optimization/outputs/v2_cnn_baseline"
```

- [ ] **Step 2: Run V2 training (long-running, ~12 minutes)**

```powershell
.venv\Scripts\python.exe experiments/009_week6_snn_mnist_optimization/run.py --config experiments/009_week6_snn_mnist_optimization/v2_cnn_baseline.yaml
```

Expected: per-iteration log lines; final acc should land in the 96–98% range. Final JSON dump of metrics.

- [ ] **Step 3: Sanity check V2 result**

Final accuracy must be ≥ V1's accuracy. If V2 < V1, something is wrong with the CNN — likely:
- Output `mem` shape mismatch causing loss to compute weird;
- Encoding shape wrong (re-verify Task 1 step 2);
- LIF state not resetting between samples.

If V2 ≥ 95%, the design goal is already met — V3 becomes "interesting comparison" rather than "needed to hit target".

- [ ] **Step 4: Verify V2 artifacts**

```powershell
.venv\Scripts\python.exe -c "
from pathlib import Path
d = Path('experiments/009_week6_snn_mnist_optimization/outputs/v2_cnn_baseline')
for f in ('checkpoint.pt', 'training_curve.png'):
    p = d / f
    assert p.exists(), p
    print(f'{p}: {p.stat().st_size:,} bytes')
"
```

Expected: both files print with non-zero sizes.

- [ ] **Step 5: Commit V2 YAML**

```bash
git add experiments/009_week6_snn_mnist_optimization/v2_cnn_baseline.yaml
git commit -m "exp 009 V2: spiking CNN baseline (beta=0.95) config"
```

---

## Task 6: V3 — spiking CNN with beta=0.90

**Files:**
- Create: `experiments/009_week6_snn_mnist_optimization/v3_cnn_tuned.yaml`

**Predict before executing:**
- V3 = same CNN as V2 with faster leak (beta=0.90 instead of 0.95).
- Expected final test acc: similar to V2, maybe ±0.5%. The hypothesis is that a faster leak makes per-step decisions crisper.
- Expected train time: same as V2 (~12 minutes) — only the leak constant changed.
- Broken signal: V3 ≪ V2 means the smaller beta is hurting (informative — note in the table).

- [ ] **Step 1: Write V3 YAML**

Write `experiments/009_week6_snn_mnist_optimization/v3_cnn_tuned.yaml`:

```yaml
# experiments/009_week6_snn_mnist_optimization/v3_cnn_tuned.yaml
# Variant 3: spiking CNN, faster leak (beta=0.90 vs 0.95).

run_name: "v3_cnn_tuned"
experiment_id: "EXP-002-V3"
tags:
  - phase-1
  - snn
  - mnist
  - optimization
  - cnn
  - beta-tuned

seed: 42

arch: "spiking_cnn"
num_outputs: 10

beta: 0.90
threshold: 1.0
reset_mechanism: "subtract"

num_steps: 25
encoding: "rate"
gain: 1.0

batch_size: 128
epochs: 3
optimizer: "adam"
lr: 5.0e-4

tracker: "both"
log_interval: 50
viz_output_dir: "experiments/009_week6_snn_mnist_optimization/outputs/v3_cnn_tuned"
```

- [ ] **Step 2: Run V3 training (long-running, ~12 minutes)**

```powershell
.venv\Scripts\python.exe experiments/009_week6_snn_mnist_optimization/run.py --config experiments/009_week6_snn_mnist_optimization/v3_cnn_tuned.yaml
```

Expected: log lines, final acc within ±1% of V2.

- [ ] **Step 3: Verify V3 artifacts**

```powershell
.venv\Scripts\python.exe -c "
from pathlib import Path
d = Path('experiments/009_week6_snn_mnist_optimization/outputs/v3_cnn_tuned')
for f in ('checkpoint.pt', 'training_curve.png'):
    p = d / f
    assert p.exists(), p
    print(f'{p}: {p.stat().st_size:,} bytes')
"
```

Expected: both files exist with non-zero sizes.

- [ ] **Step 4: Commit V3 YAML**

```bash
git add experiments/009_week6_snn_mnist_optimization/v3_cnn_tuned.yaml
git commit -m "exp 009 V3: spiking CNN with beta=0.90 config"
```

---

## Task 7: `run_all.py` — comparison table + best checkpoint

**Files:**
- Create: `experiments/009_week6_snn_mnist_optimization/run_all.py`

**Predict before executing:**
- Running `run_all.py` will **re-train all 3 variants from scratch** (~35 min total). For this task, we ONLY exercise the table-writer and best-checkpoint logic against the already-trained checkpoints — we do NOT run the full driver loop now (no time budget for that).
- `comparison.md` should have 3 rows; `comparison.csv` should have a header + 3 rows.
- `best_checkpoint.pt` should be a byte-identical copy of whichever variant won.

- [ ] **Step 1: Write `run_all.py`**

Write `experiments/009_week6_snn_mnist_optimization/run_all.py`:

```python
"""Drive all 3 variants and emit comparison artifacts.

Usage:
    python run_all.py                  # train all 3 fresh + write artifacts
    python run_all.py --tables-only    # skip training; rebuild tables from
                                       # already-saved per-variant checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from neuromorphic.config import load_config

VARIANT_YAMLS = ["v1_mlp_tuned.yaml", "v2_cnn_baseline.yaml", "v3_cnn_tuned.yaml"]


# ---- formatting helpers --------------------------------------------------

def _fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s"


def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}K"
    return str(n)


# ---- table writers -------------------------------------------------------

def write_comparison_md(results: list[dict], path: Path) -> None:
    """Pipe-table format matching the design doc's preview."""
    cols = ["Variant", "Arch", "Params", "num_steps", "beta",
            "Epochs", "Train time", "Test acc"]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in results:
        lines.append("| " + " | ".join([
            r["variant"],
            r["arch"],
            _fmt_params(r["num_params"]),
            str(r["num_steps"]),
            f'{r["beta"]:.2f}',
            str(r["epochs"]),
            _fmt_time(r["train_seconds"]),
            f'{r["final_test_acc"] * 100:.2f}%',
        ]) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_csv(results: list[dict], path: Path) -> None:
    fieldnames = ["variant", "arch", "num_params", "num_steps", "beta",
                  "epochs", "train_seconds", "final_test_acc"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})


# ---- result loading from saved checkpoints -------------------------------

def _result_from_checkpoint(yaml_path: Path) -> dict:
    """Rebuild a metrics dict from a previously-saved checkpoint.

    Used by ``--tables-only`` so we don't have to re-train to regenerate
    tables after a writer/format change.
    """
    config = load_config(yaml_path)
    ckpt_path = Path(config.viz_output_dir) / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run training first."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    return {
        "variant": cfg["run_name"],
        "arch": cfg["arch"],
        "num_params": _params_from_state(ckpt["model_state"]),
        "num_steps": cfg["num_steps"],
        "beta": cfg["beta"],
        "epochs": cfg["epochs"],
        "train_seconds": ckpt["train_seconds"],
        "final_test_acc": ckpt["final_accuracy"],
        "checkpoint_path": str(ckpt_path.resolve()),
    }


def _params_from_state(state_dict: dict) -> int:
    return sum(t.numel() for t in state_dict.values())


# ---- driver --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-only", action="store_true",
                        help="Skip training, rebuild tables from existing checkpoints.")
    args = parser.parse_args()

    out_dir = HERE / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tables_only:
        results = [_result_from_checkpoint(HERE / y) for y in VARIANT_YAMLS]
    else:
        from run import train  # local import so --tables-only doesn't need viz dep
        results = []
        for y in VARIANT_YAMLS:
            print(f"\n===== Training {y} =====")
            metrics = train(load_config(HERE / y))
            results.append(metrics)

    # Tables
    write_comparison_md(results, out_dir / "comparison.md")
    write_comparison_csv(results, out_dir / "comparison.csv")
    print(f"\nWrote {out_dir / 'comparison.md'}")
    print(f"Wrote {out_dir / 'comparison.csv'}")

    # Best checkpoint copy + sidecar
    best = max(results, key=lambda r: r["final_test_acc"])
    shutil.copy(best["checkpoint_path"], out_dir / "best_checkpoint.pt")
    with (out_dir / "best_checkpoint.json").open("w", encoding="utf-8") as f:
        json.dump({
            "variant": best["variant"],
            "final_test_acc": best["final_test_acc"],
            "config_summary": {
                "arch": best["arch"],
                "num_steps": best["num_steps"],
                "beta": best["beta"],
                "epochs": best["epochs"],
                "num_params": best["num_params"],
            },
        }, f, indent=2)
    print(f"\nBest: {best['variant']} @ {best['final_test_acc'] * 100:.2f}%")
    print(f"Copied to {out_dir / 'best_checkpoint.pt'}")

    print("\nFinal comparison:")
    print((out_dir / "comparison.md").read_text())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run `run_all.py --tables-only` to build artifacts from saved checkpoints**

```powershell
.venv\Scripts\python.exe experiments/009_week6_snn_mnist_optimization/run_all.py --tables-only
```

Expected: prints "Wrote …" lines for both tables and the best-checkpoint copy. Then prints the markdown comparison table. Best variant's accuracy should be the highest of the three.

- [ ] **Step 3: Inspect the table contents**

```powershell
.venv\Scripts\python.exe -c "
from pathlib import Path
d = Path('experiments/009_week6_snn_mnist_optimization/outputs')
print('--- comparison.md ---'); print((d / 'comparison.md').read_text())
print('--- comparison.csv ---'); print((d / 'comparison.csv').read_text())
print('--- best_checkpoint.json ---'); print((d / 'best_checkpoint.json').read_text())
"
```

Expected: markdown table renders 3 data rows; CSV has the same data with the column header row; JSON sidecar names the highest-accuracy variant.

- [ ] **Step 4: Commit `run_all.py`**

```bash
git add experiments/009_week6_snn_mnist_optimization/run_all.py
git commit -m "exp 009: add run_all.py with comparison.md/csv + best-checkpoint copy"
```

---

## Task 8: `verify_best.py` — best-checkpoint reload check

**Files:**
- Create: `experiments/009_week6_snn_mnist_optimization/verify_best.py`

**Predict before executing:**
- Re-evaluating the best checkpoint on the full test set must reproduce the recorded accuracy within ±0.1%.
- Larger divergence indicates a save/load bug (e.g., wrong arch reconstructed, batchnorm running stats lost — irrelevant here, but the principle stands).
- Broken signal: difference > 0.1% means the metric we wrote into the table was lying.

- [ ] **Step 1: Write `verify_best.py`**

Write `experiments/009_week6_snn_mnist_optimization/verify_best.py`:

```python
"""Reload the best checkpoint and re-evaluate on the full MNIST test set.

Asserts the reproduced accuracy matches the value recorded in
``best_checkpoint.json`` within ±0.1%. Implements verification gate §9 #3
of the design spec.

Usage:
    python verify_best.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from neuromorphic.utils import get_device
from models import FeedforwardSNN, SpikingCNN, encode_rate

OUT_DIR = HERE / "outputs"
TOLERANCE = 0.001  # 0.1%


def _build_model_from_dict(cfg: dict) -> torch.nn.Module:
    if cfg["arch"] == "feedforward_snn":
        return FeedforwardSNN(
            num_inputs=cfg["num_inputs"],
            hidden_dims=tuple(cfg["hidden_dims"]),
            num_outputs=cfg["num_outputs"],
            beta=cfg["beta"],
            threshold=cfg["threshold"],
            reset_mechanism=cfg["reset_mechanism"],
            num_steps=cfg["num_steps"],
        )
    if cfg["arch"] == "spiking_cnn":
        return SpikingCNN(
            num_outputs=cfg["num_outputs"],
            beta=cfg["beta"],
            threshold=cfg["threshold"],
            reset_mechanism=cfg["reset_mechanism"],
            num_steps=cfg["num_steps"],
        )
    raise ValueError(f"Unknown arch in checkpoint: {cfg['arch']!r}")


def _prep_input(data: torch.Tensor, arch: str) -> torch.Tensor:
    return data.view(data.size(0), -1) if arch == "feedforward_snn" else data


def main():
    device = get_device()
    meta = json.loads((OUT_DIR / "best_checkpoint.json").read_text())
    expected_acc = float(meta["final_test_acc"])
    print(f"Best variant: {meta['variant']!r}, expected_acc={expected_acc*100:.2f}%")

    ckpt = torch.load(
        OUT_DIR / "best_checkpoint.pt", map_location=device, weights_only=False
    )
    cfg = ckpt["config"]

    net = _build_model_from_dict(cfg).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])
    test_set = datasets.MNIST(cfg["data_root"], train=False, download=False,
                              transform=transform)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"],
                             shuffle=False, drop_last=False)

    arch = cfg["arch"]
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data = _prep_input(data.to(device), arch)
            targets = targets.to(device)
            spk_in = encode_rate(data, num_steps=cfg["num_steps"], gain=cfg["gain"])
            spk_out, _ = net(spk_in)
            _, predicted = spk_out.sum(dim=0).max(1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    reloaded_acc = total_correct / total_samples
    diff = abs(reloaded_acc - expected_acc)
    print(f"Reloaded acc: {reloaded_acc*100:.2f}%  ({total_correct}/{total_samples})")
    print(f"Diff vs. recorded: {diff*100:.3f}%  (tolerance: {TOLERANCE*100:.1f}%)")
    if diff > TOLERANCE:
        raise SystemExit(
            f"VERIFY FAILED: reloaded {reloaded_acc:.4f} vs recorded "
            f"{expected_acc:.4f} (|diff|={diff:.4f} > {TOLERANCE:.4f})"
        )
    print("VERIFY OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the verification**

```powershell
.venv\Scripts\python.exe experiments/009_week6_snn_mnist_optimization/verify_best.py
```

Expected: prints the best variant name, expected acc, reloaded acc, the diff, and a final `VERIFY OK` line. If `VERIFY FAILED` appears — investigate the checkpoint contents (likely an `arch` mismatch or state-dict key shift).

- [ ] **Step 3: Commit `verify_best.py`**

```bash
git add experiments/009_week6_snn_mnist_optimization/verify_best.py
git commit -m "exp 009: add verify_best.py reload-and-revalidate gate"
```

---

## Task 9: Eyeball pass and session-summary commit

**Files:**
- (No new files. Final visual checks + a summary commit if needed.)

- [ ] **Step 1: Eyeball the comparison table**

```powershell
.venv\Scripts\python.exe -c "
from pathlib import Path
print((Path('experiments/009_week6_snn_mnist_optimization/outputs') / 'comparison.md').read_text())
"
```

Confirm:
- Three rows, correct variant names and arch labels.
- Test acc values look monotonically informative (e.g., V1 ≤ V2 ≈ V3, or some sensible relationship).
- At least one row shows ≥95.0% (per spec §1 success criterion).

If all three are below 95%, the experiment didn't hit the target — that's still a valid outcome. Document the near-miss in the final commit message (see Step 4).

- [ ] **Step 2: Eyeball the three per-variant training curves**

Open these three PNGs in an image viewer:

```
experiments/009_week6_snn_mnist_optimization/outputs/v1_mlp_tuned/training_curve.png
experiments/009_week6_snn_mnist_optimization/outputs/v2_cnn_baseline/training_curve.png
experiments/009_week6_snn_mnist_optimization/outputs/v3_cnn_tuned/training_curve.png
```

For each, confirm: loss descends, test_acc ascends, twin axis ranges look right. Any flat curve or NaN region is a red flag — investigate the corresponding checkpoint.

- [ ] **Step 3: Confirm gitignored outputs aren't accidentally staged**

```bash
git status
```

Expected: clean working tree (everything committed) or untracked files only under `experiments/009_week6_snn_mnist_optimization/outputs/`. The `outputs/*` rule added in the morning's Task 0 should be hiding the artifacts. If `git status` shows `outputs/` files as untracked candidates, the `.gitignore` rule isn't catching them — fix the gitignore.

- [ ] **Step 4: Final session-summary commit (optional, no source changes)**

If the session is complete and you want a single commit message that summarizes the results, create an empty commit:

```bash
git commit --allow-empty -m "exp 009: complete — best variant <name> @ <acc>%, target 95% <met|missed>"
```

Replace `<name>`, `<acc>`, and `<met|missed>` with the actual values from `best_checkpoint.json`. Skip this step if you already captured the same info in an earlier commit message.

---

## Done criteria

The experiment is complete when:

1. `experiments/009_week6_snn_mnist_optimization/outputs/comparison.md` exists with 3 data rows.
2. `experiments/009_week6_snn_mnist_optimization/outputs/comparison.csv` exists with matching data, machine-readable.
3. `experiments/009_week6_snn_mnist_optimization/outputs/best_checkpoint.pt` and `best_checkpoint.json` exist.
4. `python verify_best.py` prints `VERIFY OK`.
5. At least one variant in the table shows ≥95.0% test accuracy. (If missed, the experiment is still "complete" — just document the gap in the final commit.)
6. Source files (`models.py`, `run.py`, `run_all.py`, `verify_best.py`, 3 YAMLs, `__init__.py`, the `config.py` update) are all committed; `outputs/` artifacts are not.
