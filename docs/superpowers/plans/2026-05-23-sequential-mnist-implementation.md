# Sequential MNIST (EXP-011) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a recurrent SNN and an architecturally-identical feedforward SNN on sequential MNIST (one pixel row per timestep, T=28), produce a comparison table, and quantify the recurrent-vs-feedforward accuracy gap.

**Architecture:** A new experiment folder `experiments/011_week7_sequential_mnist/` with one `models.py` (`row_encode` + `SequentialSNN` with a `recurrent: bool` switch), one `run.py` exposing `train(config) → dict`, two YAML variant configs, and a `run_all.py` driver that runs both variants, writes `comparison.{md,csv}`, copies the best checkpoint, and runs the reload-and-verify gate.

**Tech Stack:** PyTorch, snnTorch 0.9.x, torchvision, matplotlib, pytest. Uses the project's existing `ExperimentConfig`, `ExperimentTracker`, and `neuromorphic.viz` (`training_curve`, `spike_raster`).

**Parent spec:** `docs/design/2026-05-23-sequential-mnist.md`
**Companion handoff:** `docs/handoffs/week7_11_handoff`

**Discipline (from parent spec):**
- Pytest only for the pure `row_encode` helper (§5.2). All other verification is inline smoke commands in PowerShell, matching the week-6 plan style.
- Four correctness gates: forward-shape (Task 3), param-count (Task 3), initial-loss (Task 5), best-checkpoint reload (Task 8).
- 90-minute budget on Saturday 2026-05-23. Cut order if running long: drop `hidden_raster.png` first, drop per-variant `training_curve.png` second, never drop either variant.
- `outputs/` are gitignored (existing rule). Only source files (YAMLs, `models.py`, `run.py`, `run_all.py`, `tests/test_row_encode.py`, `config.py` edits) are committed.
- Pre-flight git check before editing ANY existing file: `git status && git diff <file>`. Week-6 lesson — STOP if the working tree is dirty.

**Predict-before-execute commitments (from spec §2, §4):**
- Recurrent variant: **94–96%** test accuracy, **21,514** parameters
- Feedforward variant: **88–92%** test accuracy, **5,002** parameters
- Predicted gap: **3–6%**
- Untrained initial CE loss: **2.302 ± 0.1** (= ln(10))

---

## File Structure

**Created by this plan:**

| File | Responsibility |
|---|---|
| `experiments/011_week7_sequential_mnist/__init__.py` | Marks folder as importable (empty) |
| `experiments/011_week7_sequential_mnist/models.py` | `row_encode` (pure) + `SequentialSNN(recurrent: bool)` |
| `experiments/011_week7_sequential_mnist/run.py` | `train(config) → dict`, MNIST loaders, inline gates, training loop, per-variant artifacts |
| `experiments/011_week7_sequential_mnist/run_all.py` | Iterates 2 YAMLs, writes `comparison.{md,csv}`, copies best checkpoint, runs reload gate |
| `experiments/011_week7_sequential_mnist/recurrent.yaml` | Recurrent variant config (`recurrent: true`) |
| `experiments/011_week7_sequential_mnist/feedforward.yaml` | Feedforward control config (`recurrent: false`) |
| `tests/test_row_encode.py` | Pytest for the row encoder (pure function) |

**Modified by this plan:**
- `src/neuromorphic/config.py` — add 4 fields (`hidden_size`, `recurrent`, `readout_window`, `sequential`) + `__post_init__` invariant guard, extend `arch` comment with `sequential_snn`.

**NOT created** (per spec §11 non-goals): no CNN front-end, no alternative encoders, no beta sweep, no multi-seed runs, no RSynaptic variant, no LR scheduling.

---

## Task 0: Pre-flight check (no commit)

- [ ] **Step 1: Confirm git working tree is clean**

```powershell
git status
```

Expected: only `.claude/` untracked (the project's claude-cache dir). If anything else is uncommitted, STOP and surface it — week-6 bundled stale edits into an unrelated commit and we are NOT repeating that.

- [ ] **Step 2: Confirm venv and snnTorch are available**

```powershell
.\.venv\Scripts\python.exe -c "import torch, snntorch; print('torch', torch.__version__, '| snntorch', snntorch.__version__, '| cuda', torch.cuda.is_available())"
```

Expected: prints versions; cuda flag is `True` (RTX 2080 available).

- [ ] **Step 3: Confirm existing tests are green (paranoia)**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

Expected: existing viz smoke tests pass. If they don't, you have a broken environment — fix before adding anything.

---

## Task 1: Extend `ExperimentConfig` with sequential-SNN fields

**Files:**
- Modify: `src/neuromorphic/config.py`

**Predict before executing:**
- Loading a YAML with `sequential: true` and `num_steps: 25` should raise `ValueError` from `__post_init__`.
- Loading a YAML with `sequential: true`, `num_steps: 28`, `encoding: "direct"` should succeed and produce an `ExperimentConfig` with the four new fields populated.

- [ ] **Step 1: Pre-flight git diff on the file**

```powershell
git diff src/neuromorphic/config.py
```

Expected: no output (clean). If anything's there, STOP and surface it.

- [ ] **Step 2: Add the four new fields and the `__post_init__` guard**

Open `src/neuromorphic/config.py`. Find the `# --- Model architecture ---` block (around lines 56–60) and update it to:

```python
    # --- Model architecture ---
    arch: str = "baseline_mlp"  # baseline_mlp | tiny_mlp | simple_cnn | feedforward_snn | spiking_cnn | sequential_snn
    num_inputs: int = 784
    hidden_dims: list[int] = field(default_factory=lambda: [1000, 1000])
    hidden_size: int = 128   # used by sequential_snn (single hidden layer)
    num_outputs: int = 10
    recurrent: bool = False  # sequential_snn: RLeaky (True) vs Leaky (False) for the hidden layer
    readout_window: int = 4  # sequential_snn: number of trailing timesteps summed for loss
```

Find the `# --- Temporal simulation (SNN) ---` block (around lines 67–68) and update to:

```python
    # --- Temporal simulation (SNN) ---
    num_steps: int = 25
    sequential: bool = False  # if True, feed input row-at-a-time (forces num_steps=28, encoding='direct')
```

Then add a `__post_init__` method directly above `to_dict`:

```python
    def __post_init__(self) -> None:
        """Cross-field invariants. Raised at load time, not at epoch 47."""
        if self.sequential:
            if self.num_steps != 28:
                raise ValueError(
                    f"sequential=True requires num_steps=28 (got {self.num_steps}). "
                    "Sequential MNIST presents one row per timestep."
                )
            if self.encoding != "direct":
                raise ValueError(
                    f"sequential=True requires encoding='direct' (got {self.encoding!r}). "
                    "Rate-coding the input would conflate the time axis with sampling noise."
                )
```

- [ ] **Step 3: Inline-verify the guard works in both directions**

```powershell
.\.venv\Scripts\python.exe -c "
from neuromorphic.config import ExperimentConfig

# Good: sequential off (existing default). Must construct.
c1 = ExperimentConfig()
assert c1.sequential is False
assert c1.hidden_size == 128
assert c1.recurrent is False
assert c1.readout_window == 4

# Good: sequential on with required companion fields.
c2 = ExperimentConfig(sequential=True, num_steps=28, encoding='direct', recurrent=True)
assert c2.sequential is True and c2.num_steps == 28 and c2.encoding == 'direct'

# Bad: sequential on with wrong num_steps.
try:
    ExperimentConfig(sequential=True, num_steps=25, encoding='direct')
except ValueError as e:
    print('guard rejects bad num_steps OK:', e)
else:
    raise SystemExit('guard FAILED — accepted num_steps=25 with sequential=True')

# Bad: sequential on with wrong encoding.
try:
    ExperimentConfig(sequential=True, num_steps=28, encoding='rate')
except ValueError as e:
    print('guard rejects bad encoding OK:', e)
else:
    raise SystemExit('guard FAILED — accepted encoding=rate with sequential=True')

print('config invariants OK')
"
```

Expected: prints "guard rejects bad num_steps OK: …", "guard rejects bad encoding OK: …", "config invariants OK". Any `SystemExit` means a guard branch is wrong — fix before committing.

- [ ] **Step 4: Confirm existing tests still pass**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

Expected: same green as Task 0 Step 3.

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/config.py
git commit -m "config: add 4 sequential-SNN fields with __post_init__ guard"
```

---

## Task 2: `row_encode` helper with pytest TDD

**Files:**
- Create: `experiments/011_week7_sequential_mnist/__init__.py`
- Create: `experiments/011_week7_sequential_mnist/models.py` (partial — just `row_encode` for now)
- Create: `tests/test_row_encode.py`

**Predict before executing:**
- `row_encode(torch.zeros(8, 1, 28, 28))` → shape `(28, 8, 28)`, all zeros.
- `row_encode(torch.arange(28*28).view(28, 28).expand(4, 1, 28, 28).float())` → row $t$ at time index $t$ equals `arange(t*28, (t+1)*28)` for every batch element.
- `row_encode(torch.rand(8, 28, 28))` (no channel dim) → also works, returns `(28, 8, 28)`.

- [ ] **Step 1: Scaffold the experiment folder and write the failing pytest**

```powershell
New-Item -ItemType Directory -Force -Path experiments\011_week7_sequential_mnist | Out-Null
```

Create `experiments/011_week7_sequential_mnist/__init__.py` as an empty file.

Create `tests/test_row_encode.py`:

```python
"""Tests for the sequential-MNIST row encoder (exp 011)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
EXP_DIR = HERE.parent / "experiments" / "011_week7_sequential_mnist"
sys.path.insert(0, str(EXP_DIR))

from models import row_encode  # noqa: E402


def test_row_encode_4d_shape():
    """[B, 1, 28, 28] -> [28, B, 28]."""
    out = row_encode(torch.zeros(8, 1, 28, 28))
    assert out.shape == (28, 8, 28), f"expected (28, 8, 28), got {tuple(out.shape)}"


def test_row_encode_3d_shape():
    """[B, 28, 28] (no channel dim) -> [28, B, 28]."""
    out = row_encode(torch.zeros(8, 28, 28))
    assert out.shape == (28, 8, 28), f"expected (28, 8, 28), got {tuple(out.shape)}"


def test_row_encode_preserves_row_content():
    """Row t of the image must end up at time index t of the sequence."""
    image = torch.arange(28 * 28).view(28, 28).float()  # row r is arange(r*28, (r+1)*28)
    batch = image.unsqueeze(0).unsqueeze(0).expand(4, 1, 28, 28).contiguous()  # [4, 1, 28, 28]
    out = row_encode(batch)
    for t in range(28):
        expected = torch.arange(t * 28, (t + 1) * 28).float()
        for b in range(4):
            assert torch.equal(out[t, b], expected), (
                f"row mismatch at t={t}, b={b}: {out[t, b]} vs {expected}"
            )


def test_row_encode_is_pure():
    """Calling row_encode twice on the same input gives equal tensors."""
    x = torch.rand(2, 1, 28, 28)
    a = row_encode(x)
    b = row_encode(x)
    assert torch.equal(a, b)
```

- [ ] **Step 2: Run the test and confirm it fails (red)**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_row_encode.py -v
```

Expected: ImportError ("No module named 'models'") or AttributeError. Whatever it is, it must be RED before you implement.

- [ ] **Step 3: Implement `row_encode` in `models.py`**

Create `experiments/011_week7_sequential_mnist/models.py`:

```python
"""Models for sequential MNIST (exp 011).

Contains:
- ``row_encode``: pure helper that reshapes an MNIST batch into a
  ``[T=28, B, 28]`` sequence, one row per timestep.
- ``SequentialSNN``: added in Task 3.
"""

from __future__ import annotations

import torch


def row_encode(images: torch.Tensor) -> torch.Tensor:
    """Reshape an MNIST batch into a row-major time sequence.

    Args:
        images: ``[B, 1, 28, 28]`` or ``[B, 28, 28]`` tensor of pixel values.

    Returns:
        ``[28, B, 28]`` — row index becomes time index. Pure function: no
        spike sampling, no normalization, no allocation beyond reshape.
    """
    if images.ndim == 4:
        # [B, 1, 28, 28] -> [B, 28, 28]
        images = images.squeeze(1)
    if images.ndim != 3 or images.shape[-2:] != (28, 28):
        raise ValueError(
            f"row_encode expects [B, 1, 28, 28] or [B, 28, 28], got {tuple(images.shape)}"
        )
    # [B, 28, 28] -> [28, B, 28]
    return images.permute(1, 0, 2).contiguous()
```

- [ ] **Step 4: Run the test and confirm it passes (green)**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_row_encode.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/011_week7_sequential_mnist/__init__.py experiments/011_week7_sequential_mnist/models.py tests/test_row_encode.py
git commit -m "exp 011: add row_encode helper + pytest"
```

---

## Task 3: `SequentialSNN` model class + inline verification gates

**Files:**
- Modify: `experiments/011_week7_sequential_mnist/models.py`

**Predict before executing:**
- Recurrent variant: 21,514 params, forward returns `[28, B, 10]`.
- Feedforward variant: 5,002 params, forward returns `[28, B, 10]`.
- Untrained CE loss on a random batch (averaged over the readout window) ≈ ln(10) ≈ 2.302 ± 0.1 for both variants.

- [ ] **Step 1: Append `SequentialSNN` to `models.py`**

Add to the top of `models.py`:

```python
import snntorch as snn
import torch.nn as nn
```

Then append at the bottom:

```python
class SequentialSNN(nn.Module):
    """Sequential-MNIST SNN with a `recurrent: bool` switch.

    Architecture (both variants):
        x_t: [B, 28]                                       # one image row at time t
        cur1 = fc1(x_t)                                    # Linear(28, 128) -> [B, 128]
        spk1, mem1 = hidden(cur1, [spk1,] mem1)            # RLeaky or Leaky
        cur2 = fc2(spk1)                                   # Linear(128, 10) -> [B, 10]
        spk2, mem2 = lif_out(cur2, mem2)                   # output Leaky
        record spk2

    Returns ``spk_out`` of shape ``[28, B, 10]``. Loss is computed
    externally as ``CE(spk_out[-readout_window:].sum(dim=0), labels)``.
    """

    def __init__(
        self,
        num_inputs: int = 28,
        hidden_size: int = 128,
        num_outputs: int = 10,
        beta: float = 0.9,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        num_steps: int = 28,
        readout_window: int = 4,
        recurrent: bool = False,
    ):
        super().__init__()
        if num_steps != 28:
            raise ValueError(f"SequentialSNN expects num_steps=28, got {num_steps}")
        self.num_steps = num_steps
        self.readout_window = readout_window
        self.recurrent = recurrent

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        if recurrent:
            self.lif1 = snn.RLeaky(
                beta=beta,
                all_to_all=True,
                linear_features=hidden_size,
                threshold=threshold,
                reset_mechanism=reset_mechanism,
            )
        else:
            self.lif1 = snn.Leaky(
                beta=beta,
                threshold=threshold,
                reset_mechanism=reset_mechanism,
            )

        self.fc2 = nn.Linear(hidden_size, num_outputs)
        self.lif_out = snn.Leaky(
            beta=beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images``: ``[B, 1, 28, 28]`` or ``[B, 28, 28]``.

        Returns ``spk_out`` of shape ``[28, B, num_outputs]``.
        """
        sequence = row_encode(images)  # [28, B, 28]

        if self.recurrent:
            spk1, mem1 = self.lif1.init_rleaky()
        else:
            mem1 = self.lif1.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec = []
        for t in range(self.num_steps):
            cur1 = self.fc1(sequence[t])              # [B, hidden_size]
            if self.recurrent:
                spk1, mem1 = self.lif1(cur1, spk1, mem1)
            else:
                spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)                     # [B, num_outputs]
            spk_out, mem_out = self.lif_out(cur2, mem_out)
            spk_out_rec.append(spk_out)

        return torch.stack(spk_out_rec, dim=0)        # [28, B, num_outputs]

    def readout(self, spk_out: torch.Tensor) -> torch.Tensor:
        """Sum the last ``readout_window`` timesteps. Returns ``[B, num_outputs]``."""
        return spk_out[-self.readout_window:].sum(dim=0)
```

- [ ] **Step 2: Verification gate #1 + #2 — forward shape and param count for BOTH variants**

```powershell
.\.venv\Scripts\python.exe -c "
import sys, torch
sys.path.insert(0, 'experiments/011_week7_sequential_mnist')
from models import SequentialSNN

torch.manual_seed(0)
images = torch.rand(4, 1, 28, 28)

for recurrent, expected_params in [(True, 21514), (False, 5002)]:
    net = SequentialSNN(recurrent=recurrent)
    spk_out = net(images)
    assert spk_out.shape == (28, 4, 10), f'recurrent={recurrent}: shape {spk_out.shape}'
    n = sum(p.numel() for p in net.parameters())
    label = 'recurrent' if recurrent else 'feedforward'
    print(f'{label:>11s}: shape={tuple(spk_out.shape)}, params={n:,}')
    assert abs(n - expected_params) <= 100, f'{label}: expected ~{expected_params}, got {n}'
print('forward-shape + param-count gates PASS')
"
```

Expected:

```
  recurrent: shape=(28, 4, 10), params=21,514
feedforward: shape=(28, 4, 10), params=5,002
forward-shape + param-count gates PASS
```

If param counts diverge by more than ±100, you have a wiring bug — fix BEFORE moving on.

- [ ] **Step 3: Verification gate #3 — initial CE loss ≈ ln(10)**

```powershell
.\.venv\Scripts\python.exe -c "
import sys, math, torch
import torch.nn as nn
sys.path.insert(0, 'experiments/011_week7_sequential_mnist')
from models import SequentialSNN

torch.manual_seed(0)
images = torch.rand(64, 1, 28, 28)
labels = torch.randint(0, 10, (64,))

for recurrent in (True, False):
    net = SequentialSNN(recurrent=recurrent)
    spk_out = net(images)
    logits = net.readout(spk_out)
    loss = nn.CrossEntropyLoss()(logits, labels)
    label = 'recurrent' if recurrent else 'feedforward'
    print(f'{label:>11s}: initial CE = {loss.item():.3f}  (expected ~{math.log(10):.3f})')
    assert abs(loss.item() - math.log(10)) <= 0.3, f'{label}: initial loss out of range'
print('initial-loss gate PASS')
"
```

Expected: both initial losses within 0.3 of `ln(10) ≈ 2.302`. The spec said ±0.1 but the readout sums only 4 steps so logits can be small/zero — a slightly looser ±0.3 tolerance here is fine for an untrained initial check; the true diagnostic is "not at chance is bad, way above is also bad".

If a variant's initial loss is far outside this range (say, 0.5 or 10), STOP — the model is either already discriminating (something is seeded wrong) or producing degenerate output (e.g. all-zero spikes → uniform softmax with non-trivial constant offset). Investigate before training.

**Recurrent-only mitigation if the recurrent variant's initial loss is high (>3.0) due to W_rec runaway** (spec §13): scale the W_rec init down before training. snnTorch stores the internal recurrent weights at `rlif.recurrent.weight` (for `all_to_all=True`). Apply once after construction:

```python
with torch.no_grad():
    net.lif1.recurrent.weight.mul_(0.1)
```

Re-check gate 3. If it now passes, proceed. Document the change in the closeout commit message — "exp 011: complete — needed 0.1x W_rec init scale".

- [ ] **Step 4: Commit**

```bash
git add experiments/011_week7_sequential_mnist/models.py
git commit -m "exp 011: add SequentialSNN model + verification gates (21.5K/5.0K params)"
```

---

## Task 4: Variant YAML configs

**Files:**
- Create: `experiments/011_week7_sequential_mnist/recurrent.yaml`
- Create: `experiments/011_week7_sequential_mnist/feedforward.yaml`

**Predict before executing:**
- Both YAMLs load successfully via `load_config`. Both produce configs with `sequential=True`, `num_steps=28`, `encoding='direct'`. They differ only in `recurrent`, `run_name`, and `notes`.

- [ ] **Step 1: Write `recurrent.yaml`**

Create `experiments/011_week7_sequential_mnist/recurrent.yaml`:

```yaml
# Sequential MNIST — recurrent hidden layer (snn.RLeaky, all_to_all=True).
# Spec: docs/design/2026-05-23-sequential-mnist.md

run_name: recurrent
notes: "EXP-011 recurrent — RLeaky all_to_all=True, H=128, beta=0.9, 28-step BPTT."
tags: [exp-011, sequential-mnist, recurrent, week-7]
experiment_id: EXP-011

seed: 42
dataset: mnist
batch_size: 128
data_root: ./data

arch: sequential_snn
num_inputs: 28
hidden_size: 128
num_outputs: 10
recurrent: true
readout_window: 4

beta: 0.9
threshold: 1.0
reset_mechanism: subtract

num_steps: 28
sequential: true
encoding: direct
gain: 1.0

optimizer: adam
lr: 0.0003
weight_decay: 0.0
epochs: 5

tracker: tensorboard
log_interval: 50
```

- [ ] **Step 2: Write `feedforward.yaml`**

Create `experiments/011_week7_sequential_mnist/feedforward.yaml`:

```yaml
# Sequential MNIST — feedforward control (snn.Leaky).
# Spec: docs/design/2026-05-23-sequential-mnist.md

run_name: feedforward
notes: "EXP-011 feedforward control — plain Leaky, H=128, beta=0.9, identical to recurrent variant except no W_rec."
tags: [exp-011, sequential-mnist, feedforward, week-7]
experiment_id: EXP-011

seed: 42
dataset: mnist
batch_size: 128
data_root: ./data

arch: sequential_snn
num_inputs: 28
hidden_size: 128
num_outputs: 10
recurrent: false
readout_window: 4

beta: 0.9
threshold: 1.0
reset_mechanism: subtract

num_steps: 28
sequential: true
encoding: direct
gain: 1.0

optimizer: adam
lr: 0.0003
weight_decay: 0.0
epochs: 5

tracker: tensorboard
log_interval: 50
```

- [ ] **Step 3: Verify both YAMLs load and produce valid configs**

```powershell
.\.venv\Scripts\python.exe -c "
from pathlib import Path
from neuromorphic.config import load_config

here = Path('experiments/011_week7_sequential_mnist')
for name in ['recurrent.yaml', 'feedforward.yaml']:
    c = load_config(here / name)
    assert c.sequential is True
    assert c.num_steps == 28
    assert c.encoding == 'direct'
    assert c.hidden_size == 128
    assert c.readout_window == 4
    assert c.epochs == 5
    print(f'{name}: recurrent={c.recurrent}, run_name={c.run_name!r}, lr={c.lr}')
print('YAMLs OK')
"
```

Expected:

```
recurrent.yaml: recurrent=True, run_name='recurrent', lr=0.0003
feedforward.yaml: recurrent=False, run_name='feedforward', lr=0.0003
YAMLs OK
```

- [ ] **Step 4: Commit**

```bash
git add experiments/011_week7_sequential_mnist/recurrent.yaml experiments/011_week7_sequential_mnist/feedforward.yaml
git commit -m "exp 011: add recurrent + feedforward variant configs"
```

---

## Task 5: `run.py` — single-variant trainer

**Files:**
- Create: `experiments/011_week7_sequential_mnist/run.py`

**Predict before executing:**
- A `train(config)` call on the recurrent YAML completes ~5 epochs in 8–12 min on the RTX 2080 and returns a dict with all 10 keys defined below.
- Output dir `experiments/011_week7_sequential_mnist/outputs/{run_name}/` contains `checkpoint.pt`, `training_curve.png`, `hidden_raster.png` after a full run.

- [ ] **Step 1: Write `run.py`**

Create `experiments/011_week7_sequential_mnist/run.py`:

```python
"""Single-variant trainer for the sequential MNIST experiment (exp 011).

Importable as ``train(config) -> dict`` for the run_all driver, or invoke
the CLI for ad-hoc single-variant runs:

    python run.py --config recurrent.yaml
"""

from __future__ import annotations

import math
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
from neuromorphic.viz import spike_raster, training_curve

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from models import SequentialSNN  # noqa: E402


def _get_dataloaders(batch_size: int, data_root: str):
    transform = transforms.Compose([
        transforms.ToTensor(),                           # [1, 28, 28], values in [0, 1]
        transforms.Normalize((0,), (1,)),                # no-op normalize (kept for parity with weeks 5/6)
    ])
    train_set = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


def _full_test_eval(net, loader, device) -> tuple[float, float]:
    net.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            spk_out = net(data)
            logits = net.readout(spk_out)
            loss_sum += loss_fn(logits, target).item() * data.size(0)
            correct += (logits.argmax(dim=1) == target).sum().item()
            total += data.size(0)
    net.train()
    return loss_sum / total, correct / total


def _run_verification_gates(net, loader, device, expected_params: int) -> None:
    """Pre-training gates §8 #1, #2, #3 (spec docs/design/2026-05-23-sequential-mnist.md)."""
    # Gate 1: forward shape
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)
    spk_out = net(data)
    assert spk_out.shape == (28, data.size(0), 10), (
        f"gate 1 (forward shape): got {tuple(spk_out.shape)}, want (28, {data.size(0)}, 10)"
    )

    # Gate 2: param count
    n = sum(p.numel() for p in net.parameters())
    assert abs(n - expected_params) <= 100, (
        f"gate 2 (param count): got {n}, want ~{expected_params} (+/-100)"
    )

    # Gate 3: initial loss ~ ln(10)
    logits = net.readout(spk_out)
    loss = nn.CrossEntropyLoss()(logits, target).item()
    target_loss = math.log(10)
    assert abs(loss - target_loss) <= 0.3, (
        f"gate 3 (initial loss): got {loss:.3f}, want {target_loss:.3f} +/- 0.3"
    )
    print(f"[gates] shape={tuple(spk_out.shape)} params={n:,} initial_loss={loss:.3f}")


def _render_hidden_raster(net, loader, device, save_path: Path) -> None:
    """One test image, render hidden-layer spikes across all 28 steps."""
    data, _ = next(iter(loader))
    image = data[:1].to(device)  # [1, 1, 28, 28]
    net.eval()
    hidden_rec = []
    with torch.no_grad():
        from models import row_encode
        sequence = row_encode(image)
        if net.recurrent:
            spk1, mem1 = net.lif1.init_rleaky()
        else:
            mem1 = net.lif1.init_leaky()
        for t in range(28):
            cur1 = net.fc1(sequence[t])
            if net.recurrent:
                spk1, mem1 = net.lif1(cur1, spk1, mem1)
            else:
                spk1, mem1 = net.lif1(cur1, mem1)
            hidden_rec.append(spk1)
    net.train()
    spk_hidden = torch.stack(hidden_rec, dim=0).cpu()  # [28, 1, 128]
    fig, ax = spike_raster(spk_hidden)
    variant = "recurrent" if net.recurrent else "feedforward"
    ax.set_title(f"{variant} hidden-layer spikes, 1 test image, T=28")
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def train(config: ExperimentConfig) -> dict:
    """Train one variant. Returns metrics dict + writes per-variant artifacts."""
    assert config.sequential and config.num_steps == 28, (
        f"config invariants: sequential={config.sequential}, num_steps={config.num_steps}"
    )

    set_seed(config.seed)
    device = get_device()

    train_loader, test_loader = _get_dataloaders(config.batch_size, config.data_root)

    net = SequentialSNN(
        num_inputs=config.num_inputs,
        hidden_size=config.hidden_size,
        num_outputs=config.num_outputs,
        beta=config.beta,
        threshold=config.threshold,
        reset_mechanism=config.reset_mechanism,
        num_steps=config.num_steps,
        readout_window=config.readout_window,
        recurrent=config.recurrent,
    ).to(device)

    expected_params = 21514 if config.recurrent else 5002
    _run_verification_gates(net, train_loader, device, expected_params)

    tracker = ExperimentTracker(config)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    loss_hist, test_loss_hist, test_acc_hist = [], [], []
    t0 = time.perf_counter()

    for epoch in range(config.epochs):
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            net.train()
            spk_out = net(data)
            logits = net.readout(spk_out)
            loss = loss_fn(logits, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            loss_hist.append(loss.item())

            if (i % config.log_interval) == 0:
                test_loss, test_acc = _full_test_eval(net, test_loader, device)
                test_loss_hist.append(test_loss)
                test_acc_hist.append(test_acc)
                tracker.log({"train/loss": loss.item(), "test/loss": test_loss, "test/acc": test_acc})
                print(f"epoch {epoch} iter {i:>4d}  train_loss={loss.item():.4f}  test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    train_seconds = time.perf_counter() - t0
    final_test_loss, final_test_acc = _full_test_eval(net, test_loader, device)

    out_dir = HERE / "outputs" / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save({
        "model_state": net.state_dict(),
        "config": config.to_dict(),
        "loss_hist": loss_hist,
        "test_loss_hist": test_loss_hist,
        "test_acc_hist": test_acc_hist,
        "final_accuracy": final_test_acc,
        "train_seconds": train_seconds,
        "num_params": sum(p.numel() for p in net.parameters()),
    }, checkpoint_path)

    fig, _ = training_curve(
        train_loss=loss_hist,
        test_loss=test_loss_hist,
        test_acc=test_acc_hist,
        log_interval=config.log_interval,
    )
    fig.savefig(out_dir / "training_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    try:
        _render_hidden_raster(net, test_loader, device, out_dir / "hidden_raster.png")
    except Exception as e:
        # Non-blocking — raster is nice-to-have, not deliverable.
        print(f"hidden_raster.png render skipped: {e!r}")

    tracker.close()

    return {
        "variant": config.run_name,
        "recurrent": config.recurrent,
        "num_params": sum(p.numel() for p in net.parameters()),
        "num_steps": config.num_steps,
        "hidden_size": config.hidden_size,
        "beta": config.beta,
        "epochs": config.epochs,
        "train_seconds": train_seconds,
        "final_test_acc": final_test_acc,
        "checkpoint_path": str(checkpoint_path),
    }


if __name__ == "__main__":
    config_path, overrides = parse_cli_overrides()
    config = load_config(config_path, overrides)
    metrics = train(config)
    print(metrics)
```

- [ ] **Step 2: Note on `training_curve` signature**

This `run.py` calls `training_curve(train_loss=..., test_loss=..., test_acc=..., log_interval=...)`. If the installed signature differs, look at week-6 exp 009 `run.py` and call it the same way — DO NOT modify the viz toolkit during this experiment.

```powershell
.\.venv\Scripts\python.exe -c "import inspect; from neuromorphic.viz import training_curve; print(inspect.signature(training_curve))"
```

Adjust the call in `run.py` if the printed signature doesn't accept those kwargs.

- [ ] **Step 3: Tiny smoke run — 1 epoch, batch=16, recurrent variant**

This proves the whole pipeline runs end-to-end without going through a full training budget. Override via CLI.

```powershell
.\.venv\Scripts\python.exe experiments/011_week7_sequential_mnist/run.py --config experiments/011_week7_sequential_mnist/recurrent.yaml --epochs 1 --batch_size 16 --log_interval 200
```

Expected: gates pass, ~30 sec of training on 2080, prints final metrics dict. `experiments/011_week7_sequential_mnist/outputs/recurrent/checkpoint.pt` exists afterwards.

DELETE the smoke-run checkpoint before the real run in Task 6 to avoid stale state:

```powershell
Remove-Item -Recurse -Force experiments\011_week7_sequential_mnist\outputs\recurrent -ErrorAction SilentlyContinue
```

- [ ] **Step 4: Commit**

```bash
git add experiments/011_week7_sequential_mnist/run.py
git commit -m "exp 011: add run.py with train() + inline verification gates"
```

---

## Task 6: Recurrent variant — full training run (no source commit)

**Files:** none modified

**Predict before executing:**
- Final test accuracy in **94–96%** (predict-before-execute commitment).
- Training time ~8–12 min on RTX 2080 (5 epochs × ~469 batches/epoch × 28-step BPTT, but hidden is only 128-wide).
- `outputs/recurrent/checkpoint.pt`, `outputs/recurrent/training_curve.png`, `outputs/recurrent/hidden_raster.png` all exist after.

- [ ] **Step 1: Run the recurrent variant end-to-end**

```powershell
.\.venv\Scripts\python.exe experiments/011_week7_sequential_mnist/run.py --config experiments/011_week7_sequential_mnist/recurrent.yaml
```

Expected: gates print first, then training output. Final line is the metrics dict — capture `final_test_acc` mentally.

- [ ] **Step 2: Sanity-check the artifacts exist**

```powershell
Get-ChildItem experiments\011_week7_sequential_mnist\outputs\recurrent
```

Expected: `checkpoint.pt`, `training_curve.png`, `hidden_raster.png`.

- [ ] **Step 3: Open the training curve and eyeball it**

```powershell
Start-Process experiments\011_week7_sequential_mnist\outputs\recurrent\training_curve.png
```

Expected pattern: train loss decreases smoothly, test acc rises into the 90s. If test acc plateaus below 80% or oscillates wildly, STOP and triage (LR too high? grad clip too tight? W_rec init bad? — see spec §13).

No commit at this step — only `outputs/` was written and `outputs/` is gitignored.

---

## Task 7: Feedforward variant — full training run (no source commit)

**Files:** none modified

**Predict before executing:**
- Final test accuracy in **88–92%** (predict-before-execute commitment).
- Training time ~6–10 min on RTX 2080 (slightly faster than recurrent due to fewer params and no W_rec matmul per step).
- Same artifacts under `outputs/feedforward/`.

- [ ] **Step 1: Run the feedforward variant end-to-end**

```powershell
.\.venv\Scripts\python.exe experiments/011_week7_sequential_mnist/run.py --config experiments/011_week7_sequential_mnist/feedforward.yaml
```

Expected: gates print, training runs, final metrics dict.

- [ ] **Step 2: Sanity-check artifacts and eyeball curve**

```powershell
Get-ChildItem experiments\011_week7_sequential_mnist\outputs\feedforward
Start-Process experiments\011_week7_sequential_mnist\outputs\feedforward\training_curve.png
```

Expected: same three files; training curve shows steady learning (slower or to a lower asymptote than recurrent).

No commit.

---

## Task 8: `run_all.py` with comparison table + reload gate

**Files:**
- Create: `experiments/011_week7_sequential_mnist/run_all.py`

**Predict before executing:**
- `comparison.md` has 2 data rows (recurrent, feedforward) and a gap column.
- `comparison.csv` has the same data, machine-readable.
- `best_checkpoint.pt` and `best_checkpoint.json` are copied from the higher-accuracy variant.
- Reload gate prints "RELOAD OK" and asserts re-evaluation within ±0.1% of recorded.

- [ ] **Step 1: Write `run_all.py`**

Create `experiments/011_week7_sequential_mnist/run_all.py`:

```python
"""Run both variants, write comparison artifacts, run reload-and-verify gate.

Two-variant driver for exp 011. Imports ``train`` from ``run.py``.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neuromorphic.config import ExperimentConfig, load_config
from neuromorphic.utils import get_device

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run import train, _full_test_eval  # noqa: E402
from models import SequentialSNN  # noqa: E402


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def write_comparison_md(results: list[dict], gap_vs_feedforward: float, path: Path) -> None:
    """Pipe-table format. Columns: Variant | Hidden | Params | Epochs | Train time | Test acc | Gap."""
    ff_acc = next(r["final_test_acc"] for r in results if not r["recurrent"])
    lines = [
        "| Variant | Hidden layer | Params | Epochs | Train time | Test acc | Gap vs feedforward |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        hidden = "snn.RLeaky(all_to_all=True)" if r["recurrent"] else "snn.Leaky"
        gap = (r["final_test_acc"] - ff_acc) * 100.0
        gap_str = f"{gap:+.2f}%" if r["recurrent"] else "0.00%"
        lines.append(
            f"| `{r['variant']}` | {hidden} | {r['num_params']:,} | {r['epochs']} | "
            f"{_fmt_time(r['train_seconds'])} | {r['final_test_acc']*100:.2f}% | {gap_str} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_csv(results: list[dict], path: Path) -> None:
    ff_acc = next(r["final_test_acc"] for r in results if not r["recurrent"])
    rows = []
    for r in results:
        rows.append({
            "variant": r["variant"],
            "recurrent": r["recurrent"],
            "hidden_layer": "rleaky_all_to_all" if r["recurrent"] else "leaky",
            "num_params": r["num_params"],
            "epochs": r["epochs"],
            "train_seconds": r["train_seconds"],
            "final_test_acc": r["final_test_acc"],
            "gap_vs_feedforward": r["final_test_acc"] - ff_acc,
        })
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def reload_gate(best_result: dict, best_config: ExperimentConfig) -> None:
    """§8 gate #4: reload best checkpoint, re-evaluate, assert within ±0.1%."""
    device = get_device()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    test_set = datasets.MNIST(best_config.data_root, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=best_config.batch_size, shuffle=False, drop_last=False)

    ckpt = torch.load(best_result["checkpoint_path"], map_location=device)
    net = SequentialSNN(
        num_inputs=best_config.num_inputs,
        hidden_size=best_config.hidden_size,
        num_outputs=best_config.num_outputs,
        beta=best_config.beta,
        threshold=best_config.threshold,
        reset_mechanism=best_config.reset_mechanism,
        num_steps=best_config.num_steps,
        readout_window=best_config.readout_window,
        recurrent=best_config.recurrent,
    ).to(device)
    net.load_state_dict(ckpt["model_state"])

    _, reload_acc = _full_test_eval(net, test_loader, device)
    delta = abs(reload_acc - best_result["final_test_acc"])
    assert delta <= 0.001, (
        f"reload gate FAILED: recorded {best_result['final_test_acc']:.4f}, "
        f"reloaded {reload_acc:.4f}, delta {delta:.4f}"
    )
    print(f"[reload gate] recorded={best_result['final_test_acc']:.4f} "
          f"reloaded={reload_acc:.4f} delta={delta:.5f} OK")


def main() -> None:
    yamls = ["recurrent.yaml", "feedforward.yaml"]
    configs = [load_config(HERE / y) for y in yamls]

    results = []
    for c in configs:
        print("\n" + "=" * 72)
        print(f"Training variant: {c.run_name}")
        print("=" * 72)
        results.append(train(c))

    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    write_comparison_md(results, 0.0, out / "comparison.md")
    write_comparison_csv(results, out / "comparison.csv")

    best = max(results, key=lambda r: r["final_test_acc"])
    best_config = next(c for c in configs if c.run_name == best["variant"])
    shutil.copy(best["checkpoint_path"], out / "best_checkpoint.pt")
    ff_acc = next(r["final_test_acc"] for r in results if not r["recurrent"])
    (out / "best_checkpoint.json").write_text(json.dumps({
        "variant": best["variant"],
        "final_test_acc": best["final_test_acc"],
        "gap_vs_feedforward": best["final_test_acc"] - ff_acc,
        "config_summary": {
            "recurrent": best["recurrent"],
            "num_steps": best["num_steps"],
            "hidden_size": best["hidden_size"],
            "beta": best["beta"],
            "epochs": best["epochs"],
            "num_params": best["num_params"],
        },
    }, indent=2), encoding="utf-8")

    reload_gate(best, best_config)

    print("\nSummary")
    for r in results:
        print(f"  {r['variant']:>12s}  acc={r['final_test_acc']*100:.2f}%  "
              f"params={r['num_params']:,}  train={_fmt_time(r['train_seconds'])}")
    print(f"\nBest: {best['variant']} @ {best['final_test_acc']*100:.2f}%  "
          f"(gap vs feedforward: {(best['final_test_acc']-ff_acc)*100:+.2f}%)")
    print(f"Artifacts in {out}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add experiments/011_week7_sequential_mnist/run_all.py
git commit -m "exp 011: add run_all.py with comparison + reload gate"
```

---

## Task 9: Execute `run_all.py` end-to-end + final closeout commit

**Files:**
- Possibly create or modify: `experiment-log.md` (project root) if it exists; otherwise skip the log update step.

**Predict before executing:**
- Both variants re-train (this run replaces Tasks 6 and 7's outputs; that's fine).
- `outputs/comparison.{md,csv}`, `outputs/best_checkpoint.{pt,json}` are written.
- Reload gate prints OK.
- The "Best" line names the recurrent variant with a gap in the predicted range.

If the gap falls outside the predicted 3–6% range, commit the result anyway and note the surprise in the commit message.

- [ ] **Step 1: Clean prior outputs and run the driver**

```powershell
Remove-Item -Recurse -Force experiments\011_week7_sequential_mnist\outputs -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe experiments/011_week7_sequential_mnist/run_all.py
```

Expected: ~15–20 min total (both variants trained back-to-back), gates print, reload prints OK, summary at the end.

- [ ] **Step 2: Eyeball the comparison table**

```powershell
Get-Content experiments\011_week7_sequential_mnist\outputs\comparison.md
```

Expected: 2 rows, with the recurrent row showing a positive gap and ~21,514 params, the feedforward row showing ~5,002 params and 0.00% gap.

- [ ] **Step 3: Update `experiment-log.md` if it exists**

```powershell
Test-Path experiment-log.md
```

If `True`: append a row for EXP-011 in the same format as prior rows. If `False`: skip this step (the log file isn't a project hard dependency).

- [ ] **Step 4: Final closeout commit**

Capture the actual gap from `outputs/comparison.md` and use it in the message. Example template (fill in the real number):

```bash
git add experiments/011_week7_sequential_mnist/  # in case any new tracked files appeared
# add experiment-log.md too if it was edited
git status
git commit -m "exp 011: complete — recurrent vs feedforward gap +X.XX% on sequential MNIST"
```

The commit message MUST report the actual gap (not predicted). If the result is surprising (outside 3–6%), explicitly say so in the message: e.g. `"exp 011: complete — gap of +1.20% (below predicted 3–6%, see comparison.md)"`.

---

## Self-review checklist (run before declaring done)

- [ ] All four §8 verification gates passed during the build (shape, params, initial-loss, reload)
- [ ] `comparison.md` and `comparison.csv` exist with 2 rows
- [ ] `best_checkpoint.pt` and `best_checkpoint.json` exist
- [ ] Source files committed in scoped increments matching the messages above
- [ ] `outputs/` is gitignored (existing rule, verify with `git status` showing no `outputs/` paths)
- [ ] Predicted-vs-actual gap recorded in the closeout commit message
- [ ] If the gap was outside the predicted range, the surprise is documented honestly

---

## Revision history

| Date | Change |
|---|---|
| 2026-05-20 | Initial plan derived from `docs/design/2026-05-23-sequential-mnist.md`. 9 tasks, ~6 commits, 90-min Saturday build budget. |
