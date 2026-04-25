# Neuromorphic NN / SNN Research Project

> A 12-month self-directed research project building a regionalized spiking neural network from first principles. **Learning in public** — every weekly note, design decision, and experimental result lives in this repo.

**Status:** Phase 0 (Foundations) — Week 3
**Capstone target:** *I Attempted to Build a Brain in 12 Months* (December 2026)

---

## What this is

A software-only neuromorphic AI research project. The goal is a regionalized spiking neural network — five distinct functional regions modeled loosely on the mammalian brain (sensory cortex, hippocampal memory, prefrontal planning, motor cortex, thalamic router) that learn to solve a 2x2 Rubik's Cube through experience rather than supervised training.

The wider context: most modern AI consumes megawatts. The brain runs general intelligence on ~20 watts. This project tests, at small scale, whether brain-inspired architecture can capture some fraction of that efficiency.

This repo is also the working surface of a self-directed curriculum. The author has a SCADA / industrial automation background (.NET, Ignition, signal processing) but no formal ML or neuroscience training. Every concept gets learned in the open — including the mistakes.

## What this isn't

- A polished framework. The code is the *output of learning*, not the input to anyone else's research.
- A claim that this will work. Several phase milestones may fail. Failures are documented as carefully as successes.
- A drop-in solution for any specific application. The Strategy Sentinel trading-system extension is on a separate, multi-year track and is not included in this repo.

---

## Project structure

```
src/neuromorphic/        # Reusable Python package — models, training, tracking, utils
experiments/             # One numbered subfolder per experiment (config + entry point)
notebooks/               # Exploratory Jupyter work
scripts/                 # One-off utility scripts
tests/                   # Pytest tests (added as code stabilizes)
docs/                    # Project docs (capstone phase)
data/                    # Datasets — gitignored, downloaded on first run
checkpoints/             # Model state_dicts — gitignored
runs/                    # TensorBoard logs — gitignored
wandb/                   # W&B local cache — gitignored
```

---

## Quickstart

Tested on Windows 11 with Python 3.11 and CUDA-capable GPU. Should also work CPU-only and on Linux/Mac.

### 1. Clone and create a virtual environment

```powershell
git clone https://github.com/swizzleshizzle/neuromorphic-nn-snn-research-project.git
cd neuromorphic-nn-snn-research-project

python -m venv .venv
.venv\Scripts\Activate.ps1
```

(macOS / Linux users: `source .venv/bin/activate`.)

### 2. Install PyTorch (with CUDA if you have a compatible GPU)

```powershell
# CUDA 12.1 — check https://pytorch.org/get-started/locally/ for your specific config
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Or CPU-only:
# pip install torch torchvision
```

### 3. Install the rest of the dependencies and the package itself

```powershell
pip install -r requirements.txt
pip install -e .
```

The `pip install -e .` installs the `neuromorphic` package in editable mode — code changes take effect immediately without reinstalling.

### 4. Run the smoke test

```powershell
python experiments/001_smoke_test/run.py --config experiments/001_smoke_test/config.yaml
```

If this prints `Smoke test passed.` and saves a checkpoint to `checkpoints/001_smoke_test.pt`, the scaffold is healthy.

---

## Experiment tracking

Two trackers are wired in and used together by default:

- **Weights & Biases** — cloud-hosted, best comparison UI. Sign up at [wandb.ai](https://wandb.ai), then run `wandb login` once.
- **TensorBoard** — local, no account, lives in `runs/`. Launch with `tensorboard --logdir runs/`.

Switch trackers per-experiment by editing `tracker:` in the config YAML — `wandb`, `tensorboard`, `both`, or `none`.

---

## Configuration system

Every experiment is described by a single YAML file in its `experiments/NNN_*/` folder. The schema is the `ExperimentConfig` dataclass in `src/neuromorphic/config.py`. Unknown fields fail loudly at load time — this prevents typos like `lerning_rate: 0.01` from silently using the default.

CLI overrides work for any field:

```powershell
python experiments/001_smoke_test/run.py --config experiments/001_smoke_test/config.yaml --lr 0.001 --epochs 10
```

---

## Phase roadmap

| Phase | Window | Focus |
|---|---|---|
| 0 — Foundations | Apr–May 2026 | NN concepts, PyTorch, snnTorch tutorials, RL basics |
| 1 — Single-region SNNs | May 2026 | Spiking MNIST, recurrence, surrogate gradients, STDP |
| 2 — Multi-region brain | Jun–Jul 2026 | Five-region architecture, inter-region communication, grid-world task |
| 3 — Rubik's Cube | Aug–Sep 2026 | 2x2 cube environment, curriculum learning, regional specialization |
| 4 — Capstone | Oct–Dec 2026 | Documentation, write-up, video, public release |

Future (Year 2+): **Strategy Sentinel** — applying the architecture as a meta-cognitive layer on top of an existing LEAN/QuantConnect quantitative trading stack. Separate repo, not included here.

---

## License

MIT. See `LICENSE`.

---

## Acknowledgments

Built with [snnTorch](https://snntorch.readthedocs.io/) and [PyTorch](https://pytorch.org/). Curriculum draws on Gerstner's *Neuronal Dynamics*, Sutton & Barto's *Reinforcement Learning*, the 3Blue1Brown neural network series, and snnTorch's tutorials.
