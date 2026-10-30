# Neuromorphic NN / SNN Research Project

> A 12-month self-directed research project building a regionalized spiking neural network from first principles. 

**Status:** Phase 4. 83 numbered experiments, 67 with a committed results file.
**Goal:** a spiking network that solves a randomly scrambled 2x2 cube.
**The write-up:** [`docs/writeup/report.md`](docs/writeup/report.md), the full research report, every number cited to the experiment that produced it.

---

## What this is

A software-only neuromorphic AI research project. The goal is a regionalized spiking neural network (five distinct functional regions modeled loosely on the mammalian brain: sensory cortex, hippocampal memory, prefrontal planning, motor cortex, thalamic router) that learns to solve a 2x2 Rubik's Cube through experience rather than supervised training.

The wider context: most modern AI consumes megawatts. The brain runs general intelligence on ~20 watts. This project tests, at small scale, whether brain-inspired architecture can capture some fraction of that efficiency.

This repo is also the working surface of a self-directed curriculum. The author has a SCADA / industrial automation background (.NET, Ignition, signal processing) but no formal ML or neuroscience training. Every concept gets learned in the open, including the mistakes.

## What this isn't

- A polished framework. The code is the *output of learning*, not the input to anyone else's research.
- A claim that this will work. Several phase milestones may fail. Failures are documented as carefully as successes.
- A drop-in solution for any specific application. The Strategy Sentinel trading-system extension is on a separate, multi-year track and is not included in this repo.

---

## Results so far

The cube task, 12 seeds, against a **measured** chance floor rather than an assumed one (it is
20.8% at depth 1, not 1/6, because a random walk inside a `2d+3` step budget can stumble into
solved). **The evaluation column matters: depth 1 is scored on the training distribution, every
depth from 3 down the table is scored on held-out states the policy never trained on.**

| scramble depth | success | evaluated on | chance floor | source |
|---|---|---|---|---|
| 1 | **87.5%** | train-dist | 20.8% | EXP-029, the v1 baseline |
| 3 | **50.0%** | held-out | 1.4% | EXP-035, up from 2.2% with no architectural change |
| 6 | **35.3%** | held-out | 0.1% | EXP-049 |
| 7 | **20.0%** | held-out | **0.0000** | EXP-053, replicated by EXP-060 |
| 8 | **7.8%** | held-out | **0.0000** | EXP-062, the frontier for a trained policy alone |

### Look-ahead around the trained networks (Phase 4)

The table above is the trained policy acting on reflex, one forward pass per move. Phase 4 asks
whether a search wrapped around the same networks goes further
(`docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md`). Every search below but the last four columns looks three
moves ahead and refuses to re-enter a state it has already visited; what differs is the judge that
scores the candidates:

| scramble depth | reflex policy | judged by the policy head | judged by a learned judge | judged by a wider learned judge | wider judge, 4-move tree | wider judge, 5-move beam | wider judge, 5-move tree | wider judge retrained on longer walks, 4-move tree | source |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 0.2067 | 0.3217 | 0.7908 | 0.8908 | not run | not run | not run | not run | EXP-071, EXP-074, EXP-075 |
| 8 | 0.0688 | 0.1496 | 0.3792 | 0.5158 | not run | not run | not run | not run | EXP-072, EXP-074, EXP-075 |
| 9 | 0.0171 | 0.0629 | 0.1933 | 0.2571 | 0.4004 | **0.5571** | 0.5558 | 0.3454 | EXP-072, EXP-074, EXP-075, EXP-076, EXP-077, EXP-078, EXP-079 |
| 11 (exploratory) | not run | 0.0308 | 0.1013 | 0.1200 | **0.1950** | 0.1475 | 0.1854 | 0.1713 | EXP-074, EXP-075, EXP-076, EXP-077, EXP-078, EXP-079 |

Pre-registered and **confirmed**: the policy-judged search beats the same search without the
no-revisit rule at depth 8 (p 0.0005) and the reflex at depth 9 (p 0.0002, 12 of 12 seeds) (EXP-072);
the learned judge beats the policy-judged search at depth 9 (+0.1304, p 0.0002, 12 of 12 seeds) and
beats the same judge reading only the 64-unit concept (+0.0642, p 0.0010) (EXP-074). Depths 7 and 8
of the learned-judge column are secondaries, and depth 11 is exploratory. Two of EXP-074's
evaluation seeds were looked at before its spec was written; both claims hold with them dropped.
The wider judge puts the sensory region at hidden 512, four times the 128 before. It beats EXP-074's judge
at depth 9 by +0.0638 (p 0.0002, 12 of 12 seeds) and the same recipe at hidden 128 by +0.0867
(p 0.0002, 12 of 12), both **confirmed**; at depth 11 its +0.0188 over the
earlier judge is not significant (p 0.0757) (EXP-075). EXP-076 then searched four
moves ahead instead of three with these judges and no retraining: it beats the 3-move tree at depth 11
(+0.0750, p 0.0002, 12 of 12 seeds) and at depth 9 (+0.1433, 12 of 12), and beats a 3-move tree given only
a 4-move goal check by the same +0.0750 at depth 11, all **confirmed**. The goal check alone changed no
cell, so the gain is the judge scoring leaves nearer the goal. Depth 11 is now about one cube in five.
EXP-077 then replaced the full tree with a width-64 beam five moves deep: it beats the 4-move tree at depth 9
(0.5571 against 0.4004, p 0.0002, 12 of 12 seeds, **confirmed**) but loses at depth 11 (0.1475 against 0.1950,
wins 1 of 12, both depth-11 claims **refuted**). A 5-move goal check alone changed no cell.
EXP-078 scored the full 5-move tree: level with the beam at depth 9 (0.5558, **confirmed** over the 4-move
tree, 12 of 12) and level with the 4-move tree at depth 11 (0.1854 against 0.1950, **refuted**, a third of
the detectable difference). It beats the beam at depth 11 by 0.0379 (confirmed on 12 seeds, not significant
on the 10-seed line). With these judges, more search no longer helps at depth 11.
EXP-079 retrained the wider judge on walks of up to 40 moves instead of 14, so that a fifth of its
training states sit 11 moves from solved rather than about 1%. Searched with the same 4-move tree, it
is worse at both depths: 0.1713 against 0.1950 at depth 11 (ahead on 3 of 12 seeds) and 0.3454
against 0.4004 at depth 9 (1 of 12), both **refuted**. Its values far from the goal moved nearer the
truth, but it ranks those states no better than before (p 0.231), and the search uses only the ranking.
EXP-080 asked whether any judge on that spiking encoder could rank far states better, by training
it directly on the true distances as a diagnostic ceiling. It barely can: a fresh head on the
fine-tuned encoder ranks states 9 to 11 moves out +0.0228 better than the wider judge (p 0.055,
**refuted**), and on the frozen encoder +0.0106 (**refuted**). A plain MLP on the raw facelets, given
the same labels, ranks them +0.1778 better (12 of 12 seeds). The ordering is in the cube's state and
is lost in the encoder, so the verdict is **encoder-limited**, and no search was run on it.
EXP-081 then trained that fine-tuned arm four times as long, continuing it from 4,000 to
16,000 updates on the same seeds. It now ranks far states +0.0678 better than the wider judge
(10 of 12 seeds, p 0.0012, **confirmed**), so EXP-080's verdict was a budget reading. Used as the
4-move tree's judge, it solves **0.4088 at depth 11 against 0.1950** (12 of 12 seeds, p 0.0002).
That judge was trained on the true distances, so 0.41 is a ceiling on what the encoder can carry,
not a solver result: the open question is how a judge trained from experience gets there.
EXP-082 asked whether the wider judge itself, trained only by value iteration, was budget-limited
too, continuing it from 4,000 to 16,000 updates. At depth 11 it solves 0.2083 against 0.1950
(9 of 12 seeds, p 0.1035, **not confirmed**), and its far-state ranking rises only +0.0278, so the
verdict is **saturated at 4x budget**: the limit there is the objective. The gain went to states
near the goal instead, and at depth 9 the same 4-move tree solves **0.6200 against 0.4004** (12 of 12
seeds, p 0.0002, reported rather than pre-registered), the best depth-9 number the project has.
EXP-083 pre-registered that depth-9 gain and re-tested it on scrambles neither judge ever trained
on, found by replaying the training walks exactly. It holds: **0.5537 against 0.3279** (+0.2258,
12 of 12 seeds, p 0.0002), and it reaches depth 10, **0.2904 against 0.1992** (+0.0912, 12 of 12,
p 0.0002). The gain fades by depth, about +0.22 at 9, +0.09 at 10 and +0.01 at 11.

The learned judge reads all 192 sensory neurons and was trained by value iteration from its own
look-ahead, never from the true distances, with the spiking encoder trained beneath it. The search
around it is still hand-coded: a simulator, a three-move tree and a visited-state table. The reflex
column is re-evaluated on the same evaluation stream as the search arms, which is why it differs
slightly from the published numbers above (`docs/reproducibility-audit.md`). A random 2x2 is
usually 11 moves from solved, and about one in ten of those is solved so far.

### What moved the needle

1. **A depth curriculum.** Depth 3 went from 2.2% to 50.0%, a 22.7x improvement, with **no change
   to the architecture at all** (EXP-034/035).
2. **Compute.** Success is linear in the *logarithm* of training spend, about **0.22 per log10**,
   with no knee (EXP-044/045/046, held out of sample at depth 8 by EXP-062). So the "depth series"
   was a **budget series**, and the architectural "wall" reported in earlier weeks was an exchange
   rate.
3. **Training the spiking encoder during RL** (EXP-039/040/047/048). **This is the one genuinely
   neuromorphic change in the project that pays for itself.**
4. **A learned critic**, whose benefit is within-episode state-dependence (EXP-056/057) and which
   replicated on fresh seeds (EXP-060). It is also **ordinary reinforcement learning**.

### What did not work, stated as plainly as the above

- **Episodic memory does not help, and actively hurts** (-0.0954, p 0.0056). The hippocampal
  recall block is **indistinguishable from matched-magnitude noise** (EXP-059/061), and a learned
  attention over a *perfect* episodic cache is **0.198 worse than having no memory at all**
  (EXP-063). That last arm was built as a ceiling so the negative would be decisive.
- **The neuromorphic claim for the credit-assignment stage is refuted.** The `neuromod`
  neuromodulatory bus was made load-bearing and bought nothing measurable (EXP-053).
- **More training of the reflex policy will not reach the goal.** Depth-11 random scrambles would
  cost about **33 days of compute per seed** (EXP-062). Nothing about it requires new science; the
  arithmetic is what stops it, which is why Phase 4 turned to look-ahead.
- **Five measurement instruments were retired**, including the probe that carried most mechanism
  claims through weeks 17 to 20. Each worked as a threshold and failed as a gradient. See
  `docs/retired-instruments.md`.

**Full account: [`docs/writeup/report.md`](docs/writeup/report.md).** The Phase 3 audit, graded
criterion by criterion, is `docs/phase3-honest-assessment.md`. The short
version is that almost every architectural idea lost to a training idea, and the project's main
achievement is that it kept finding that out instead of not finding it out.

---

## Project structure

```
src/neuromorphic/        # Reusable Python package: models, regions, training, monitor
src/dashboard/           # Batch matplotlib dashboard (multi_region_viz.py)
dashboard/               # NEURO·SCOPE live web app (Vite + React + TS + react-three-fiber)
experiments/             # One numbered subfolder per experiment (config + entry point)
notebooks/               # Exploratory Jupyter work
scripts/                 # One-off utility scripts
tests/                   # Pytest tests (added as code stabilizes)
docs/                    # Specs, plans, ADRs, and design handoffs
outputs/                 # Experiment traces + generated figures (committed)
data/                    # Datasets (gitignored), downloaded on first run
checkpoints/             # Model state_dicts (gitignored)
runs/                    # TensorBoard logs (gitignored)
wandb/                   # W&B local cache (gitignored)
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
# CUDA 12.1: check https://pytorch.org/get-started/locally/ for your specific config
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Or CPU-only:
# pip install torch torchvision
```

### 3. Install the rest of the dependencies and the package itself

```powershell
pip install -r requirements.txt
pip install -e .
```

The `pip install -e .` installs the `neuromorphic` package in editable mode, so code changes take effect immediately without reinstalling.

### 4. Run the smoke test

```powershell
python experiments/001_smoke_test/run.py --config experiments/001_smoke_test/config.yaml
```

If this prints `Smoke test passed.` and saves a checkpoint to `checkpoints/001_smoke_test.pt`, the scaffold is healthy.

---

## Experiment tracking

Two trackers are wired in and used together by default:

- **Weights & Biases**: cloud-hosted, best comparison UI. Sign up at [wandb.ai](https://wandb.ai), then run `wandb login` once.
- **TensorBoard**: local, no account, lives in `runs/`. Launch with `tensorboard --logdir runs/`.

Switch trackers per-experiment by editing `tracker:` in the config YAML (`wandb`, `tensorboard`, `both`, or `none`).

---

## Configuration system

Every experiment is described by a single YAML file in its `experiments/NNN_*/` folder. The schema is the `ExperimentConfig` dataclass in `src/neuromorphic/config.py`. Unknown fields fail loudly at load time, which prevents typos like `lerning_rate: 0.01` from silently using the default.

CLI overrides work for any field:

```powershell
python experiments/001_smoke_test/run.py --config experiments/001_smoke_test/config.yaml --lr 0.001 --epochs 10
```

---

## Visualization: NEURO·SCOPE

The brain is monitored through **NEURO·SCOPE**, a data-driven dashboard platform that renders whatever a saved trace *declares* (region topology, pathways, spikes), never hardcoded to today's five regions. Traces are versioned JSONL written by the `neuromorphic.monitor` package; the platform architecture and phase roadmap live in `docs/superpowers/specs/2026-06-18-neuroscope-platform-design.md`.

Two front-ends read the same trace contract:

### Batch dashboard (matplotlib, static figure)

One multi-panel PNG from a saved trace: 5 spike rasters, an inter-region communication heatmap, the grid-world state, and a reward curve. From the repo root with the venv active:

```powershell
python -m dashboard.multi_region_viz outputs/week11_dashboard_trace.jsonl
# options: --step N (default last frame) · --out PATH · --show
```

(`dashboard` here resolves to `src/dashboard/`, importable via `pip install -e .`.)

### Live web dashboard (NEURO·SCOPE: Vite + React + react-three-fiber)

A real-time replay with a WebGL neuron-field hero and reactive panels, under `dashboard/` (a separate JS toolchain, Node ≥ 20). First run:

```powershell
cd dashboard
npm install
npm run sync:trace   # copies the trace into public/ (required before dev/build)
npm run dev          # serves on http://localhost:5173 (press ▶ to animate)
```

Also: `npm run build`, `npm test` (Vitest), `npm run e2e` (Playwright smoke). **Phase 0** (foundation, minimal hero, two panels) and **Phase 1a** (all five data-driven panels) are built and merged; the full hero treatments and theming are the remaining Phase 1 slices, planned in `docs/superpowers/{specs,plans}/`.

---

## Phase roadmap

| Phase | Window | Focus | Status |
|---|---|---|---|
| 0: Foundations | Apr–May 2026 | NN concepts, PyTorch, snnTorch tutorials, RL basics | Complete |
| 1: Single-region SNNs | May 2026 | Spiking MNIST (98.05%), recurrence (+37.3% over feedforward), surrogate gradients, STDP | Complete |
| 2: Multi-region brain | Jun–Jul 2026 | Five-region architecture, inter-region communication, grid-world task | Complete, tagged `phase-2-complete`. Honest audit: 1 met / 3 partial |
| 3: Rubik's Cube | Jul–Sep 2026 | 2x2 cube environment, curriculum learning, regional specialization | Graded at `phase-3-checkpoint`: 3 of 4 criteria met or exceeded, 1 partial; the self-imposed Stage 4 is priced out |
| 4: Capstone | Oct–Dec 2026 | Research report, and look-ahead toward solving a random 2x2 | **In progress.** Report written; look-ahead experiments EXP-070 to 072 run, a learned judge (EXP-074, superseding EXP-073) confirmed at depth 9, a wider region under it (EXP-075) confirmed at depth 9, a 4-move tree over those judges (EXP-076) confirmed at depths 9 and 11, and two 5-move searches (the EXP-077 beam and the EXP-078 full tree) confirmed at depth 9 and refuted at depth 11, then a judge retrained on longer walks (EXP-079) refuted at both depths, and a judge trained on true distances (EXP-080) that looked encoder-limited at 4000 updates and, trained 4x longer (EXP-081), doubles depth-11 solves as a ceiling, and the value-iteration judge trained 4x longer (EXP-082), saturated at depth 11 and raising depth 9 to 0.6200, a gain confirmed on never-trained-on states and reaching depth 10 (EXP-083) |

Future (Year 2+): **Strategy Sentinel**, applying the architecture as a meta-cognitive layer on top of an existing LEAN/QuantConnect quantitative trading stack. Separate repo, not included here.

---

## License

MIT. See `LICENSE`.

---

## Acknowledgments

Built with [snnTorch](https://snntorch.readthedocs.io/) and [PyTorch](https://pytorch.org/). Curriculum draws on Gerstner's *Neuronal Dynamics*, Sutton & Barto's *Reinforcement Learning*, the 3Blue1Brown neural network series, and snnTorch's tutorials.
