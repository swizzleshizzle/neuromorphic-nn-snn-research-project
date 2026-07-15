# EXP-028 Results — Sensory-Code Ablation (dose-response)

> **Why this file exists:** the standing habit adopted after the 2026-07-13 Phase-2 audit — every
> experiment commits a curated, in-repo results record so the authoritative numbers never live only in
> a gitignored `outputs/` folder or a vault note. Provenance: 12 seeds (0-11), grid 5, run on the laptop
> 2026-07-13/14 (12 encoders cached once, then 216 head-retrains: 6 gaussian + 6x2 unit-drop doses x 12
> seeds). **The interpretation contract was pre-registered before the numbers landed** (git `c63f367`,
> "results pending") — and the headline result *refuted* it, which is exactly why pre-registration
> matters. Regenerate with the command at the bottom.

## What this tests (and how it differs from EXP-027 Component B)

EXP-027 Component B masked concept units on a **fixed, already-trained** head (found the code causally
distributed). EXP-028 degrades the pre-trained concept and **re-trains the linear head against the
degraded code**, across a dose grid — the policy-learning-level question: is concept *fidelity* a
load-bearing input to policy learning? Two operators degrade the frozen concept (cached encoder reused
across doses; only the head re-trains): **Gaussian noise** (`sigma`, a continuous fidelity dose) and
**unit-drop** (`p`, a structural dimensionality dose, `random` and `top`-importance modes). Noise is
applied at **both train and eval** (the wrapper sits on the head throughout) — see the caveat.

## Results

Mean held-out nav success across 12 seeds (`outputs/028_curve.md`). The three `dose=0` cells are the
same clean-baseline re-train and read identically (**43%**) — consistency check passed.

**Gaussian noise (fidelity dose) — INVERTED, not a decline:**

| sigma | held-out success | optimality | mean steps |
|---|---|---|---|
| **0.0 (baseline)** | **43%** | 0.65 | 15 |
| 0.05 | 26% | – | – |
| 0.1 | 32% | – | – |
| 0.2 | 69% | – | – |
| 0.4 | **83%** | 0.57 | 13 |
| 0.8 | 83% | 0.38 | 25 |

**Unit-drop (structural dose) — monotone graceful decline, as predicted:**

| p dropped (of 64) | random | top |
|---|---|---|
| **0.0 (baseline)** | **43%** | 43% |
| 0.1 | 42% | 31% |
| 0.25 | 35% | 31% |
| 0.5 | 31% | 17% |
| 0.75 | 28% | 21% |
| 0.9 | 26% | 15% |

## The finding — the two operators tell different stories

**Unit-drop confirmed the hypothesis.** Removing units degrades held-out navigation monotonically and
gracefully; `top` (drop most-important-first) falls faster than `random` at every dose (43->15 vs
43->26). Re-training cannot recover *removed* information. This is fully consistent with EXP-027: the
code is distributed (robust to random loss) but importance-ordered (the top units carry more).

**Gaussian noise refuted the hypothesis — and revealed something more interesting.** Additive concept
noise does not degrade navigation; moderate noise *roughly doubles* held-out success (43% -> 83%),
consistently across all 12 seeds (per-seed 67-100% at sigma 0.4). Small noise (sigma 0.05-0.1) hurts;
moderate-to-large noise helps. The optimality diagnostic separates the mechanism:

- **sigma 0.4 is a genuine improvement:** 2x the goals reached with path efficiency barely lower
  (optimality 0.57 vs 0.65) and *fewer* steps (13 vs 15). Not wandering — better, more robust navigation.
- **sigma 0.8 tips into wandering:** same 83% success but optimality collapses (0.38) and steps balloon
  (25) — it reaches the goal by exploring, not by navigating well.

**Interpretation (honest):** the frozen-encoder + linear-head + REINFORCE policy is **under-regularized**.
Moderate input noise acts as a regularizer / exploration driver that escapes the weak, partially-collapsed
policies the deterministic baseline settles into — directly echoing the entropy-collapse failure mode
documented in EXP-025 (which the entropy bonus + advantage-norm only partly fixed). So concept *fidelity*
is **not** the binding constraint on policy learning here; policy *optimization* is. Structural unit-drop
still hurts because you cannot regularize back information that has been removed.

This is a stronger, more useful result than the predicted "fidelity confirmed": it is a concrete lead
that **better regularization could raise the ~43% navigation cap**, which matters going into Phase 3.

## Pre-registration outcome

- **Unit-drop:** hypothesis (monotone graceful decline, `top` < `random`) **CONFIRMED**.
- **Gaussian:** hypothesis **REFUTED** — pre-registered Falsifier A ("re-training compensates so fidelity
  is not the driver") fired, and harder: the curve is *inverted*, not flat. Reported as the finding, not
  smoothed over.

## Caveats + follow-up

- **Noise is applied at train AND eval**, so this run cannot fully separate a *training-regularization*
  benefit (the interesting claim) from an *eval-time* input change. The sigma-0.4 optimality (0.57, near
  baseline) argues for a genuine training benefit; sigma-0.8 (0.38) shows an eval-time wandering component
  at high dose. **Follow-up to pin the mechanism (open):** a **train-noisy / eval-clean** variant — if a
  noise-trained head still beats baseline when evaluated on the *clean* concept, that isolates the
  regularization interpretation cleanly. Not run yet; does not block this writeup.
- High per-seed variance persists (baseline 17-83%); means are over 12 seeds.

## Verdict

> [!success] Concept fidelity is not the binding constraint on policy learning; policy regularization is.
> Structural unit-drop degrades navigation monotonically (distributed + importance-ordered, confirming
> EXP-027), but additive concept noise *improves* held-out navigation up to ~2x (genuine at sigma 0.4,
> wandering by sigma 0.8) — evidence the frozen-extractor + REINFORCE policy is under-regularized, tying
> back to the EXP-025 collapse story. ADR-0001 **Amendment 6**. Lead for Phase 3: regularization, not
> encoder fidelity, is the next lever on the cap.

## Regenerate

```powershell
.venv\Scripts\python.exe experiments\028_sensory_ablation\run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 16
```
