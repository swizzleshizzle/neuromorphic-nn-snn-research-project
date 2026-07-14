# EXP-028 Results — Sensory-Code Ablation (dose-response) `RESULTS PENDING`

> **Why this file exists:** the standing habit adopted after the 2026-07-13 Phase-2 audit — every
> experiment commits a curated, in-repo results record so the authoritative numbers never live only in
> a gitignored `outputs/` folder or a vault note. Provenance: 12 seeds (0-11), grid 5, run on the laptop
> 2026-07-13 (12 encoders cached once, then 216 head-retrains: 6 gaussian + 6x2 unit-drop doses x 12
> seeds). **The interpretation contract below is pre-registered — written before the numbers land — so
> the read stays honest.** Regenerate with the command at the bottom.

## What this tests (and how it differs from EXP-027 Component B)

EXP-027 Component B masked concept units on a **fixed, already-trained** head and found the code
causally distributed (dropping half the units cost ~11 pts). EXP-028 asks the complementary,
policy-learning-level question:

> If we **degrade the pre-trained sensory concept and then re-train the linear head against the
> degraded code**, how much does held-out navigation fall, and how gracefully — i.e. is concept
> *fidelity* a load-bearing input to policy learning, in a dose-dependent way?

Two operators degrade the frozen concept (cached encoder reused across doses; only the head re-trains):
- **Gaussian noise** (`sigma`): additive `N(0, sigma^2)` on the concept rate — a continuous *fidelity* dose.
- **Unit-drop** (`p`): zero a fraction of concept units for the whole run — a structural *dimensionality*
  dose, in two modes: `random` (neutral) and `top` (drop the most displacement-important units first).

## Pre-registered hypothesis + falsifiers (written before results)

- **Expected:** held-out success is a **monotone, graceful decreasing** function of dose. The `dose=0`
  anchor ≈ the EXP-026/027 pretrained baseline (~35-45%); at maximum dose it approaches the
  **random-encoder floor** (~8-17%, the EXP-026 random arm). Gaussian and unit-drop should **agree** on
  a graceful monotone decline, and `unitdrop_top` should fall faster than `unitdrop_random` (importance
  is real).
- **Falsifier A — flat curve:** if success barely moves with dose, then re-training compensates for
  degradation and concept *fidelity* is **not** what drives the lift → contradicts EXP-026's causal
  story; investigate before claiming EXP-028 confirms it.
- **Falsifier B — cliff at tiny dose:** if a small dose craters success, the code is **brittle** →
  contradicts EXP-027's distributedness. A genuine A-vs-B conflict to reconcile, not smooth over.

## Results `PENDING`

Mean held-out nav success across 12 seeds (source: `outputs/028_curve.md` / `028_summary.json`). The
three `dose=0` cells are the same clean-baseline re-train and should read ≈ equal (a consistency check).

**Gaussian noise (fidelity dose):**

| sigma | held-out success |
|---|---|
| **0.0 (baseline)** | __% |
| 0.05 | __% |
| 0.1 | __% |
| 0.2 | __% |
| 0.4 | __% |
| 0.8 | __% |

**Unit-drop (structural dose):**

| p dropped (of 64) | random | top |
|---|---|---|
| **0.0 (baseline)** | __% | __% |
| 0.1 | __% | __% |
| 0.25 | __% | __% |
| 0.5 | __% | __% |
| 0.75 | __% | __% |
| 0.9 | __% | __% |

*Anchors to sanity-check the operators are calibrated: baseline ≈ __% (should match EXP-026/027 ~35-45%);
max-dose ≈ __% (should approach the random-encoder floor ~8-17%). If the endpoints don't bracket, the
operator is miscalibrated — fix before reading the middle.*

## How to read it — the interpretation contract

- **Monotone & graceful (the expected, EXP-026-confirming result):** success declines smoothly from the
  baseline toward the floor as dose rises; both operators agree; `top` falls faster than `random`. This
  is the *causal, re-trained* confirmation that concept fidelity drives the navigation lift — and it
  **quantifies** how much fidelity the policy needs (see ED50 below). Reconciles cleanly with EXP-027:
  a fixed head tolerates *unit loss* (distributed), but the policy still *depends on overall fidelity*
  when that fidelity is degraded across the board.
- **ED50-style summary** `PENDING`: the dose at which success drops halfway between baseline and floor —
  gaussian sigma ≈ __, unit-drop p ≈ __. One interpretable number for "how much degradation the policy
  absorbs before it matters."
- **Ordering check:** at matched `p`, expect `success(random) >= success(top)`. A violation means the
  importance ranking or mask plumbing is off — flag, don't average past it.

## Verdict `PENDING`

> [!note] Fill when results land
> Expected: **concept fidelity is causally load-bearing for policy learning, dose-dependently** —
> completing the EXP-026 (fidelity lifts the cap) / EXP-027 (the lift is a distributed, specialized
> code) / EXP-028 (degrading fidelity dose-dependently removes the lift) causal chain. Then: ADR-0001
> **Amendment 6**, and this becomes the final evidence brick for the Phase-2 "regional specialization"
> checkpoint criterion. If the curve is flat or a cliff instead, do NOT merge the confirming claim —
> reconcile against EXP-026/027 first.

## Regenerate

```powershell
.venv\Scripts\python.exe experiments\028_sensory_ablation\run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 16
```
