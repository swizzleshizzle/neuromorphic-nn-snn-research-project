# Phase 3 Kickoff Brief (2x2 Rubik's Cube)

**Decided:** 2026-07-20 (Mon study session, gap week) · **Kickoff:** Sat 2026-07-25 · **Phase 3 runs:** Jul 25 to Sep 27
**Grounds:** `docs/architecture-spec-v3.md` §7 (the three parked decisions), `docs/phase2-to-phase3-transition.md`,
`docs/phase2-honest-assessment.md`, ADR-0001 Amendments 1 to 6.

This brief closes the three open kickoff decisions so Saturday is execution, not deliberation.

---

## 1. The three decisions (locked)

### D1 — Opening move: **fail-first baseline**
Build the cube env, then run the **v1 recipe on 1-move scrambles** and let it fail informatively, rather
than redesigning training first.

**Why:** it quantifies the real gap; it produces the baseline that Phase-3 checkpoint criterion 2
("comparison vs monolithic on at least one metric") requires anyway; and EXP-028 showed the grid cap was
**optimization-limited, not encoder-limited**, so the cheap levers deserve a test before architectural
surgery.

**Fairness requirement:** the baseline carries **EXP-028's input-noise regularization**, so it is the
*best* v1 can do, not an under-tuned strawman. A strawman baseline would make the regionalization
comparison dishonest.

### D2 — Cube encoding: **honest port, no pre-training proxy yet**
Sensory input becomes **24 facelets x 6 colors = 144 one-hot inputs, Poisson-rated**, directly mirroring
`encode_gridworld` (which today is `agent one-hot + goal one-hot` = 50 inputs at `max_rate=0.5`, `T=32`).
**No supervised pre-training proxy at kickoff.**

**Why not a proxy yet:** EXP-026 established that pre-training the encoder lifts the cap, but EXP-028
then localized the *remaining* cap to policy optimization. Investing in a cube pre-training target before
we know the encoder binds would repeat the mistake EXP-028 warned about. Measure first.

**Why not tuning-curve population coding (the L13 nuance):** population coding with overlapping tuning
curves is built for **continuous** variables (position, angle, distance). Facelet colors are
**categorical**: there is no "between red and blue," so there is no continuum to tile. Forcing tuning
curves onto color identity would be cargo-culting the technique.

**Where the genuine L13 win lives (deferred, not discarded):** population-code the quantities that really
are continuous and task-relevant, as the cube's analog of the grid's displacement signal:
- **distance-to-solved** (exactly computable for a 2x2; the state space is only ~3.67M), and/or
- **per-face uniformity / solved-ness**, and/or
- **per-facelet correctness** (24-d), the richest direct analog of displacement.

Revisit these only if evidence says the encoder is the binding constraint.

**Biological-plausibility constraint (keep):** 24 facelets is the correct *sensory* representation, since
it is what an eye sees. A solver's compact encoding (corner permutation + orientation) would smuggle
solution structure into the retina. **Keep the sensory region sensory.**

### D3 — Plasticity: **recall-in-loop only** (as the engagement ceiling)
When we engage, go only as far as **hippocampus recall in the loop**. R-STDP and reward-modulated
memory/router (ADR migration steps 1 to 3) stay deferred.

**Why:** the cube genuinely needs move history (avoid cycles, recognize visited states), so memory is the
first region with an evidence-backed reason to participate. Everything beyond that is speculation until
the baseline data exists.

---

## 2. Sequencing (resolves the D1/D3 tension)

D1 measures "the v1 recipe," but v1 runs `recall=False`. Turning recall on *for the baseline* would mean
we never measured v1. So:

1. **Baseline = pure v1**, `recall=False`, frozen encoder + linear head + noise regularization.
2. **Recall-in-loop is the first engagement step AFTER the baseline**, once we know where v1 actually stands.

This is also principled on the task itself: **a 1-move scramble is solvable reactively** (one move undoes
one move), so memory should not be needed at depth 1. Memory earns its place as scramble depth rises,
which is exactly the evidence we want.

---

## 3. Saturday's concrete first moves

1. **Build the 2x2 cube Gymnasium env** + tests: 24-facelet state, 6 moves, scramble-depth curriculum
   (1 -> 2 -> 3), mirroring `GridWorldEnv`'s structure. Safe, isolated infra; nothing depends on the
   training decisions.
2. **Port the sensory encoding**: `encode_cube` (144 one-hot, Poisson), mirroring `encode_gridworld`.
3. **Run the v1-recipe baseline** on 1-move scrambles, with noise regularization, 12 seeds
   (the n>=12 standing rule; n=5 lied to us in EXP-026).
4. **Stand up the monolithic same-neuron-count baseline early** so checkpoint criterion 2
   ("does regionalization help?") is answerable from day one rather than retrofitted.

## 4. Pre-registered expectation for the baseline (write it before the numbers)

> **Revised 2026-07-24 (design audit).** This section originally said *12-way* classification. The env
> uses a **6-move** action space, because on a 2x2 a face turn equals the opposite face's counter-turn
> (`U == D'`, `R == L'`, `F == B'`), making a 12-move space exactly 2x redundant. See the cube-env spec §1.
> The revised numbers below supersede the originals; the *shape* of the prediction is unchanged.

A 1-move scramble is effectively a **6-way classification** ("recognize the perturbation, apply its
inverse"), which a reactive frozen-extractor + linear head *should* be able to learn. Each depth-1
scramble has **exactly one** solving move, so **chance is 1/6 = 16.7%**.

- **Expected:** solid success at depth 1; sharp degradation at depth 2; near-chance by depth 3. **The
  informative number is the depth at which it collapses.**
- **Read the depth axis honestly.** `scramble_depth` is a move count, not a distance: a random walk of
  `k` moves can land closer than `k` (measured: 0% contamination at depths 1-2, ~3.6% at depth 3, ~15%
  at depth 6). Pass `exact_depth=True` with a distance provider if the collapse-vs-depth curve is the
  headline number, otherwise report depth as an upper bound.
- **If it fails at depth 1:** that is NOT an architecture verdict. It points at the encoding or the
  training setup, and must be debugged before any conclusion about regionalization is drawn.
- **If it succeeds well beyond depth 1:** the cube is less of a forcing function than assumed, and the
  Phase-3 mandate (engage the other regions) needs re-justification rather than assertion.

Per the standing habit: **commit an in-repo `RESULTS.md`** for the baseline, whatever it shows.

## 5. Known dependency, not blocking Saturday

The **dashboard task panel is grid-hardcoded** in both `_grid_world` (Python) and React `TaskState`, so a
**cube trace will not render meaningfully**. This does not block the env build or the baseline, but it is
the next Phase-3-relevant dashboard item and should land before cube traces are worth watching. The rest
of the dashboard (NEURO-SCOPE Phase 2 Live shipped 2026-07-19) is ready.

## 6. Still deferred (explicitly, so it is not silently dropped)

- Cube pre-training proxy (distance-to-solved / per-facelet correctness / per-face uniformity) — D2 above.
- R-STDP and reward-modulated memory + router (ADR migration steps 1 to 3) — D3 above.
- EXP-028 follow-up: **train-noisy / eval-clean**, to isolate the training-regularization benefit from the
  eval-time input change.
- NEURO-SCOPE 1c chrome, including the Mike-prioritized **pulse-system rework** (make pulse motion encode
  real flow rather than a wall-clock loop).
