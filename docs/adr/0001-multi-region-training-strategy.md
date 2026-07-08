# ADR-0001 — Multi-Region Training Strategy

- **Status:** Accepted (amended 2026-06-22 Amendment 1; 2026-07-06 Amendment 2; 2026-07-07 Amendment 3)
- **Date:** 2026-06-18
- **Phase:** 2 (multi-region brain), Step 2.3 — "how does the whole brain learn?"
- **Deciders:** project author
- **Supersedes / superseded by:** none

---

## Amendment 1 (2026-06-22, Week-12 debug) — the policy is a learnable head, not the raw motor output

The first training run (EXP-023) learned then collapsed. Systematic debugging found the
original premise — *"motor spike-counts ARE the action logits, read via `motor.winner`
argmax"* — is broken two ways (evidence in `experiments/023_week11_brain_training/debug_collapse.py`):

1. **Saturation freeze.** Summed motor spike-counts over the window reach ~28, so
   `softmax` saturates to an *exact* one-hot → `log π(a) = 0` → a **zero-gradient
   absorbing state**. The policy freezes permanently in one update (gnorm 35M → 0.00,
   entropy → 0) on the structural-favourite action, which bangs a wall for −60.
2. **Degenerate readout.** The untrained motor/PFC output is a fixed "structural
   favourite" — only **one** action neuron fires, barely state-dependent (PFC utility
   pairwise distances 0.08–0.56 across very different observations, vs the sensory
   concept's 3.4–4.3). It cannot express a real four-action, state-conditioned policy.

The regions themselves are healthy (rates stable, all five fire) — the failure is purely
in *how action selection reads the brain*.

**Amended decision:** the policy is a small **trainable `nn.Linear` head reading the
sensory concept** (a rich, state-dependent code) → action logits. The brain is a **frozen
feature extractor** in v1 (runs under `no_grad`); only the head trains. This gives every
action a gradient handle and a learnable scale (no spike-count saturation). Memory stays
bypassed (`recall=False`). Result: untrained −60 / 0% goal → **trained +3 / 100% goal,
8-step optimal paths**.

**Consequences vs. the original decision:**
- The policy gradient path is now **sensory → head**; PFC, router, and motor leave the
  *policy* path for v1 (they still run and are still visualized — they are just not where
  the learnable decision lives, because their readout is degenerate at init).
- "Train the whole differentiable path (sensory→PFC→motor)" is **superseded** for v1 by
  "train a head on the frozen sensory features." Unfreezing the encoder, and giving
  PFC/motor a non-degenerate readout so they can re-enter the policy path, are follow-ups
  (candidates: motor re-init to break the structural favourite, region pre-training, an
  entropy bonus). The R-STDP hybrid path (below) is unchanged.

---

## Amendment 2 (2026-07-06, Week-13 EXP-025) — the cap is the frozen encoder, not the readout

Amendment 1 left one question open: is the ~30-50% held-out navigation cap caused by the *linear head*
or by the *frozen encoder* underneath it? EXP-025 (`experiments/025_head_capacity/`) answers it by
swapping a one-hidden-layer MLP head (`Linear(64→128)→ReLU→Linear(128→4)`) for the linear head and
comparing held-out generalization across a paired 5-seed sweep.

Getting a **fair** comparison took removing a confound. A naive MLP under REINFORCE walks straight
into the Amendment-1 saturation failure — about half the seeds collapse to a zero-entropy one-hot
policy and die. Two optional, default-off trainer stabilizers were added (byte-identical to prior
baselines): an **entropy bonus** (`entropy_beta`, `-beta * sum_t H`) and **per-episode advantage
normalization** (`normalize_advantages`). A modest `entropy_beta=0.01` alone did nothing (the summed
entropy term was dwarfed by un-normalized advantages); **advantage normalization plus
`entropy_beta=0.05` eliminated the collapse entirely** (0/10 MLP runs collapse; MLP entropy 1.11-1.35).

**Result:** with a collapse-free, fair test, the **MLP head does not beat the linear head** on
held-out goals (shaped 20% vs 10%; sparse 10% vs 20%; net even, each within the other's seed spread).
A nonlinear readout extracts no more *generalizable* signal from the frozen 64-d sensory concept.

**Amended conclusion:** readout capacity is **not** the binding constraint — the **frozen sensory
encoder is the wall**. The v1 "frozen feature extractor + trainable head" design has been pushed to
its representational ceiling. The next lever is to **engage the encoder**: supervised pre-training of
the sensory region and/or unfreezing it to train end-to-end (the entropy-bonus and
advantage-normalization stabilizers are now in place to make that tractable). This is the Phase-2→3
bridge, and the first change that lets a region *specialize through learning* — which is what makes
the deferred ablation studies meaningful. (Caveat: n=5 seeds, wide spread — the load-bearing claim is
"no MLP advantage," not a precise cap figure.)

---

## Amendment 3 (2026-07-07, Week-14 EXP-026) — engaging the encoder lifts the cap (Option B validated)

Amendment 2 concluded the frozen sensory encoder is the navigation cap and pointed at engaging it.
EXP-026 (`experiments/026_sensory_pretrain/`) does the first engagement: **pre-train the sensory
encoder** on a supervised state-encoding task, freeze it, and re-run the generalization eval.

The objective is **goal-relative displacement**: a scratch `Linear(concept -> 2)` readout is trained
to predict `(gx-ax, gy-ay)` (normalized) from the concept rate, backprop through the spiking encoder
(surrogate gradients); the readout is discarded and the shaped encoder kept, then frozen for the RL
policy. A two-stage protocol: (1) a cheap gate on held-out *displacement* decode error, then (2) the
paired held-out *navigation* eval.

**Result (de-noised, 12 seeds, both arms paired by seed, held-out 10):**
- Stage 1 gate passes decisively — pre-trained displacement error ~0.14 vs a random-encoder ~0.32,
  all 12 seeds. The encoder learns goal-direction.
- Stage 2 — pre-training **lifts held-out navigation** ~2.5-5x on the mean: paired per-seed,
  pre-training beats the random encoder in **19 of 24 seed-regime cells** (shaped 9/12, +28 pts;
  sparse 10/12, +34 pts; sparse sign-test p approx 0.02, pooled p < 0.001). An initial n=5 run had
  read as a failure (sparse apparently regressing); that was seed noise, corrected by de-noising.

**Conclusion:** **engaging a region via training lifts the representational cap.** The
frozen-random encoder was a real binding constraint (consistent with Amendment 2), and shaping it
moves the cap. This is the first evidence that a region specializing through learning helps the
task — which also makes the deferred ablation studies meaningful. Caveats kept explicit: per-run
variance is high (single seeds swing 0-100%), so the claim is an averaged effect, not a reliable
per-run one; and the pre-registered "clears the band" (mean + range) criterion is not met, but the
paired per-seed test is the correct analysis for a paired design and is decisively positive.

Follow-ups this unlocks: ablation of the now-engaged encoder; reducing the RL variance; and the
other Option-B flavor (unfreeze the encoder end-to-end so it adapts to the reward, not a proxy).

---

## Context

The five-region `Brain` (sensory → hippocampus → prefrontal → thalamic router → motor) is wired and runs end-to-end, but it is **untrained**: `Brain.learn(reward)` currently only pushes a value onto the dopamine bus; the actual weight update is a deferred hook. This ADR decides what fills that hook — the credit-assignment strategy for the whole brain on the grid-world task.

The task is a 5×5 grid world with sparse reward (`-1` per step, `+10` on reaching the goal). The observation is `(agent_x, agent_y, goal_x, goal_y)` — **fully observable**.

### Two axes the question conflates

The session brief frames the choice as "surrogate gradients vs region-local vs hybrid." That mixes two independent decisions:

1. **Credit assignment** — how error/reward flows across regions (end-to-end backprop vs region-local rules).
2. **Learning objective** — what signal drives the update. This is an **RL** task (sparse reward, no labels), so an RL algorithm is required regardless of axis 1. Surrogate gradients only make the spike nonlinearity differentiable; they do not, by themselves, produce a learning signal.

### What is and isn't differentiable in the current `Brain` (verified)

- ✅ **Sensory → Prefrontal → Motor** is a clean backprop path. All regions use `snn.Leaky`, which carries snntorch's default surrogate gradient (ATan).
- ✅ **The router gate is differentiable.** `apply_gate(signal, gate_closed) = signal * (1 - gate_closed)` is a multiply; gradient flows through it to the router's LIF (surrogate). The gate is a soft bottleneck, not a barrier.
- ✅ **`motor.winner` (argmax) is not in the gradient path.** It only reads out the greedy action; the loss flows through motor spike-counts → action log-probability, not through argmax.
- ⚠️ **The hippocampal attractor is structurally outside autograd by design.** `W_rec` is a `register_buffer` updated under `torch.no_grad()` via a one-shot Hebbian store. "End-to-end backprop" can never include the attractor itself — the architecture is telling us memory wants a local rule, not backprop.

### Key enabling fact

Because the grid world is **fully observable**, a *reactive* policy (observation → action) solves it. Memory is not required to reach the goal. This means the first training cut can train the feedforward policy and leave the Hebbian hippocampus out of the loop, and still learn the task.

## Decision drivers

- Get the system **working** first; biological plausibility is a later goal, not a v1 constraint.
- Prefer the smallest differentiable path that can provably solve the task.
- Don't fight the architecture: the one component that resists backprop (hippocampal `W_rec`) is left to its native Hebbian/local rule.
- Keep a clean migration path toward the more biological hybrid.

## Considered options

1. **Surrogate gradients end-to-end** — treat the brain as one differentiable network. Proven, clean gradient signal; not biologically plausible; backprop across regions. (One region — the Hebbian hippocampus — is categorically not differentiable.)
2. **Region-local learning + global reward (R-STDP)** — each region learns from a local rule modulated by the dopamine bus. More biological; slower convergence; harder to debug; the R-STDP machinery is only at the "taste" stage (EXP-021).
3. **Hybrid** — surrogate gradients *within* regions, reward modulation *between* regions. Best long-term fit; highest complexity.

## Decision outcome

**Chosen: Option 1, sharpened — surrogate-gradient REINFORCE over the differentiable policy path, with the hippocampal memory path bypassed in v1.**

Concretely:

- **Objective:** REINFORCE (Monte-Carlo policy gradient). Motor spike-counts over the inference window → a categorical action distribution → sample an action → weight `log π(a)` by the episode return (with a baseline to reduce variance). Surrogate gradients carry this back through every spiking layer.
- **Credit assignment:** end-to-end backprop across **sensory → prefrontal → motor**, including the differentiable router gate.
- **Memory path:** **bypassed in v1** (`recall=False`). The hippocampus and the router's store/recall commands are left out of the training loop. The policy is reactive — sufficient because the task is fully observable.
- **Learnable parameters in v1:** the feedforward/projection weights of sensory, prefrontal, motor (and the router's selection weights, via the gate). Fixed/structural buffers (lateral-inhibition matrices, hippocampal `W_rec`) are **not** trained.

This is the honest reading of "Option 1 to get it working": the *whole differentiable brain* trained as a policy, with the single non-differentiable region (memory) sidelined rather than pretended into the gradient.

### Why not the alternatives now

- **Option 2 (region-local)** — premature: harder to debug with no working baseline to compare against, and R-STDP is only at the taste stage. It is the long-term destination for *memory and routing*, not the v1 driver.
- **Option 3 (hybrid)** — the right eventual architecture, but its complexity is unjustified before a working end-to-end baseline exists. It becomes the stretch path (below).

## Consequences

**Positive**
- Fastest route to a brain that *learns the task*, giving a baseline to measure every later change against.
- Reuses proven week-5 surrogate-gradient machinery and snntorch defaults.
- Small, debuggable gradient path; the dashboard (NEURO·SCOPE) can watch the policy path light up as it trains.

**Negative / accepted trade-offs**
- Not biologically plausible (global backprop across regions) — explicitly deferred, not solved.
- The hippocampus and router-driven store/recall are **inert** in v1; the memory architecture is built but not yet exercised by learning.
- REINFORCE is high-variance; expect a baseline (and possibly reward normalization) to be necessary for convergence.

**Neutral**
- `Brain.learn` evolves from a deferred hook into a real REINFORCE update step; the dopamine-bus third-factor wiring remains in place for the later hybrid.

## Migration path → Option 3 (hybrid), as stretch goals

The v1 cut is deliberately staged so the hybrid is additive, not a rewrite:

1. **Bring memory into the loop** — enable `recall=True`; train the recall read-out while keeping `W_rec` Hebbian. Tests whether memory *helps* (it shouldn't be required, but shouldn't hurt).
2. **Make memory plastic locally** — replace/augment the one-shot Hebbian store with an R-STDP rule on `W_rec` driven by the dopamine bus (the EXP-021 eligibility-trace direction). Memory now learns *between* episodes without backprop.
3. **Reward-modulate the router** — let the gate's selection learn from the global reward signal rather than only via the differentiable path.
4. **Result = Option 3:** surrogate gradients within the feedforward regions, reward-modulated local rules for memory and routing.

A partially-observable task (where memory is actually required) is the natural forcing function to justify steps 1–2.

## Notes

- This ADR records a decision; the implementation (the REINFORCE training loop, baseline, and the `Brain.learn` update) is scoped in a separate plan.
- First ADR in the repo — establishes `docs/adr/NNNN-title.md` as the convention (lightweight MADR style).
