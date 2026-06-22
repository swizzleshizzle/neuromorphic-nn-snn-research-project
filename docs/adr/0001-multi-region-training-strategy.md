# ADR-0001 — Multi-Region Training Strategy

- **Status:** Accepted
- **Date:** 2026-06-18
- **Phase:** 2 (multi-region brain), Step 2.3 — "how does the whole brain learn?"
- **Deciders:** project author
- **Supersedes / superseded by:** none

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
