# EXP-047 design - fine-tune the encoder during RL

> **PRE-REGISTERED. Committed before any number exists.** Every threshold below is fixed at
> commit time. If a threshold changes later, the change must be a separate commit, dated, with
> its reason, and made before the data it applies to exists - the EXP-039 pattern (`0c77e5b`,
> `bb88580`).

## 1. Why this experiment

Week 20 priced the only lever the project had been using. The depth series is a **budget**
series: 4.4x the episodes buys about one depth (EXP-044, EXP-046), the gain comes from total
spend rather than deep-end exposure (EXP-045, back-loading a fixed budget scored -0.0479 at
p 0.0010), and the curve across 10k/25k/44k is **log-linear at about 0.22 success per log10 of
spend, with no knee**.

That is a solved and unattractive lever. Depth 8 at matched exposure is ~194,000 episodes for
~0.18, and a log-linear curve has no cheap fraction to exploit.

**Fine-tuning the encoder during RL is the only untried lever that could change the exchange
rate rather than pay it.** Every cube result in this project runs on a frozen encoder with a
`Linear(64 -> 6)` head: 390 trainable parameters. EXP-039 showed the encoder *can* be improved
by gradient descent (inverse-model pretraining, +0.3396 probe top-1 at depth 4, 12-0 seeds).
What has never been tested is whether the RL objective itself can keep improving it.

## 2. What is being changed, precisely

The encoder is frozen **by construction**, not by configuration. `reinforce.action_distribution`
hardcodes:

```python
with torch.no_grad():
    out = brain.step(obs, store=store, recall=recall, record=False, generator=generator)
```

so no gradient can reach `brain.sensory` regardless of what the optimizer is given. This is a
trainer change.

**The change:** a new `CubeConfig.encoder_lr: float | None = None`. When set, the `no_grad`
context becomes a no-op and the optimizer gains a second parameter group over
`brain.sensory.parameters()` at that learning rate. The head keeps `cfg.lr`.

Three properties were **measured on 2026-08-20 before this spec was written**, not assumed:

| property | measurement |
|---|---|
| gradients reach the encoder through snnTorch's surrogate | `fc1.weight` grad absmax 3.445e-01, 3072/18432 entries nonzero; `fc2.weight` 1.933e-01, 4992/8192 |
| enabling grad does not change the forward | `brain.step` under grad returns a **bit-identical** concept and action to the `no_grad` path, same generator seed |
| cost per step | frozen 56.17 ms, fine-tuned 74.87 ms = **1.33x** |

The second property is the seam. `encoder_lr=None` must reproduce every prior cube record
**byte-for-byte**, and a test asserts it against a pre-change baseline - the same discipline
`tests/training/test_encoder_seam.py` already applies to `encoder_state_path`.

### 2.1 This is a different architecture and is reported as one

| | frozen (every prior cube result) | EXP-047 fine-tuned |
|---|---|---|
| head | `Linear(64 -> 6)` = 390 | `Linear(64 -> 6)` = 390 |
| `sensory.fc1` | frozen (18,560) | **trainable** |
| `sensory.fc2` | frozen (8,256) | **trainable** |
| **trainable total** | **390** | **27,206** (70.0x) |

"The same 390 trainable parameters" is load-bearing in every RESULTS.md and in the whole visual
story. **A fine-tuned arm is never a cell in the depth series.** It gets its own row, its own
parameter count, and its own name in any figure or write-up.

## 3. The confound, and why it does not need its own arm

Fine-tuning adds trainable parameters **and** compute per step. After EXP-044/045/046 the
leading alternative explanation for any win is simply budget.

The two ways to control it both cost something:

- match **episodes** -> the fine-tuned arm gets 1.33x the compute;
- match **compute** -> the frozen control gets 1.33x the episodes, which EXP-046 prices.

**This spec matches episodes**, and prices the residual analytically instead of running a third
arm. EXP-046 measured depth 6's budget curve at **0.22 success per log10 of spend**, log-linear
with no knee across 10k/25k/44k. So:

> 1.33x compute is worth **0.22 x log10(1.33) = +0.027** at depth 6.

That is what makes the +0.05 bar in Claim 1 meaningful here rather than merely conventional: it
sits **1.9x above** the gain that the extra compute alone would buy a frozen encoder.

**What this cannot rule out.** The pricing assumes the budget curve measured on a *frozen*
encoder transfers to an arm whose compute is spent differently (backward passes, not extra
episodes). It also assumes log-linearity holds down to 1.33x, well inside the measured 4.4x
range but not separately verified there. A delta between +0.027 and +0.05 is therefore
**ambiguous and is pre-committed to be reported as ambiguous**, not as a small win.

## 4. Test bed

**Depth 6 at 10,000 episodes**, paired against EXP-043's `exp043_capped_d6` cells - not re-run.

- Baseline **0.1800** (EXP-043), with headroom to depth 6's 44k ceiling of 0.3225.
- Same 12 seeds, same EXP-040 pretrained encoders at init, same curriculum `(1..6)`, same
  `max_steps_by_depth=((1,2),)`, same `entropy_beta=0.0`, `normalize_advantages=False`.
- **One variable versus EXP-043: whether the encoder is trainable.**

Depth 7 was considered and rejected: its 10k baseline is 0.0621, too near the floor for a paired
test to be sensitive.

Fine-tuning runs **throughout the curriculum**, not only at the final stage. That keeps the arm
at one variable. The risk it accepts is that stage 1 (depth 1, capped at 2 steps by EXP-042) now
shapes the encoder as well as the head; Claim 4 instruments it.

## 5. The learning rate, and how it is chosen

`encoder_lr` has no prior in this project. Pretraining used 3e-3 at batch 256 against a
supervised MSE objective; REINFORCE's gradient here is far noisier and arrives one short episode
at a time, so the usable rate is expected to be smaller by orders of magnitude, not by a factor.

### 5.1 The pilot

**Grid:** `encoder_lr` in {1e-3, 1e-4, 1e-5}, log-spaced, bracketing pretraining's 3e-3 from
below.

**Seeds: 12 and 13. Deliberately NOT 0-11.** EXP-039 §6a refused to select its learning rate by
the probe because the probe was its outcome metric, and recorded that the probe would have
picked the other rate. The same trap applies here. Running the pilot on seeds outside the
confirmatory set means **no seed contributing to any claim was used for selection**, so the
selection cannot tune either claim's metric and n stays at 12.

Seeds 12-23 have no pretrained encoders. They are generated by EXP-040's `pretrain_one`
unmodified (~20 min at 12 workers), which is a prerequisite of the fallback experiment anyway.

**Full 10,000 episodes, not a shortened run.** A short pilot would measure damage at 1/5 the
exposure the gate is applied at, and force an extrapolation. At full exposure the gate is
measured at the exposure it governs. Cost: 6 cells, ~2.5 h.

### 5.2 The selection rule - mechanical, probe-only, fixed here

`select_lr.py` reads **only the probe outputs**. It never opens a success rate. This is enforced
by construction, not by intention.

1. **Gate.** An `encoder_lr` passes only if, on **both** pilot seeds, the fine-tuned encoder's
   depth-4 probe top-1 is no more than **0.02** below that same seed's own starting encoder.
   0.02 is under **6%** of the +0.3396 that pretraining bought at depth 4 (EXP-039), so the gate
   says: fine-tuning may not give back more than a sixteenth of what pretraining earned.
2. **Choice.** Among the rates that pass, take the highest mean depth-4 probe top-1 across the
   two seeds. Tie-break to the **larger** rate.
3. **Stop condition.** If **no** rate passes the gate, the chain **halts and does not dispatch
   the confirmatory arm**. "REINFORCE's gradient damages the pretrained representation at every
   rate tried" is a real result and is written up as one. It is not a reason to widen the grid
   opportunistically; a wider grid would be a new pre-registration.

The pilot's success rates exist on disk from the moment the pilot finishes. **They are not read
until the selected rate is recorded in the run log.** This mirrors EXP-043, whose `aggregate.py`
was written after the records were fetched but before any value in them was read.

## 6. Claims

All comparisons **paired per seed**, exact permutation over all `2**n` sign flips. No scipy.

### Claim 1 - PRIMARY. Does fine-tuning beat the frozen encoder at matched episodes?

Held-out success at depth 6, EXP-047 (selected rate, seeds 0-11) minus EXP-043 `exp043_capped_d6`
(same seeds). Exact permutation over `2**12 = 4096`.

**CONFIRMED at delta >= +0.05 and p <= 0.05.**

+0.05 is the bar EXP-043, EXP-045 and EXP-046 all used, so the four are directly comparable, and
§3 shows it clears the +0.027 that compute alone buys. A delta between **+0.027 and +0.05**, or
positive at p > 0.05, is **ambiguous**: it refutes the strong reading without showing the effect
is zero, and §3's caveat is reported with it.

### Claim 2 - MECHANISM. Did RL improve the representation, or only fit the head to it?

Probe top-1 on each fine-tuned encoder against **its own EXP-040 starting encoder**, seeds 0-11,
depths 3-6, using EXP-033's `probe.py` imported unmodified and EXP-039's batched feature path.
Depth 4 is the headline depth, as in EXP-039.

> [!warning] This claim carries a **pre-registered asymmetry** and must be read through it.
> RL fine-tunes on the RL split; the probe holds out a **different** split. Most states the
> probe scores were therefore seen by the encoder during fine-tuning.
>
> - **Degradation is clean.** No leak can cause the probe to fall.
> - **Improvement is confounded** with memorising states the probe scores.
>
> The **leak-free slice** is reported alongside: depth 6's RL held-out states (200 per seed),
> which neither arm trained on during RL. An improvement that survives on that slice is
> materially stronger than one that does not, and must be reported as the weaker of the two if
> they disagree.

Descriptive, with the direction pre-committed: a **rising score with a falling probe** localises
the effect to the head; a **rising score with a rising probe** is the representation improving;
a **falling probe with a falling score** is catastrophic forgetting and is the expected failure
mode at too high a rate.

### Claim 3 - ARCHITECTURE ACCOUNTING. Descriptive, no p-value.

Report trainable parameters (390 -> 27,206), measured wall-clock per step (1.33x), and the
budget-equivalent gain from §3 (+0.027) in the results table itself, next to the delta. **The
fine-tuned arm is never presented as a depth-series cell.**

### Claim 4 - COLLAPSE. Descriptive, no p-value.

The deepest stage's `stage_trace` entropy and `train_solved_frac` against EXP-045's collapse
signature (entropy 0.5914 -> 0.0979, min 2.7e-06, solve rate 0.0218). Fine-tuning gives the
optimizer a 70x larger surface to collapse, and stage 1's 2-step depth-1 cap now shapes the
encoder as well as the head, so this is where that shows up if it does.

### Claim 5 - THE NULL IS PRE-COMMITTED AND IS A REAL RESULT.

If Claim 1 refutes, the finding is: **REINFORCE's gradient cannot improve the encoder faster
than budget can be bought**, and the frozen-encoder / 390-parameter result stands exactly as
published, with no caveat added. Combined with EXP-046's pricing of budget, that closes the last
untried lever on the current objective, and the next move is a **different pretraining objective**
(value or heuristic, rather than inverse dynamics) - not more RL, and not more episodes. This
redirection is stated now so a null is not later left open-ended.

## 7. Failure modes this design accepts

- **The rate does not transfer across seeds.** Selected on seeds 12-13, applied to 0-11. If the
  usable rate is strongly seed-dependent, the confirmatory arm is run at a rate that is wrong for
  some seeds. Claim 4's per-seed entropy traces are what would show this.
- **The gate is a representation gate, not a performance gate.** It can pass a rate that
  preserves the probe while helping the policy not at all. That is deliberate: the alternative
  selects on the outcome.
- **1.33x is measured on the VPS** (2 cores, `torch.set_num_threads(1)`). The ratio is expected
  to hold on the laptop but the absolute times will not.

## 8. Execution

Chained on the laptop `SwizzlesDuo`, unattended, per `docs/playbooks/remote-experiment-runs.md`.

Costs below are derived by the **playbook's** method, not from `CLAUDE.md`'s 90 ms figure:
`wall_hours = steps * 0.153 / 3600 / workers`, with the 1.33x applied to training steps only
(evaluation uses `greedy_action`, which takes no `grad_brain` and stays at frozen cost). One
depth-6 cell at 10,000 episodes is 94,962 training steps plus 6,000 evaluation steps: **4.29
core-hours frozen, 5.62 fine-tuned.** Ten workers, not sixteen - the laptop is memory-bound at
~920 MB private per worker.

| phase | what | core-h | workers | wall |
|---|---|---|---|---|
| 0 | EXP-040 `pretrain_one`, seeds 12-23 | - | 12 | ~20 min |
| 1 | EXP-047 pilot: 3 rates x seeds {12,13} x 10,000 episodes | 33.7 | 6 | ~5.6 h |
| 2 | `select_lr.py` - probe-only, mechanical, may **halt** the chain | - | - | ~10 min |
| 3 | EXP-047 confirmatory: selected rate, seeds 0-11 | 67.5 | 10 | ~6.7 h |
| 4 | Fallback: EXP-040 and EXP-043 depth 5, seeds 12-23, **both arms** | 94.0 | 10 | ~9.4 h |
| | **total** | | | **~22 h** |

> **Phase 4 corrects the handoff.** `SESSION-HANDOFF-2026-08-20.md` §3 budgets ~4.6 h for "the
> RL arm" on 12 new seeds. EXP-043's Claim 1 is a **paired** delta against EXP-040 depth 5, and
> EXP-040 phase 2 only ever ran seeds 0-11 - so seeds 12-23 need **both** arms or there is no
> pair to compute. The fallback is 24 RL runs, not 12.

Estimates in this project have consistently run **short** (EXP-046: 15.5 h against 23 h
estimated; EXP-044 arm B: 25.4 h against 32 h), so these are upper bounds. At EXP-046's ratio
the chain is nearer **15 h**.

Phase 4 runs **even if phase 2 halts the chain.** The fallback is independent, already
justified, and needs no result from EXP-047, so a halted selection should still leave the
machine doing useful work rather than idle.

### 8.1 A defect this design's own test caught, recorded because it would recur

`run_cube_baseline` passes `feature_fn=readout` on the **training** call for every readout
including `"concept"`; only the two evaluation calls pass `None`. `MemoryReadout.__call__`
wrapped its whole body in `torch.no_grad()`, so the identity concept branch - on the policy path
of every cube run ever recorded - **detached the concept**. The first fine-tuning implementation
therefore trained nothing: `fc1.weight` moved by exactly 0.0 while the run looked entirely
normal, produced an ordinary-looking success rate, and passed every neutrality assertion.

It was caught only by a complement test asserting the encoder MUST move (`drift > 1e-3`), which
is the `CLAUDE.md` test-strength rule doing exactly what it exists for. The fix returns the
concept before the `no_grad`; that is numerically inert for frozen runs, and the byte-identity
tests prove it.

**Any future "make X trainable" change on this codebase inherits the same trap.** Verify the
gradient arrives at the parameter, never that the switch is set.

### 8.2 Pre-flight validation, run 2026-08-20 before dispatch

Every number here predates the experiment and none of them is a result.

| check | outcome |
|---|---|
| `encoder_lr=None` and `encoder_lr=0.0` reproduce the pre-change baseline | asserted to 1e-6 |
| fine-tuned run serialises its encoder | 110 KB, `fc1` drift 3.154e-03 over 60 episodes |
| trainable parameters actually change | 390 -> 27,206 in the record |
| **run-level cost model** | predicted 1.10x on the smoke shape, **measured 1.09x** |
| memory overhead per worker | +9 MB (293 -> 302 MB peak), so 10 workers stays safe |
| selection rule, both outcomes | choose and **halt** both exercised on synthetic probe data |
| halt propagates | confirmatory arm refuses to start |
| `--depths` neutrality on EXP-040/043 | default cell sets bit-identical |

**Claim 2's instrument was falsified, not assumed.** A metric that cannot move is not a metric,
and the smoke run's probe delta was exactly +0.0000. Deliberately damaging an encoder
(`fc1 += N(0, 0.5)`) moved depth-4 top-1 from **0.8507 to 0.6493, a delta of -0.2015** - about
**10x the 0.02 gate**. So the gate can fire, and the +0.0000 is a real reading of a drift too
small to flip any of ~2,978 held-out decisions, not a broken probe. Features were confirmed to
differ (max 0.156, i.e. 5 of 32 spikes), so the change does reach the code; the concept rate is
quantised in 1/32 steps and top-1 accuracy is discrete, so small drift genuinely reads as zero.
