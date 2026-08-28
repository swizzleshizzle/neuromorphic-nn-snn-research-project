# EXP-053 design - a learned critic, and a neuromodulated plasticity gate

> **PRE-REGISTERED. Committed before any number exists.** Thresholds fixed at commit time.
> **Date:** 2026-08-28 · **Phase:** 3, Stage 3 · **Grounds:** `docs/handoffs/WEEK-21-KICKOFF.md`
> section 2, `docs/handoffs/SESSION-HANDOFF-2026-08-27.md` section 0, EXP-044 through EXP-052.

## 0. The premise correction that reshaped this spec

**The kickoff justified Stage 3 on a failure that is not in the current recipe.** It cited
EXP-045's depth-7 entropy collapse, `0.591 -> 0.098` on a 2.2% train solve rate, and proposed
pre-registering the mechanism claim on the entropy trace.

That collapse belongs to EXP-045's **back-loaded curriculum** `(1,1,1,1,1,1,10)` - the arm that
experiment **refuted** (paired -0.0479, W-L-T 0-11-1, p 0.0010, 9/12 zeros, modal action frac
0.847). It is a property of a configuration that was measured and rejected, not of the recipe
everything since has used.

Read from the stage traces of the uniform-curriculum depth-7 records, computed 2026-08-28 from
the committed `outputs/` of EXP-044 and EXP-051. **Regenerate the whole table with**

```bash
.venv/bin/python experiments/053_neuromod_stage3/premise_check.py
```

| depth-7 arm | deepest-stage entropy, first -> last 10% | train solved | held-out success | seeds at exactly 0.0000 |
|---|---|---|---|---|
| EXP-044 arm A, E0 frozen, 10k | 0.440 -> 0.380 | 0.0823 | 0.0621 | 1 / 12 (s6) |
| EXP-044 arm B, E0 frozen, 44k | 0.296 -> 0.281 | 0.1998 | 0.1971 | 0 / 12 |
| EXP-051, E1 frozen, 10k | 0.368 -> 0.361 | 0.1548 | 0.1471 | 1 / 12 (s5) |

**Two things follow, and both are load-bearing for this design.**

**(a) The depth-7 failure in the current recipe is discrete, not gradual.** Entropy holds. What
happens instead is that roughly one seed in twelve dies outright, at exactly 0.0000, and it is a
**different seed under each encoder** (s6 with E0, s5 with E1). That is a bad basin, not a bad
seed. `dead_seeds` is therefore a pre-registered reported quantity here (Claim 5).

**(b) The entropy trace is a FOURTH instrument with the reversal signature, and it is excluded.**
*Within* an arm it tracks success strongly - Spearman **+0.881** in EXP-044 arm A, **+0.804** in
EXP-051. *Between* arms the sign flips: the better arm has the **lower** entropy (0.281 at 0.1971
success, against 0.380 at 0.0621). A mechanism claim of the form "the critic keeps entropy higher,
therefore the policy is better" is read across arms, in exactly the direction where the sign
inverts.

This is the same shape as the retired EXP-033 probe and as EXP-050's Claim 4, where a
pre-registered inference was satisfied and still wrong. **Entropy is recorded in every arm and may
be read only within an arm. It appears in no claim.** Mechanism is `revisit_rate` and `optimality`,
per `CLAUDE.md`.

**What survives as the critic's justification.** Not the collapse. At depth 7 the deepest stage
solves **15.5%** of training episodes on the best current recipe, so **roughly five in six episodes
return no reward** and REINFORCE's advantage against a scalar EMA baseline is mostly noise. That is
a signal-density argument and it stands on its own. It is weaker and more honest than the one the
kickoff wrote.

## 1. What this experiment is for

Two increments, settled with Michael on 2026-08-27 after the fork was deliberately left unmade by
the previous session.

**3a - a learned critic.** Replace the scalar EMA baseline with a state-dependent one.

**3b - a neuromodulated plasticity gate.** Gate encoder plasticity on `NeuromodBus`. This is the
increment that makes the bus **load-bearing**, and it is the only one that licenses the sentence
"a neuromodulatory signal is on the critical path".

> [!danger] THE TRAP, CARRIED FORWARD FROM THE KICKOFF
> **3a is actor-critic.** Routing a scalar through a bus object does not make it neuromorphic. If
> nothing *reads* the bus, the routing is decoration, and the honest description of arm B is
> **"a learned baseline reduces gradient variance"** - a real result, and an RL one.
>
> `CLAUDE.md` and `road-to-a-solved-cube` both record that nothing neuromorphic participates in the
> learning yet. Week 20 changed that once, when the spiking encoder started training. **Arm B must
> not be written up as if it changed it twice.**

**They are run in the regime where each has the best chance, not in one regime for tidiness.** The
critic's rationale is sparse reward, which binds at depth 7. The gate is depth-agnostic, so it runs
at depth 6 where the signal is strongest and a free 12-seed control already exists.

## 2. Design

**Three new arms of 12 seeds (0 to 11). Two controls exist on disk and are NOT re-run.**

| arm | depth | what changes | control | measured cost |
|---|---|---|---|---|
| **B** | 7 | learned state-dependent baseline replaces the scalar EMA | EXP-051, **0.1471** | ~6.3 h |
| **G** | 6 | encoder plasticity gated on `bus.learning_enabled` | EXP-047 confirmatory, **0.2700** | ~6.4 h |
| **R** | 6 | same, gate replaced by a rate-matched coin flip | arm G | ~6.4 h |

Both controls were verified locally on 2026-08-28 against their published values: EXP-051 seeds
0-11 mean **0.1471** (sd 0.0784); EXP-047 `exp047_ft_d6_lr0.0001` restricted to seeds 0-11 mean
**0.2700** (sd 0.0810), matching its RESULTS.md exactly. EXP-047's records also carry pilot seeds
12 and 13 at the same lr; **the control is seeds 0-11 only.**

### 2.1 Arm B - the critic

Config copied from EXP-051 field for field: depth 7, 10,000 episodes, `curriculum=(1..7)`,
`max_steps_by_depth=((1,2),)`, `entropy_beta=0.0`, `normalize_advantages=False`, `max_depth=7`,
`lr=1e-2`, `gamma=0.99`, `baseline_beta=0.1`, `heldout_cap=200`, the E1 encoders from EXP-047
loaded via `encoder_state_path` and **frozen** (`encoder_lr=None`).

One variable: the baseline.

```
V = Linear(64 -> 1)                 # 65 parameters, on the same concept the policy head reads
advantage_t = G_t - V(s_t)          # replaces  G_t - baseline
critic loss = MSE(V(s_t), G_t)
```

**Monte-Carlo advantage, not TD bootstrapping. This is a deliberate deviation from the kickoff,
which specified `delta = r + gamma*V(s') - V(s)`.** `train_episode` already computes
returns-to-go, so `G_t - V(s_t)` is the strictly smaller change, carries no bootstrapping bias, and
still yields a per-step reward-prediction error. The kickoff's own design question 2 says to prefer
the smaller change first, and the invariance argument it raises is only needed if the signal is
added to the reward, which it is not here.

**Critic input is the frozen concept, not its own spiking region** - the cheap version first, on
the EXP-025 precedent that a wider readout did not beat a linear one.

`trainable_params` becomes **455** (390 head + 65 critic). Recorded per record, as always.

### 2.2 Arm G - the neuromodulated gate

Config copied from EXP-047's confirmatory arm field for field: depth 6, 10,000 episodes,
`curriculum=(1..6)`, `encoder_lr=1e-4` (the value EXP-047 selected under a rule that could not see
a success rate), E0 pretrained encoders, 27,206 trainable.

One variable: **when the encoder is allowed to move.**

```
per episode:
    dopamine = mean_return - baseline          # Brain.learn(mean_return, baseline)
    bus.set(dopamine=dopamine)                 # the bus gets its first writer in the cube loop
    threshold  = running median of |dopamine| over all episodes so far
    if bus.learning_enabled:  encoder_opt.step()
    head_opt.step()                            # the head ALWAYS steps
```

**`Brain.learn()` has no caller anywhere in the cube training loop today, and nothing in the
codebase reads `bus.learning_enabled`.** Verified 2026-08-28: `rg '\.learn\(|bus\.'` over
`src/neuromorphic/training/` returns nothing. This arm supplies both.

**The threshold is a running median, not a constant.** `NeuromodBus.learning_threshold` defaults to
0.5, which is meaningless against a return scale set by `solve_reward=10.0` and
`step_penalty=-1.0`. A quantile self-calibrates, needs no tuned number, and puts the realized
update rate near 50% by construction. `|dopamine|` rather than signed dopamine, so the gate fires
on **surprise in either direction**, which is what a plasticity gate should do and what keeps the
rate stable as `baseline` tracks.

**Warmup: the first 100 episodes always update** (1% of the run), because the median is undefined
before there is history. Warmup episodes enter the median history.

Each record reports **`gate_rate`**, the realized fraction of episodes on which the encoder stepped.
Arm R needs it, and a rate far from 0.5 is itself diagnostic.

> [!danger] IMPLEMENTATION TRAP - ZEROING THE GRADIENT DOES NOT GATE ADAM
> The control builds **one** Adam with two parameter groups. Gating by zeroing the encoder group's
> gradients **would still move the encoder**, because Adam applies `exp_avg` on every `step()` and
> a zero gradient only decays momentum, it does not suppress the update. The arm would then be
> "encoder moves on every episode, slightly less on ungated ones", which is not what it claims.
>
> **The gate must be a separate optimizer.** Adam state is per-parameter, so splitting one
> two-group Adam into a head optimizer and an encoder optimizer is mathematically identical while
> the gate is always true - which is exactly the seam test in section 4.
>
> This is the same family of failure as the `no_grad` trap in `CLAUDE.md`: the switch is set, the
> run looks perfectly ordinary, and the arm is not the arm.

### 2.3 Arm R - the rate-matched random gate

Identical to arm G except the gate is `bernoulli(p_s)` where `p_s` is **arm G's realized
`gate_rate` for the same seed s**, paired. Drawn from a dedicated `random.Random` stream that never
touches the action sampler, the scramble stream or the Poisson generator, so the two arms remain
comparable in everything but which episodes were chosen.

**Why this arm exists, and it is not optional.** A gated arm also performs **fewer encoder
updates**. If G matches its always-on control, that is either "surprise-gating selects the updates
that matter" or "half the encoder updates were redundant" - and EXP-049 already found the encoder
gain does not compound (a constant ~+0.05 per round), which makes the second reading the more
likely one. Without R, the outcome we most expect is uninterpretable, and it is also the outcome
easiest to write up as a neuromorphic success.

This is the EXP-030 lesson applied in advance: **ask what a control holds fixed besides the thing
you named.** There, `memory` beat a shuffle-null by 10.8 points and the amnesic control by 1.2, and
the primary comparison had been measuring the harm of incorrect memory rather than the benefit of
correct memory. Two arms would have published a false positive. Three did not.

### 2.4 What arm G's absolute number may and may not be compared to

Depth 6 carries several numbers and they are not interchangeable. Recorded here so the write-up
cannot drift:

| number | what it is |
|---|---|
| 0.1800 | EXP-043, E0 frozen, 10k. The frozen baseline. |
| **0.2700** | **EXP-047 arm C, E1 fine-tuned JOINTLY during RL, co-adapted head. Arm G's control.** |
| 0.3112 | EXP-048 arm B, EXP-047's encoder **frozen** with a **fresh head**. |
| 0.3525 | EXP-049, the best depth-6 recipe, at 2.66x compute. |
| 0.3225 | EXP-046, E0 frozen at 44,000 episodes. Budget, not architecture. |

**Arm G's control is 0.2700 and only 0.2700**, because arm G is a fine-tuning-during-RL arm and the
gate is the single variable. Its absolute number must not be read against 0.3112 or 0.3525, which
were produced by a different recipe.

**One consequence is worth stating before the numbers exist, because it is an interpretation and
interpretations are what got over-read twice last week.** EXP-048 found that freezing E1 and
training a fresh head (0.3112) **beats** joint fine-tuning (0.2700) by +0.0412 - the co-adapted
head costs. A gate that lets the encoder move less could plausibly reduce that co-adaptation and
land arm G between the two.

**That is a hypothesis, not a claim, and it is not pre-registered.** It is written down only so
that if arm G lands near 0.31 nobody discovers this reading afterwards and presents it as a
prediction. Claim 3 still governs: even a win of that size means nothing about the bus unless G
also beats R.

## 3. Claims

Paired per seed against the same seed of the control. Exact permutation over `2**12` = 4096 sign
flips - no scipy in the venv, and at n=12 the exact test is cheap and assumption-free.

### Claim 1 - PRIMARY, the critic. Arm B minus EXP-051.

**CONFIRMED at `>= +0.05` with `p <= 0.05`**, the standing bar.

A **negative** result at the same bar is reported as a refutation, not a null: it would say a
learned state-dependent baseline does not help at the frontier, which bounds a standard RL
assumption in this regime and is worth the same page space.

### Claim 2 - PRIMARY, the gate. Arm G minus EXP-047 confirmatory.

**CONFIRMED at `>= +0.05` with `p <= 0.05`.**

**Direction is deliberately not pre-committed.** A significant loss is as informative as a win: it
would say that restricting encoder plasticity to surprising episodes actively costs, which
constrains any future plasticity design here.

### Claim 3 - ATTRIBUTION. Arm G minus arm R. This claim governs how Claim 2 may be written.

**Any statement that the gate's SIGNAL matters requires G to beat R at `>= +0.03`, `p <= 0.05`.**
The bar is lower than the primary because this contrast holds the update *rate* fixed and isolates
only which episodes were chosen.

Fixed in advance, and binding:

| G vs control | G vs R | what may be claimed |
|---|---|---|
| G wins | G beats R | **The neuromodulatory gate works.** The bus is load-bearing and the signal is doing the work. |
| G wins | G ~ R | **Fewer encoder updates is the whole effect.** Report as an efficiency finding. The neuromorphic claim is NOT made. |
| G ~ control | G ~ R | **Encoder updates are redundant at this rate.** The neuromorphic claim is **refuted, not deferred.** |
| G loses | any | The gate costs. Report the loss and the size. |

**"G ~ R, so we need a better gate" is not an available conclusion from this experiment.** It is
the escape hatch this table exists to close.

### Claim 4 - MECHANISM. `revisit_rate` and `optimality`, paired, both contrasts.

These are the instruments that carried the mechanism in EXP-048 and EXP-049 and are present in
every record since EXP-029. Reported with paired deltas and exact p, descriptive - no bar.

**No probe number appears in this experiment.** The EXP-033 probe is retired as a policy predictor
(EXP-049: 0-12, p 0.0005 while policy rose; EXP-050: 12-0, p 0.0005 while policy halved).

### Claim 5 - DEAD SEEDS. Count at exactly 0.0000 per arm.

Descriptive, no bar, n=12 is far too small to test a 1-in-12 rate. Reported because section 0 found
it is the actual depth-7 failure mode and because a critic that rescues dead seeds while moving the
mean little would be invisible to Claim 1.

### Claim 6 - COMPUTE. The confound, priced in advance.

Per-step wall clock measured for every arm, as EXP-047 did (it measured 56.17 -> 74.87 ms, 1.33x).
Priced against EXP-046's budget curve: **0.22 per log10 at depth 6, 0.210 per log10 at depth 7.**

**The asymmetry is stated now, before the numbers.** Arm B **adds** 65 parameters and one forward
per step, so a win must clear its own compute cost. Arms G and R do **strictly fewer** optimizer
steps than their always-on control, so **a win there cannot be a compute artifact, but a loss might
be** - a slower-learning encoder that simply moved less.

Any arm whose compute-adjusted delta lands inside the ambiguous band is reported as ambiguous, not
rounded toward the hypothesis.

## 4. Implementation

Library changes in `src/neuromorphic/`, drivers in `experiments/053_neuromod_stage3/`, per the
standing placement rule.

### 4.1 Seams

`CubeConfig` gains two fields, both `None` by default and both strict no-ops when unset:

- `critic_lr: float | None` - `None` keeps the scalar EMA baseline, which is every cube record from
  EXP-029 onward.
- `plasticity_gate: str | None` - `None` | `"dopamine"` | `"random"`. Requires `encoder_lr` to be
  set; refused otherwise, in the style of the existing `encoder_lr` guards, because gating
  plasticity that does not exist would silently do nothing.
- `gate_rate_by_seed: tuple[tuple[int, float], ...]` - arm R's per-seed rates, empty by default.

`train_episode` gains `critic`, `critic_optimizer` and a `gate` callable. Defaults reproduce the
executed statements exactly.

### 4.2 Tests, written to the test-strength rule

Every test below must **fail against pre-change code**. A test that passes either way is
documentation, not a test.

1. **Byte-identical no-op.** With `critic_lr=None` and `plasticity_gate=None`, a short run
   reproduces a pre-change baseline record byte for byte. Follows `test_encoder_finetune_seam.py`.
2. **Byte-identical always-open gate.** With the gate forced always-true, the split
   head/encoder optimizers reproduce the single two-group Adam byte for byte. **This is the test
   that proves the Adam-momentum trap in section 2.2 was avoided**, and it fails against the
   naive zero-the-gradients implementation.
3. **The gate gates exactly.** Record `fc1.weight` before and after each episode over a short run
   and assert it changed on **exactly** the episodes the gate opened, and on no others. A
   `count_nonzero(delta) > 0` check would pass either way and is forbidden - it is the assertion
   that hid `Hippocampus.store()` assigning instead of accumulating.
4. **Gradient ARRIVES at the critic.** Assert `V.weight` moves by strictly more than 0.0 after one
   update, and that its `.grad` is not `None`. `CLAUDE.md` records that EXP-047's first fine-tuning
   implementation trained nothing, moved `fc1.weight` by exactly 0.0, and produced a perfectly
   ordinary success rate. A critic that is silently frozen looks exactly like a null result.
5. **The advantage actually changed.** Assert that with a critic, the per-step advantages differ
   from `G_t - baseline` on a fixed seed. Guards against a critic that is built, optimized and
   never read.
6. **Rate matching.** Arm R's realized rate is within a tolerance of the rate it was given, and its
   gate stream leaves the action and scramble streams bit-identical to arm G's.

### 4.3 Critic learning rate has no prior, and is selected before the main run

`encoder_lr` had no prior either, and EXP-047 settled it with a pilot governed by a rule fixed in
advance and executed by a script that **could not see a success rate**. Same discipline here.

- Pilot seeds **12 and 13** - not 0-11, which are the confirmatory seeds.
- Grid `{1e-3, 1e-2, 1e-1}`, at depth 7, the regime the critic is used in.
- **Selection rule, fixed now:** choose the lr maximising the critic's **explained variance of
  realized returns**, `1 - Var(G_t - V(s_t)) / Var(G_t)`, measured over the deepest stage. Ties
  broken toward the smaller lr. The selection script reads the critic fit only and has no access
  to `success_rate`.
- ~3.2 h, 6 cells, one wave at 6 workers.

An untuned critic that diverges would make Claim 1 a null for the wrong reason, and "the learning
rate was arbitrary" would be an unanswerable alternative explanation.

## 5. Dispatch

Serial. R depends on G's realized rates; B depends on the pilot.

| step | what | cost |
|---|---|---|
| 0 | critic-lr pilot, seeds 12-13, depth 7 | ~3.2 h |
| 1 | arm B, 12 seeds, depth 7 | ~6.3 h |
| 2 | arm G, 12 seeds, depth 6 | ~6.4 h |
| 3 | arm R, 12 seeds, depth 6, rates from step 2 | ~6.4 h |
| | **total** | **~22.3 h** |

**6 workers, not 10.** 12 cells on 6 workers is two clean waves; EXP-051 measured 6.3 h that way
against a 7.2 h estimate, and the handoff records that 12 cells on 10 workers runs ragged at
~0.16 s/step against 0.115. Estimates use `ceil(cells/workers) * per_cell`, rounding up to whole
waves - dividing by workers understated a 23 h estimate into 42 h of reality once already.

Per `docs/playbooks/remote-experiment-runs.md`: dispatch over Tailscale via the `ssh laptop` alias
only, `powershell -NoProfile -ExecutionPolicy Bypass -File`, `python -u` so the log is not fully
buffered, `Tee-Object` to a log file, and **gate on artifacts, never on exit codes** - a dispatching
ssh dying does not kill the run.

## 6. What would refute the whole stage

- **Claim 1 negative and Claim 2 negative.** Neither a learned baseline nor a plasticity gate helps.
  Signal density is not the binding constraint at the frontier, and Wall 2 needs a different answer.
- **Claim 2 positive but Claim 3 flat.** The gate's benefit is its rate, not its signal. The
  neuromorphic claim fails and the finding is an efficiency one: the encoder can be updated half as
  often for free.
- **Arm B rescues dead seeds without moving the mean.** Claim 1 fails while Claim 5 moves. That
  would be a real result about basins rather than about averages, and it is the reason Claim 5 is
  pre-registered rather than noticed afterwards.

## 7. Deliberately not in this experiment

- **The EXP-033 decodability probe.** Retired as a policy predictor.
- **The entropy trace as a claim.** Section 0. Recorded, readable within an arm, in no claim.
- **Any depth-8 push.** EXP-046 prices it at ~194,000 episodes and EXP-051 shows the encoder route
  is already within 9% of break-even against buying budget at depth 7.
- **Potential-based reward shaping.** The kickoff raised it; the advantage-only form is the smaller
  change and the invariance argument is not needed unless the signal enters the reward.
- **A spiking critic region.** The cheap linear version first, on the EXP-025 precedent.
