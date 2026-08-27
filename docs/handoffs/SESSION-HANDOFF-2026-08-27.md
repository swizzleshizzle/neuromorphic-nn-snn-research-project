# Session Handoff - 2026-08-27 (Thu) - for Week 21 session 3

> **Nothing is running. The laptop is idle, the repo is clean and pushed at `54f2510`, the vault
> is current through EXP-052.**
>
> **There is one decision waiting and it is deliberately unmade** - see section 0.

## 0. The decision waiting for you

**Stage 3 is the week's main build and its direction was never chosen.** I put the fork to
Michael twice and did not pick, because the two options aim at different goals and both are
defensible. Do not start building until this is settled.

`NeuromodBus` (`src/neuromorphic/neuromod.py`) is a **46-line stub**. It holds `dopamine`, `ach`
and a `learning_enabled` property gated on a threshold. `Brain.learn()` writes
`reward - baseline` into `dopamine`. **Nothing in the codebase reads any of it** - plasticity was
deferred back in L11.

| | what it is | what it buys |
|---|---|---|
| **3a** | learned critic `V(s)`, TD error `delta = r + gamma*V(s') - V(s)` as the advantage instead of the EMA baseline. `Linear(64 -> 1)`, 65 params | Targets the **measured** failure: EXP-045's depth-7 collapse, entropy 0.591 -> 0.098 on a 2.2% solve rate. Safer, more certain. |
| **3b** | gate encoder plasticity on `learning_enabled` - the encoder updates only when `|delta|` clears threshold | Makes the bus **load-bearing**. Direct synthesis with the week-20 encoder work. |

> [!danger] The trap, and it is why this is not my call to make
> **3a is actor-critic. Routing `delta` through a bus object does not make it neuromorphic.** If
> nothing reads the bus, the routing is decoration and the honest description of 3a is "a learned
> critic reduces gradient variance" - a real result, and an RL one.
>
> `CLAUDE.md` and `road-to-a-solved-cube` both record that **nothing neuromorphic participates in
> the learning yet**. Week 20 changed that once, when the spiking encoder started training.
> **Do not let 3a be written up as if it changed it twice.**

Full sketch, including the four design questions to settle before writing a spec:
`docs/handoffs/WEEK-21-KICKOFF.md` section 2.

## 1. Where the project stands

**Depth 6 at 0.3525; depths 3 through 7 all work given budget.** The full 2x2 needs depth 11.

The three things that shape every decision now:

**1. The depth series was a BUDGET series.** Success is linear in the *logarithm* of spend,
~0.22 per log10 at depth 6 and 0.210 at depth 7, **with no knee**. Every depth number written
before EXP-046 means *"at 10,000 episodes"*, and "the break point" was never architectural.

**2. The encoder can be trained, and it is bounded.** Fine-tuning during RL is confirmed
(EXP-047), the gain is **in the encoder** not its head (EXP-048), it is **RL's objective** and not
merely more gradient (EXP-050), but it **does not compound** (EXP-049, a constant ~+0.05 per
round) and **its advantage erodes with depth** (EXP-051). As a compute ratio the encoder route is
**1.69x cheaper than buying budget at depth 6 and 1.09x at depth 7** - near break-even at the
frontier.

**3. THREE INSTRUMENTS NOW MOVE AGAINST POLICY QUALITY.** This is the most important standing
fact in the project and it is easy to forget.

| instrument | what it does | evidence |
|---|---|---|
| EXP-033 decodability probe | falls while policy rises | EXP-049: 0-12 seeds, p 0.0005 |
| the same probe | rises while policy halves | EXP-050: 12-0 seeds, p 0.0005 |
| pretraining move-accuracy | monotone up while policy is flat then collapses | EXP-052: 0.383 -> 0.452 against 0.2012 -> 0.0887 |

**Use `revisit_rate` and `optimality`.** They are in every record since EXP-029 and carried the
mechanism in EXP-048. **Do not put a probe number in a new spec**, and treat any pretext-task
metric as anti-informative until shown otherwise.

## 2. Week 21 so far

| | what | outcome |
|---|---|---|
| EXP-050 | objective vs more gradient | **The objective.** A control favoured on four axes lost by 0.22, and more pretraining *halves* the policy. |
| EXP-051 | depth-7 transfer | **Transfers** (+0.0850, p 0.0039) but its advantage over budget collapses to +0.0079. |
| EXP-052 | pretraining optimum | **Plateau 10-40, collapse by 80.** The inherited 40 was fine; 10 epochs buys the same at a quarter of the cost. |

All three have `RESULTS.md` files and vault rows. Week 20's closeout is done, including a
correction block at the top of `road-to-a-solved-cube` - **its Stage 2 success metric was the
probe, which is now retired.**

## 3. Open items, roughly by value

1. **Stage 3** - section 0. The main build, direction unchosen.
2. **The plateau's left edge is unmeasured.** Between 0 epochs (0.0000) and 10 (0.2012) the curve
   rises steeply and nobody has looked. If 3 epochs also gives ~0.20, pretraining is doing far
   less work than the EXP-039/040 story implies. That would be **substantive**, not an efficiency
   note. Cheap: one pretraining arm plus one RL arm, ~4.5 h.
3. **Sequence-blindness is unproven across four experiments.** The hypothesis - the inverse model
   is purely single-step, so over-training it makes the code sequence-blind - now has four
   consistent non-significant trends behind it (EXP-050, EXP-052). It needs **a metric that does
   not depend on solving**, e.g. whether the concept distinguishes a state one move away from one
   two moves away. Offline, cheap, and it would convert a recurring hand-wave into a result.
4. **Redo the probe-based inferences.** EXP-033, EXP-039 and EXP-047 each read a probe delta as
   evidence about policy. Their numbers stand; the inferences need trajectory metrics beside them.
5. **Re-ask the memory question.** EXP-030 was measured on a 2.2% policy against a working one.
6. **Cut pretraining to 10 epochs** for future encoder builds. Free 4x, changes no result.

**Do not run depth 8 on the current recipe expecting a bargain.** EXP-046 prices it at ~194,000
episodes, and EXP-051 shows the encoder route is already near break-even at depth 7.

## 4. Operational notes earned this week

- **Estimate pretraining separately from RL.** Pretraining is far more memory-bandwidth-bound:
  measured **2.86 effective cores from 10 workers** against RL's ~7.4. Playbook updated.
- **Round up to whole waves.** `ceil(cells/workers) * per_cell` - dividing by workers understates
  badly. 12 cells on 10 workers is **two** waves. This alone made a 23 h estimate take 42 h.
- **Prefer a worker count that divides the cell count.** 12 cells on 6 workers beats 10 workers:
  two clean waves at 0.115 s/step against ragged waves at ~0.16.
- **A dispatching ssh dying does not kill the run.** Confirmed repeatedly this week, including
  deliberate kills. Gate on artifacts, never on exit codes. Read `phase*.log`, not the transcript.
- **The laptop slept once mid-run** (EXP-048) and the chain stalled. AC standby is now disabled,
  but Tailscale cannot wake a sleeping machine and there is no remote recovery path.
- **`scp 'laptop:.../outputs/*' dest/` fails if the LOCAL directory does not exist.** The error
  says `Is a directory` and is misleading. `mkdir -p` the destination first.

## 5. Two process failures worth not repeating

Both happened this week with **every pre-registered threshold obeyed**, which is the point.

**A pre-registered inference can be satisfied and still be wrong.** EXP-050's Claim 4 predicted
*"E0+ probes higher while arm F's gain is smaller"* and concluded that would prove the
anti-correlation was RL-specific. Both conditions held. The conclusion was wrong, because "gain is
smaller" was written without considering that arm F might post a **loss** - and with a loss the
inference inverts. Thresholds do not protect the reasoning built on them.

**An interpretive layer can over-read data the thresholds handled correctly.** EXP-052's
aggregator declared *"monotone decreasing, peak below 10, substantially weakens the EXP-039/040
premise"* from the ordering of four means, three of which were indistinguishable at p 0.49-0.84.
Ranking noise, reported as a finding. It now requires significance before naming a shape.

## 6. Pointers

- `docs/handoffs/WEEK-21-KICKOFF.md` - the Stage 3 sketch and its design questions
- `experiments/052_pretraining_optimum/RESULTS.md` - the newest result
- `experiments/051_depth7_transfer/RESULTS.md` - why the encoder line is bounded
- `experiments/050_objective_vs_gradient/RESULTS.md` - the objective, and the probe reversal
- `experiments/046_depth6_budget/RESULTS.md` - the budget curve that prices every confound
- `docs/playbooks/remote-experiment-runs.md` - dispatch, and the corrected estimator
- Vault: `[[week-20-budget-not-depth]]` (carries weeks 20 and 21), `experiment-log.md`
