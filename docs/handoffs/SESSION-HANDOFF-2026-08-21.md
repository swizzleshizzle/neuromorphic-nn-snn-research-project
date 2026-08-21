# Session Handoff - 2026-08-21 (Fri) - for Week 20 session 4

> **A ~23 h CHAIN IS IN FLIGHT ON THE LAPTOP.** Dispatched 2026-08-20 18:30 laptop time from
> commit `9d2c757`. Do not dispatch anything else to `SwizzlesDuo` until it finishes.
>
> **Nothing needs a human while it runs.** The learning-rate choice inside it is mechanical and
> pre-registered; the chain selects, or halts itself, without anyone present.

## 0. First thing: find out where the chain is

```bash
ssh -n laptop 'powershell -NoProfile -Command "cd C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\experiments\047_encoder_finetuning\outputs; Get-ChildItem phase*.log | ForEach-Object { $_.Name + \"  \" + $_.LastWriteTime }"'
```

Then read the newest one. Phases in order, with what each writes:

| phase | what | ~cost | artifact that proves it finished |
|---|---|---|---|
| 0 | EXP-040 encoders, seeds 12-23 | **1.6 h** | `exp040_encoder_s{12..23}.pt` |
| 1 | EXP-047 pilot, 3 rates x seeds 12,13 | 5.6 h | 6 records matching `_s1[23]_sig` |
| 1b | probe the pilot encoders | 15 min | `probe_pilot.json` |
| 2 | **`select_lr.py`** | 1 min | `selected_lr.json` |
| 3 | EXP-047 confirmatory, seeds 0-11 | 6.7 h | 12 records at the selected rate |
| 3b | probe the confirmatory encoders | 30 min | `probe_confirm.json` |
| 4 | fallback: depth 5, seeds 12-23, BOTH arms | 9.4 h | EXP-040 and EXP-043 d5 records |

> [!warning] A dispatching ssh reporting a non-zero exit does NOT mean the run failed.
> Seen twice on EXP-046. Check for the artifacts above and for worker count before reacting.
> **Worker processes are `python3.13.exe`, not `python.exe`** - match on `^python`.
>
> **Confirmed again on this dispatch, and this time deliberately.** The dispatching ssh client
> was killed outright about 10 minutes in. The chain kept running: 14 python processes and 2
> powershell processes were still up afterwards. Windows has no SIGHUP semantics, and the
> launcher is started with `ssh -n` precisely so it survives the pipe closing.
>
> The only casualty is the launcher's own `Say` progress lines, which went to the dead stdout.
> **Every phase's real output goes to `phase*.log` files, not the pipe**, so nothing that matters
> was lost. Read the logs, never the dispatch transcript.

**Estimates in this project run short** (EXP-046: 15.5 h against 23 h). ~16 h is the likelier
total against the ~23 h nominal.

> [!warning] Phase 0 is **1.6 h**, not the 20 min the 2026-08-20 handoff claimed.
> That figure came from dividing EXP-039's ~900-1000 s per seed by 12 workers, which assumes
> perfect 12-way parallelism. The laptop delivers about **6.6 effective cores** at 12 workers.
> **EXP-040's own `RESULTS.md` records the real cost**: "Phase 1 pretrains 12 encoders (~1.6 h at
> 10 workers)". Verified live on this dispatch - 81 minutes in, phase 0 had written nothing and
> was consuming 6.56 effective cores, which is on track rather than stalled.
>
> **Encoders appear all at once near the end**, because all workers start together and each
> writes only when its seed finishes. An empty directory mid-phase is expected.

> [!important] MEASURED ON THIS DISPATCH: **10 workers beat 12**, so phase 0 was dispatched wrong.
> The launcher ran phase 0 at `--workers 12`. That took **>130 min** for the same 12 encoders
> EXP-040 built in **100 min at 10 workers** (its RESULTS.md provenance: 2026-08-09 14:40-16:20,
> same laptop, same operation). All 12 workers sat at ~0.5 CPU each, about **6.0 effective cores**,
> against the **~8.4** EXP-040 got from 10.
>
> Twelve loses **even though it fits every seed in one scheduling wave** where 10 needs two. The
> machine is memory-bound, not core-bound. This extends the playbook's "10 beat 16" down to
> **"10 beat 12"**, and `launch047.ps1` has been corrected to 10 for any future run. The cost on
> this dispatch was about 30 minutes and is already spent; the other phases were always 6 and 10.

## 1. What was done on 2026-08-20

**EXP-047 designed, pre-registered, implemented, smoke-tested and dispatched.** Spec at
`docs/superpowers/specs/2026-08-20-exp047-encoder-finetuning-design.md`, committed at `69bf1dc`
**before** any code existed that could produce a number.

The encoder was frozen **by construction**: `reinforce.action_distribution` hardcoded a
`no_grad` around `brain.step`. `CubeConfig.encoder_lr` now makes that conditional and gives Adam
a second parameter group over `brain.sensory`. Unset, it is a strict no-op.

### The three traps the old handoff named, and what was done about each

1. **"It is a trainer change, not a config flag."** Correct. Done as one, plus the
   `MemoryReadout` fix below.
2. **"The same 390 trainable parameters dies the moment the encoder trains."** The record now
   carries `trainable_params` (390 frozen, **27,206** fine-tuned, a factor of 70), and
   `aggregate.py` prints Claim 3 **first**, before any score, so the write-up cannot forget which
   architecture produced the number.
3. **"Pre-register the confound."** Done, and **it did not need its own arm.** Fine-tuning was
   measured at **1.33x per step** (56.17 -> 74.87 ms), and EXP-046's log-linear curve prices
   1.33x of budget at **+0.027**. So episodes are matched, and the standing +0.05 bar sits 1.9x
   above what compute alone buys. A delta of +0.027 to +0.05 is pre-committed as **AMBIGUOUS**.

### The defect that matters most, because it will recur

**The first implementation trained nothing and looked completely normal.** `fc1.weight` moved by
exactly `0.0`, the run produced an ordinary success rate, and every neutrality assertion passed.

`run_cube_baseline` passes `feature_fn=readout` on the **training** call for every readout
including `"concept"`; only the two evaluation calls pass `None`. `MemoryReadout.__call__`
wrapped its whole body in `torch.no_grad()`, so that identity branch **detached the concept** -
and it sits on the policy path of every cube run ever recorded.

Caught only by a complement test asserting the encoder MUST move (`drift > 1e-3`). Now recorded
in `CLAUDE.md`: **verify the gradient arrives at the parameter, never that the switch is set.** A
frozen-vs-trainable comparison where both arms are secretly frozen is indistinguishable from a
genuine null.

## 2. How the learning rate gets chosen, and why you must not touch it

`encoder_lr` had no prior. It is selected by a pilot, under a rule fixed in spec section 5.2
before any pilot number existed.

- **Grid** 1e-3, 1e-4, 1e-5. **Pilot seeds 12 and 13, deliberately NOT 0-11.**
- **Gate**: depth-4 probe top-1 may not fall more than **0.02** below that seed's own starting
  encoder, on **both** seeds. That is under 6% of the +0.3396 pretraining bought (EXP-039).
- **Choice**: highest mean depth-4 probe top-1 among rates that pass. Tie-break to the larger.
- **Halt**: if none passes, the chain **skips phases 3/3b and goes straight to the fallback**.

> [!important] Two stacked defences, because either alone is leaky.
> EXP-039 section 6a refused to pick its lr by the probe because the probe was its outcome
> metric, and recorded that the probe **would have picked the other rate**. EXP-047 inherits that
> trap one level up, so: `select_lr.py` reads **only** `probe_pilot.json` and cannot see a
> success rate; and the pilot seeds are **disjoint** from the seeds carrying either claim.
> Nothing that decides a claim was used to make the choice.
>
> **Do not pass `--encoder-lr` by hand.** The driver refuses without `selected_lr.json` on
> purpose. Choosing the rate yourself is precisely what the design forbids.

**If phase 2 halted, that is a RESULT, not a failure.** It means REINFORCE's gradient damages the
pretrained representation at every rate in the grid. Write it up. Widening the grid needs a new
pre-registration.

## 3. When it finishes

```bash
.venv/bin/python experiments/047_encoder_finetuning/aggregate.py
```

`aggregate.py` was written **before dispatch**, with the thresholds from the spec. Claims:

1. **PRIMARY, PAIRED.** Held-out success at depth 6 vs EXP-043's `exp043_capped_d6`, same seeds,
   exact permutation over 2**12. CONFIRMED at >= +0.05, p <= 0.05. Baseline **0.1800**.
2. **MECHANISM.** Re-probe. Carries a **pre-registered asymmetry**: RL trains on the RL split and
   the probe holds out a different one, so **degradation is clean, improvement is confounded**
   with memorising probed states. The **leak-free slice** is depth 6's 200 RL held-out states,
   which EXP-040 also excluded from pretraining, so neither stage ever saw them. Where the two
   disagree, **report the weaker**.
3. **ARCHITECTURE ACCOUNTING**, descriptive. Never a cell of the depth series.
4. **COLLAPSE**, descriptive, no p-value, against EXP-045's signature.
5. **THE NULL IS PRE-COMMITTED.** A refuted Claim 1 means REINFORCE cannot improve the encoder
   faster than budget can be bought; the frozen 390-parameter result stands **with no caveat
   added**, and the next move is a **different pretraining objective** (value or heuristic rather
   than inverse dynamics) - not more RL, not more episodes.

Then write `experiments/047_encoder_finetuning/RESULTS.md` with provenance, and mark each claim
confirmed or refuted.

## 4. The fallback, and a correction to the previous handoff

Phase 4 settles **EXP-043's Claim 1** at depth 5 (+0.1108 at p 0.0815, open since Aug 14) by
taking it to 24 seeds.

> **The 2026-08-20 handoff budgeted this at ~4.6 h for "the RL arm" on 12 new seeds. That is
> half of it.** Claim 1 is a **paired** delta against EXP-040's depth-5 cell, and EXP-040 phase 2
> only ever ran seeds 0-11 - so seeds 12-23 have no baseline to pair against unless it is also
> run. It is **24 RL runs, ~9.4 h**, and the chain does both arms.

Aggregating it needs pooling EXP-043's existing 12 seeds with the new 12. That aggregator does
**not** exist yet; it is the one piece of this work left to write.

## 5. Also open, not urgent

1. **Restate the depth series** with the budget caveat in the vault's `road-to-a-solved-cube` and
   `progress-tracker` - **both still at EXP-037**, ten experiments behind. **Needs Michael's
   call**: they are planning documents, not logs.
2. The manim scenes do not include depth 7, the budget finding, or EXP-047. **No deadline** - see
   `CLAUDE.md`, Content Day is defunct.
3. The vault is **not yet updated for EXP-047**. `[[week-20-budget-not-depth]]` and
   `experiment-log.md` stop at EXP-046.

## 6. Operational notes

- **Cost estimates: use the playbook's 153 ms/step, not `CLAUDE.md`'s 90 ms.** The 90 ms figure
  is single-step latency and understates a parallel run by ~70%. `wall_hours = steps * 0.153 /
  3600 / workers`, with the 1.33x applied to **training** steps only - evaluation uses
  `greedy_action`, which takes no `grad_brain` and stays at frozen cost.
- **Ten workers, not sixteen.** Memory-bound at ~920 MB private each. Fine-tuning adds only
  **+9 MB** per worker (measured), so this is unchanged.
- The laptop must **stay awake** for the whole chain. Tailscale cannot wake it, and a mid-flight
  sleep stalls the run.
- `run.py --dry-run` prints the pre-flight banner and starts nothing.
- Pre-flight validation table is in spec section 8.2. Notably, **Claim 2's probe was falsified
  before use**: a deliberately damaged encoder moved depth-4 top-1 by **-0.2015**, about 10x the
  0.02 gate, so the gate can fire.

## 7. Pointers

- `docs/superpowers/specs/2026-08-20-exp047-encoder-finetuning-design.md` - the contract
- `experiments/047_encoder_finetuning/` - driver, probe, selector, aggregator, launcher
- `experiments/046_depth6_budget/RESULTS.md` - the budget curve that prices the confound
- `experiments/043_cap_at_depth_5_6/RESULTS.md` - the paired baseline, and the open Claim 1
- `docs/playbooks/remote-experiment-runs.md` - dispatch procedure
