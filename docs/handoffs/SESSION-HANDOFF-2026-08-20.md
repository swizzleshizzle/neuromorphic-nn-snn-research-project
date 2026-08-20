# Session Handoff - 2026-08-20 (Thu) - for Week 20 session 3

> **Nothing is in flight. Both machines are idle. The repo is clean and pushed at `189ca68`.**
>
> **EXP-046 is complete, both arms.** Read `experiments/046_depth6_budget/RESULTS.md`.
> The vault is current through it: `[[week-20-budget-not-depth]]` and `experiment-log.md`.

> [!important] SCHEDULING CONSTRAINT - session 3 runs tonight, and Michael is out tomorrow night
> **Aim to have something dispatched before this session ends.** Otherwise the laptop sits idle
> through tomorrow, and a 12-worker run is the one thing that does not need anyone present.
>
> **But do not rush the pre-registration to manufacture a dispatch.** Rules-on-disk-before-numbers
> is the one thing this project does not trade, and there is a good cheap fallback below
> specifically so that trade never has to be made.

## 0. The decision rule for tonight

1. Work the **primary job** (encoder fine-tuning, section 2). It needs a trainer change, a
   pre-registration and a smoke test.
2. **When the design is written and smoke-tested, dispatch it.** It is the most valuable run
   available and it will happily use tomorrow.
3. **If it is not smoke-tested and ready by the time the session is winding down, dispatch the
   FALLBACK instead** (section 3) and finish the encoder design without a clock on it. The
   fallback is ~5 h, already justified, and needs no new science.

**A half-designed experiment dispatched to fill a night is worth less than nothing** - it burns the
machine and produces a number nobody can interpret.


## 1. Where the project stands

**The depth series is a budget series, and the budget curve is now priced.**

| depth 6 budget | mean | sd | paired vs 10k | exact p |
|---|---|---|---|---|
| 10,000 (EXP-043) | 0.1800 | 0.0985 | - | - |
| **25,000** | **0.2729** | 0.0838 | **+0.0929** | 0.0010 |
| **44,000** | **0.3225** | 0.0373 | **+0.1425** | 0.0005 |

- **Log-linear, ~0.22 success per log10 of spend. No knee.** A line through the endpoints predicts
  0.2681 at 25k; the measured 0.2729 deviates by +0.0048 against an SE of 0.0242.
- **4.4x the budget buys about one depth.** d5@10k 0.3412 vs d6@44k 0.3225; d6@10k 0.1800 vs
  d7@44k 0.1971. Both inside noise.
- **Variance falls with budget**: sd 0.0985 -> 0.0838 -> 0.0373, regressions 1/12 -> 0/12.
- Every earlier depth number in this project means **"depth N at 10,000 episodes"**.

> [!important] Budget is a solved and unattractive lever
> It works, it is predictable, and it costs 4.4x per depth for a flat gain. Depth 8 at matched
> exposure is ~194,000 episodes, roughly 4 days, for ~0.18. **Chasing depth with compute is a
> losing exchange rate**, and a log-linear curve means there is no cheap fraction of it.

## 2. PRIMARY JOB: fine-tune the encoder during RL

**The only untried lever that could change the exchange rate rather than pay it.** On the list
since Week 18, and after this week's results it is the only thing left that is not "spend more".

### Three traps to get right BEFORE writing code

1. **The encoder is frozen by construction.** `make_agent` builds the brain and nothing unfreezes
   it; the 390-parameter head is the entire trainable surface. This is a **trainer change**, not a
   config flag. Expect to touch `policy_parameters` and the optimizer construction in
   `run_cube_baseline`, and to add a config field rather than repurposing one.
2. **"The same 390 trainable parameters" dies the moment the encoder trains.** That phrase is
   load-bearing in every RESULTS.md and in the whole visual story. A fine-tuned arm is a
   **different architecture** and must be reported as one - never folded into the depth series as
   though it were another cell.
3. **Pre-register the confound, because this week supplied it.** Fine-tuning adds trainable
   parameters AND compute per step, so a naive win is explainable by budget - the leading
   alternative after EXP-044/045/046. Decide in advance what the control holds fixed:
   - matching **episodes** gives the fine-tuned arm more compute;
   - matching **compute** means the frozen control gets more episodes, and EXP-046 says that alone
     is worth about **+0.22 per log10 of spend**, which could swamp the effect.
   **Whichever is chosen, the other must be reported.** A budget-matched frozen arm is the obvious
   control; say so in the spec and say what it cannot rule out.

### A starting sketch, not a decision

- **Depth 6 at 10,000 episodes** is the natural test bed: baseline **0.1800** (EXP-043), plenty of
  headroom below depth 6's 44k ceiling of 0.3225, and only ~5 h for 12 seeds.
- Paired against EXP-043's `exp043_capped_d6` cells - same seeds, same encoders at init.
- The interesting quantity may be the **probe**, not the score: EXP-039 measured what the encoder
  can represent, so re-probing a fine-tuned encoder says whether RL improved the representation or
  merely fitted the head to it. That is a mechanism measurement, which this project prefers.

## 3. FALLBACK if the design is not ready: depth 5 at 24 seeds

Settles EXP-043's Claim 1, which has been open since Aug 14: **+0.1108 at p 0.0815**, the clearest
case in the project where 12 seeds is the binding constraint rather than the effect.

> [!warning] It is cheap but NOT free - it needs 12 more encoders first
> Only **seeds 0-11** have pretrained encoders (`exp040_encoder_s*.pt`). A 24-seed depth-5 run
> needs EXP-039's pretraining for **seeds 12-23** first: ~900-1000 s per seed, so **~20 min** at
> 12 workers. Then the RL arm is 84,000 steps per run, about **4.6 h** for the 12 new seeds.
>
> **Total ~5 h**, and the existing 12 seeds are reused rather than re-run.

## 4. Also open, not for tonight

1. **Restate the depth series** with the budget caveat in the vault's `road-to-a-solved-cube` and
   `progress-tracker` - **both are still at EXP-037** (modified 2026-08-07), nine experiments
   behind. **Needs Michael's call**: they are planning documents, not logs.
2. The manim scenes do not include depth 7 or the budget finding. ~30 minutes if wanted; there is
   **no deadline** (see `CLAUDE.md`).

## 5. Operational notes earned this week

- **A dispatching ssh reporting `exit 1` does NOT mean the run failed.** Seen twice on EXP-046.
  `*>&1 | Tee-Object` - added so tracebacks stop being lost - puts torch's `UserWarning` in the
  output stream and the PowerShell wrapper exits non-zero on it. Check for `12/12` and `done.`
  before reacting. Same lesson the playbook records for 124 and 255.
- **`probe_run.ps1` takes `-LogName`.** It hardcoded `run.log`, which reports the wrong arm's log
  when an experiment has more than one.
- **`run.py --dry-run`** prints the pre-flight banner and starts nothing. Use it instead of
  running the driver and killing the pipe.
- Throughput held at ~6.4-7.1 effective cores at 12 workers throughout. Estimates ran long:
  EXP-044 arm B 25.4 h against 32 h estimated, EXP-046 15.5 h against 23 h, the midpoint 9.8 h
  against 13 h.

## 6. Pointers

- `experiments/046_depth6_budget/RESULTS.md` - the budget curve, both arms
- `experiments/045_budget_vs_coverage/RESULTS.md` - why it is budget and not coverage
- `experiments/044_depth7_frontier/RESULTS.md` - depth 7, and the corrected framing
- Vault: `[[week-20-budget-not-depth]]`, `[[week-19-the-depth1-trap]]`, `experiment-log.md`
- Remote runs: `docs/playbooks/remote-experiment-runs.md`
