# Session Handoff - 2026-08-20 (Thu) - Week 20 session 2 - budget is characterised and priced

> **Nothing is in flight. Both machines are idle. The repo is clean and pushed.**
>
> **EXP-046 is complete, both arms.** Read `experiments/046_depth6_budget/RESULTS.md`.
> The vault is current through it: `[[week-20-budget-not-depth]]` and the experiment log.

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

## 2. THE NEXT EXPERIMENT: fine-tune the encoder during RL

**It is the only untried lever that could change the exchange rate rather than pay it**, and it
has been on the list since Week 18.

Unlike the last four experiments this one **needs real design work**, so start it fresh rather
than at the end of a session:

- The encoder is **frozen by construction** - `make_agent` builds the brain and nothing unfreezes
  it, and the 390-parameter head is the entire trainable surface. This is a trainer change, not a
  config flag.
- **The 390-parameter claim dies the moment the encoder trains.** Every write-up and the whole
  visual story lean on "same 390 trainable parameters". A fine-tuning arm is a different
  architecture and must be reported as one, not folded into the series.
- **Pre-register the confound**: a fine-tuned arm has more trainable parameters AND more compute
  per step. Decide in advance what the control holds fixed - most likely a budget-matched frozen
  arm, since EXP-044/046 make budget the obvious alternative explanation.

Also open, cheaper:

1. **Re-run depth 5 with 24+ seeds** to settle EXP-043's Claim 1 (+0.1108 at p 0.0815).
2. **Restate the depth series** with the budget caveat in the vault's `road-to-a-solved-cube` and
   `progress-tracker` - **both are still at EXP-037** (modified 2026-08-07), nine experiments
   behind. Needs Michael's call: they are planning documents, not logs.
3. The manim scenes do not include depth 7 or the budget finding. ~30 minutes if wanted; there is
   **no deadline** (see `CLAUDE.md`).

## 3. Operational notes earned this week

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

## 4. Pointers

- `experiments/046_depth6_budget/RESULTS.md` - the budget curve, both arms
- `experiments/045_budget_vs_coverage/RESULTS.md` - why it is budget and not coverage
- `experiments/044_depth7_frontier/RESULTS.md` - depth 7, and the corrected framing
- Vault: `[[week-20-budget-not-depth]]`, `[[week-19-the-depth1-trap]]`, `experiment-log.md`
- Remote runs: `docs/playbooks/remote-experiment-runs.md`
