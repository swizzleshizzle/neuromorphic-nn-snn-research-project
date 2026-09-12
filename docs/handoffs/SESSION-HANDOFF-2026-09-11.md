# Session Handoff - 2026-09-11 (Week 23) - EXP-059 COMPLETE AND MERGED

> **EXP-060 IS RUNNING.** Phase 0 of 3 launched **2026-09-11 21:33 UTC**, ~17 h total.
> `main` is clean, no branches, no PRs.
>
> **WINDOWS UPDATE: I COULD NOT VERIFY THE PAUSE.** Michael paused it, but no `Pause*` values
> exist under `HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings` where the Settings UI
> normally writes `PauseUpdatesExpiryTime`. **`ActiveHours` are 09:00-03:00 local**, so Windows may
> only auto-restart between **03:00 and 09:00 local = 07:00-13:00 UTC**. **EXP-059's killer reboot
> was at 03:35 local, 35 minutes into that window.** If it happens again, that is when.
>
> **Suite: 591 passed** under `-m "not slow"` (356 outside `tests/training`, 31 in
> `test_cube_baseline.py`, 204 in the rest), plus 22 slow-marked deselected.
>
> **EXP-059 answered the memory question, and the answer is that memory HURTS.**

## 1. EXP-059 - the headline, and why it is stronger than a null

**`M` minus `A` on held-out success: -0.0954 at p 0.0056**, clearing the pre-registered 0.05 bar
downward and clearing Bonferroni 0.0167. W-L-T 6-17-1 across 24 seeds.

| arm | readout | success |
|---|---|---|
| **A**, amnesic | `memory_amnesic` | **0.3138** |
| **M**, memory | `memory` | 0.2183 |
| **S**, shuffled | `memory_shuffled` | 0.1979 |

**That reading was fixed in the spec before any number existed** - a significant negative is a real
finding required in the headline, not a failed confirmation - specifically so it could not be
softened afterwards.

### The sharpest result is Claim 4, and it is easy to misread

**`M` minus `S` is +0.0204 at p 0.4268: correct memory is INDISTINGUISHABLE from wrong memory.**
Reading the attractor at all costs 0.095; the content being correct buys nothing measurable.

**Arm A is not "no recall".** `MemoryReadout`'s docstring records that 65% of the recall block is a
memory-free transform of the current concept, so A is *recall features without stored content*.
**Adding stored content is precisely what costs.**

### A two-arm design would have published a win for the THIRD time

Memory versus shuffle-null measures **+0.0204** here and reads as a modest win. Only the amnesic arm
shows both memory arms losing to it by about 0.10. EXP-030 and EXP-058 had the identical shape.
**Three arms is not thoroughness; it is the minimum that answers the question.**

### The mechanism claim did NOT confirm, and its sign runs the wrong way

`revisit_rate` moved **+0.0157 at p 0.4450** where confirmation needed `<= -0.02`. Pre-registered
joint reading: performance moved while the cycling mechanism did not. **Whatever memory did here, it
was not the anti-cycling story, and must not be narrated as one.**

### Supporting checks that make the result hard to explain away

- **Not a held-out artefact**: `train_success_rate` shows the same ordering (A 0.2823, M 0.1925,
  S 0.1696) and the generalisation gap is within 0.006 across arms.
- **Arm A reproduces the known config**: 0.3138 against `exp043_capped_d5`'s 0.3229.
- **`optimality` tracks success** (A 0.6707, M 0.6477, S 0.5745) while `revisit_rate` is flat.

## 2. The validity gate PASSED, and the gate-calibration rule paid for itself

| condition | measured | threshold | |
|---|---|---|---|
| arm M `recall_content_cos` | **0.8514** (range 0.7527-0.9042) | < 0.95 | PASS |
| arm S `unshuffled_frac` | **0.1480** | < 0.20 | PASS |

**Fourth gate this project has used, and the first right by construction.** Calibrated across its
full attainable range before the spec existed, it can fail (empty attractor = 1.000000) and can pass
(real run = 0.8128), and it is a **cosine, so bounded and scale-free** - which is what stops it
repeating EXP-057's regime-dependence.

**It was also checked EARLY**, on arm M alone, while the run was still going: a failure would have
meant 13 h of remaining arm-S compute spent on a void experiment. That is not peeking - Claim 3 is a
condition, not a contrast, arm A was never loaded, and the records went to a scratchpad so the
aggregator could not compute anything on partial data.

> [!note] One cross-check in the aggregator is structurally INERT, and it is better to know.
> It prints arm A's `recall_content_cos` expecting ~1.0. It is `None` for all 24 seeds: the
> `memory_amnesic` branch returns at `cube_baseline.py:402`, before `_probe_recall` at line 418.
> Correct behaviour, but the check can never fire.

## 3. What else this week produced

- **`docs/retired-instruments.md`** - the standing note, with a `CLAUDE.md` pointer. **Every one of
  the five works as a THRESHOLD and fails as a GRADIENT.**
- **`experiments/059_memory_depth5/aggregate.py` + 22 tests**, written while zero records existed,
  each test verified to fail against the bug it names by mutating the aggregator eleven ways.
- **EXP-060 spec** - `docs/superpowers/specs/2026-09-10-exp060-flattened-critic-replication-design.md`.
- **Playbook**: an unreachable tailscale peer is an **unknown**, not a paused job; the laptop is
  **UTC-4**; do not scale per-cell cost by hand, use `steps()`; check reboot history before a long
  dispatch.

## 4. EXP-060 - dispatched, and the spec's cost was WRONG

**Three phases, in order, ~17 h**, not the ~14 h the spec carried. The launcher enforces the order.

| phase | what | cells | cost | produces |
|---|---|---|---|---|
| **0 `baseline`** | EXP-043 depth 6, seeds 14-23 | 10 | **4.3 h MEASURED** (est. 3 h) | 10/10 **DONE** |
| **1 `finetune`** | EXP-047 `--mode confirm`, seeds 14-23 | 10 | **7.1 h MEASURED** (est. 6 h) | 10/10 **DONE** |
| **2 `rl`** | EXP-060 arms B and F | 20 | est. 8-11 h, running since **08:59 UTC** | `exp060_*.json` |

**Both completed phases overran their estimates, in the same direction and for the same reason** -
4.3 h against 3 h and 7.1 h against 6 h. Phase 2's ~8 h came from the same 6-worker-scaling method,
so treat it as a floor and expect **~17:00-20:00 UTC**. Wave 1 (10 of the 20 cells) settles it.

> [!note] **A correction worth keeping: phase 1 is NOT all-or-nothing.** An earlier note here said
> it was a single wave of 10 whose encoders appear together, so a reboot would cost the whole ~6 h.
> **Wrong.** Encoders are written per cell and the seeds vary enough that they finish separately
> (3 of 10 at 4.8 h). Completed artifacts survive and `--skip-existing` keeps them, so a reboot
> costs only the in-flight cells - in every phase.

**Phase 2's dispatch was verified from the launcher's own banner**, not an exit code:
`E1 encoders for seeds 14-23 = 10 of 10`, 20 runs, seeds 14-23 confirmed FRESH, arms B and F.

**Phase 0 measured 4.3 h against a 3 h estimate**, so the total is **~18.3 h** and the ETA is
**2026-09-12 ~16:00 UTC**. The miss is the same shape as EXP-059's: the estimate was scaled from a
6-worker measurement using the playbook's 10-vs-6 penalty ratio, and **the real penalty at 10
workers is steeper than that ratio implies.** Phase 0's gates all passed and the dispatch was
verified from the launcher's own banner (`EXP-043 d6 for seeds 14-23 = 10 of 10`), not from an exit
code.

> [!warning] **BOTH REMAINING PHASES CROSS THE REBOOT WINDOW.** Windows may only auto-restart
> 03:00-09:00 local = **07:00-13:00 UTC**. Phase 1 runs into ~08:00 UTC and phase 2 spans
> ~08:00-16:00, so each crosses it. All phases pass `--skip-existing`, so a reboot costs the
> in-flight wave rather than the phase - but phase 1 is a SINGLE wave of 10, so a reboot there
> costs the whole ~6 h.

> [!warning] **THE MISSING DEPENDENCY, found at dispatch and not at spec time.**
> E1 encoders cannot be made for seeds 14-23 without an **EXP-043 depth-6 baseline for those
> seeds first**: EXP-047's `confirm` mode refuses without it, because those records are the paired
> baseline for **its own** Claim 1. **They exist only for seeds 0-11.** Depth **5** has all 24,
> which is exactly why the gap was easy to miss - EXP-059 ran on depth-5 encoders and found
> everything it needed. The spec's cost section is **amended, before any EXP-060 number exists**,
> which is the only time an amendment is legitimate.

**Phase 0's estimate is the weakest number here.** A depth-6 capped cell with a `concept` readout
has never been timed at 10 workers; it is scaled from EXP-058's 3.37 h depth-6 *memory* cell, which
does strictly more work. **Read wave 1 before trusting the total** - EXP-059 was dispatched without
that discipline and its 2.5 h/cell turned out to be 3.05.

**The design point, which must survive into the write-up:** the **primary is seeds 14-23 ALONE at
n=10**, not the pooled n=22. Extending because a p-value was marginal and then pooling is **optional
stopping**. The primary is only ~50-60% powered, and EXP-056's -0.0646 is **upward-biased because it
was selected for significance**, so a null primary is a **bound, never a refutation**.

Monitoring advances the phases automatically; the launcher's gates make a wrong phase fail safe.

## 5. Other open items

1. **The live scientific thread: WHY does the memory read hurt?** EXP-059's gate ruled out an
   *empty* attractor, far narrower than ruling out a badly scaled or uninformative recall code.
   **More interesting than EXP-055's leads.**
2. **EXP-055's two leads**, both needing compute.
3. **Vault `4f13` IS EXP-059 and can be ticked** - Michael's call.
4. **Vault, needs Michael**: `0576` dashboard render, `0817` Phase 0/1 checkpoints.

## 6. Standing facts

- **READ `docs/retired-instruments.md` BEFORE PUTTING AN INSTRUMENT IN A SPEC.**
- **READ "THE GATE-CALIBRATION RULE" IN `CLAUDE.md` BEFORE WRITING A GATE.** Calibrate across the
  attainable range; prefer a ratio or a bounded quantity to an absolute.
- **The critic question is CLOSED**: the benefit is within-episode state-dependence, not calibration.
- **The memory question is now ANSWERED at depth 5**: the recall read is harmful, and its content's
  correctness is worth nothing measurable.
- **An unreachable peer is an UNKNOWN.** Nothing is durable until a cell completes.
- **n >= 12 seeds. Measure the chance floor. No scipy.**

## 7. Pointers

- `experiments/059_memory_depth5/RESULTS.md` - the full write-up
- `docs/retired-instruments.md`, `CLAUDE.md`
- `docs/playbooks/remote-experiment-runs.md` - the corrected last-seen table, cost estimation
- Vault: `experiment-log.md` (through EXP-059, plus the gate-calibration addendum)
