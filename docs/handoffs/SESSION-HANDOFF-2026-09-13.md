# Session Handoff - 2026-09-13 (Sun, end of Week 23) - THREE EXPERIMENTS LANDED

> **Nothing is running. The laptop is FREE. `main` is at `c52fe18`, clean, no branches, no PRs.**
> **Suite: 601 passed** under `-m "not slow"` (356 + 31 + 214), 22 slow deselected, **623 total**.
>
> **Week 23 delivered EXP-059, EXP-060 and EXP-061.** The memory line is now closed with a
> mechanism, and the critic line has an independent replication.

## 1. The week's findings, in the order they build

### EXP-059 - episodic memory HURTS a working policy

`M` minus `A` = **-0.0954 at p 0.0056**. Arms: amnesic **0.3138**, memory 0.2183, shuffled 0.1979.
`M` minus `S` = +0.0204, p 0.4268: **correct memory is indistinguishable from wrong memory.**

### EXP-061 - and the reason is that the recall is NOISE

EXP-059 left two explanations standing. **Matched-magnitude noise separated them.**

| arm | success |
|---|---|
| **A**, amnesic (memory-free transform of the CURRENT concept) | **0.3138** |
| M, memory | 0.2183 |
| S, shuffled | 0.1979 |
| **N, matched noise** | **0.1973** |

**`M` minus `N` = +0.0210 at p 0.4989** (indistinguishable) and **`N` minus `A` = -0.1165 at
p 0.0001** (clears Bonferroni). **The three uninformative arms sit within 0.021 of each other while
the memory-free transform sits 0.10-0.12 above all three.**

**So: the recall block is valuable for what it says about the CURRENT state, and stored content
destroys that as thoroughly as random noise does.**

> [!warning] **H1 IS SUPPORTED, NOT CONFIRMED.** The primary is a null, and a null is a bound. The
> spec pre-registered that reading before dispatch precisely so it could not be upgraded
> afterwards. What the data rules AGAINST is H2, which predicted noise would *beat* real memory.
> **`N` minus `S` = -0.0006 is descriptive with no p-value** - unregistered, and a post-hoc p would
> inflate a multiplicity the spec fixed at two contrasts.

### EXP-060 - EXP-056 replicates, and the effect grew

`F` minus `B` = **-0.0925 at p 0.0156** on fresh seeds 14-23, larger than the original -0.0646.
**The critic conclusion no longer rests on one knife-edge contrast.**

**Do not quote its pooled n=22 (-0.0773, p 0.0005) without BOTH caveats**: optional stopping, and
it pools across a level shift of 0.05-0.08 in both arms between seed blocks.

## 2. Method lessons, which may outlast the findings

1. **An instrument that cannot detect the defect it exists for is the gate-that-cannot-fail trap in
   disguise.** EXP-061's leak detector compared the real recall against the *noise vector*, both
   pre-substitution, so it read ~0 by construction. A 50% leak passed every assertion. **Only
   mutation testing found it** - six of seven mutations caught, and the survivor was the lesson.
   Now in `CLAUDE.md`.
2. **Reuse of another experiment's arms must be PROVEN inert, not assumed.** EXP-061 reused
   EXP-059's A, M and S. The added instruments were run before and after and verified identical to
   full float repr, with a regression test locking it in.
3. **Cost estimates: the same-worker-count one held exactly; four scaled ones came in low.** Price
   from a measurement at the worker count you will actually use. The playbook now says to treat a
   cross-worker-count figure as a floor and add 20-40%.
4. **Confirm completion from COUNTS, not the log.** EXP-060's aggregator was written after ~12 cell
   values had been seen while verifying the run had finished. EXP-061 used record/checkpoint counts
   plus zero processes instead, and its aggregator was written with **nothing** seen.
5. **An unreachable peer is an UNKNOWN.** EXP-061 slept ~5 h mid-run and resumed with nothing lost;
   from outside that was indistinguishable from the reboot that destroyed EXP-059's first attempt.
   The three-reading test resolved it the moment the machine answered.

## 3. Open items

1. **The next question is the READOUT, not the hippocampus.** EXP-059's gate showed stored content
   genuinely changes the recall (`recall_content_cos` 0.8514, well below 1.0), so **the content is
   there and this readout cannot use it.** `MemoryReadout` concatenates a raw hippocampal read; a
   readout that *learned what to attend to* is untested and is the obvious follow-up.
2. **The EXP-060 level shift**, unexplained: both arms ~0.06 lower on seeds 14-23 under the same
   recipe and `selected_lr.json`. **Laptop-free to probe** - compare the new E1 encoders against the
   old on any frozen metric.
3. **EXP-055's two leads**, both needing compute, both lower value than item 1.
4. **Vault**: `4f13` IS EXP-059 and `eef1` (dispatch EXP-060) is done - **both Michael's call to
   tick**. `0576` dashboard render and `0817` Phase 0/1 checkpoints still need him.
5. **Windows Update on the laptop is STILL not verifiably paused.** No `Pause*` values under
   `HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings`. `ActiveHours` 09:00-03:00 local means
   auto-restart is possible only **07:00-13:00 UTC**, and EXP-059's killer reboot was 03:35 local,
   inside it. **Check before the next long dispatch.**

## 4. Standing facts

- **READ `docs/retired-instruments.md` BEFORE PUTTING AN INSTRUMENT IN A SPEC.** Five retired; all
  work as a THRESHOLD and fail as a GRADIENT.
- **READ "THE GATE-CALIBRATION RULE" IN `CLAUDE.md`.** Six gates used; two were wrong, and the four
  since were right because they were calibrated across the attainable range and expressed as
  bounded or scale-free quantities.
- **The critic question is CLOSED and replicated**: within-episode state-dependence.
- **The memory question is CLOSED at depth 5 WITH a mechanism**: the recall is noise, and its
  content's correctness is worth nothing measurable.
- **Nothing is durable until a cell completes.** Records are written per cell.
- **n >= 12 seeds. Measure the chance floor. No scipy.**

## 5. Pointers

- `experiments/059_memory_depth5/RESULTS.md`, `experiments/060_flattened_critic_replication/RESULTS.md`,
  `experiments/061_noise_matched_recall/RESULTS.md`
- `docs/retired-instruments.md`, `CLAUDE.md`
- `docs/playbooks/remote-experiment-runs.md`
- Vault: `experiment-log.md`, through EXP-061 plus three gate-calibration addenda
