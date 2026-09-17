# Session Handoff - 2026-09-17 (Week 24) - EXP-062 DONE, Phase 3 checkpoint is next

> **Nothing is running. The laptop is FREE and idle. `main` is at `5f79a62`, clean, no branches,
> no PRs.** Suite: **601 passed** under `-m "not slow"` (356 + 31 + 214), 22 slow, **623 total**.
>
> **Week 24's experimental work is COMPLETE.** Week 25 (Sep 21-27) is the **Phase 3 checkpoint**,
> and it is the largest remaining piece of work.

## 0. The one thing that needs Michael, and it has now cost two runs

> [!danger] **WINDOWS UPDATE IS STILL NOT PAUSED, AND THE RESTART FIRES AT ~03:31 LAPTOP-LOCAL.**
> `PAUSE_VALUES` under `HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings` reads **NONE**, checked
> again 2026-09-16 20:50 local. Michael paused updates on 2026-09-11; **two runs have been killed
> since.**
>
> | run | reboot, laptop-local | cost |
> |---|---|---|
> | EXP-059 attempt 1 | **03:35** | **~19 CPU-h — ALL of it**, no cell had completed |
> | EXP-062 attempt 1 | **03:31** | ~4 CPU-h, depth 8 had completed |
>
> **`ActiveHours` are 09:00-03:00, so Windows may only restart from 03:00 local and takes the first
> opportunity about half an hour in.** That is a narrow, predictable point, not a vague window.
>
> **Before any dispatch that will still be running at 03:31 local (07:31 UTC): confirm Settings
> shows "Updates paused until <date>".** Absent that, either schedule to clear 03:31 or accept
> losing the in-flight wave.
>
> **Per-cell durability is what sets the bill** — same failure, 19 CPU-h vs 4, because one run had
> banked completed cells and the other had not.

## 1. EXP-062 - the frontier, and Stage 4 priced out on evidence

**The budget law HOLDS out of sample.** Depth 8 measured **0.0783**, 95% interval
**[0.0523, 0.1044]**, which **contains** the prediction of **0.0588** from a law fitted at depths
3-7. A one-sample containment test, not a paired contrast.

| depth | success | note |
|---|---|---|
| 7 | **0.2004** | EXP-053 arm B, the current recipe |
| **8** | **0.0783** | **the frontier.** 12 of 12 seeds beat the floor |
| 9 | 0.0163 | at the floor by the 0.02 bar, though **7 of 12 beat zero** |
| chance floor at 7/8/9 | **exactly 0.0000** | measured, 12 seeds, random policy |

**Depth 9 was never a test of the law** — success is bounded below at zero, so the -0.0828
prediction can only manifest as "about zero". Reading it as confirmation would be reading a bound
as evidence.

> [!warning] **THE CENSORING TRAP - the most transferable thing this week produced.**
> The measured steps are **0.1221** (depth 7->8) and **0.0620** (8->9). **Averaging them into a
> cheaper exchange rate is wrong**: success is bounded below at zero, so an arm nearing the floor
> **must** show a smaller absolute decline. The 8->9 step is **compressed by censoring**, not
> genuinely cheaper.
>
> | basis | per depth | depth-11 cost |
> |---|---|---|
> | week-20 fitted | 0.1416 | 74.8 days/seed |
> | **measured 7->8 only (clean)** | **0.1221** | **33.1 days/seed** |
> | naive 7->9 average — **DO NOT USE** | 0.0920 | 9.4 days/seed |
>
> **Any future depth projection built on data near zero will under-price.** Use steps whose
> endpoints are both well clear of the floor.

**Stage 4 (depth-11 random scrambles) costs ~33 days per seed**, ~18 days wall clock for a 12-seed
arm. `road-to-a-solved-cube` called it *"genuinely achievable... nothing about it requires new
science"* — written before the budget law existed. **Nothing new is needed scientifically; it is
arithmetic that stops it.** The vault road note now carries a dated correction saying so.

**Provenance is not one continuous run**: depth 8 and depth 9 ran ~15 h apart at 6 and 12 workers
with the reboot between them. Same commit, seeds and config, so the contrast is unaffected — but
`RESULTS.md` says so rather than implying otherwise.

## 2. Where Phase 3 stands, for the checkpoint

**Three lines are closed, and two of them closed negative. That is the honest story.**

| line | status |
|---|---|
| **The critic** | **CLOSED and REPLICATED.** Benefit is within-episode state-dependence (EXP-056/057), replicated on fresh seeds at -0.0925, p 0.0156 (EXP-060) |
| **Episodic memory** | **CLOSED NEGATIVE, with a mechanism.** Memory HURTS (-0.0954, p 0.0056), and the recall is **noise** — matched-magnitude noise performs the same (EXP-059/061) |
| **The neuromorphic claim for Stage 3** | **REFUTED.** The `neuromod` bus was made load-bearing and bought nothing (EXP-053 arm G) |
| **Stage 4, the deliverable** | **PRICED OUT** at ~33 days/seed (EXP-062) |
| **Frontier** | **depth 8 at 0.0783** |

**Week 20's spiking-encoder training remains the one genuinely neuromorphic change that pays.**
The critic that *did* work is ordinary RL.

## 3. Open items

1. **THE PHASE 3 CHECKPOINT (week 25).** 62 numbered experiments need one coherent narrative. This
   is the largest remaining piece and it is **laptop-free**. Start from
   `docs/retired-instruments.md`, the three closed lines above, and the weekly notes 16-24.
2. **The live scientific thread: can any readout use the stored content?** EXP-059's gate showed
   stored content genuinely changes the recall (`recall_content_cos` 0.8514, well below 1.0), so
   ==the content is there and this readout cannot use it==. `MemoryReadout` concatenates a **raw**
   hippocampal read; a readout that *learned what to attend to* is untested. **Deliberately not
   started** — it is new science that will not resolve before the checkpoint.
3. **The EXP-060 level shift**, unexplained and **laptop-free to probe**: both arms sat 0.05-0.08
   lower on seeds 14-23 under the same recipe and `selected_lr.json`. Compare the new E1 encoders
   against the old on any frozen metric.
4. **EXP-055's two leads**, both needing compute, both lower value than item 2.
5. **Vault todos**: `4f13` **is** EXP-059 and `eef1` (dispatch EXP-060) is **done** — both
   Michael's call to tick. `0576` dashboard render and `0817` Phase 0/1 checkpoints still need him.
6. **Week 24's weekly note is unwritten.** Weeks 21-23 were written retroactively on 2026-09-14;
   do not let 24 drift the same way.

## 4. Standing facts

- **READ `docs/retired-instruments.md` BEFORE PUTTING AN INSTRUMENT IN A SPEC.** Five retired; all
  work as a THRESHOLD and fail as a GRADIENT.
- **READ "THE GATE-CALIBRATION RULE" IN `CLAUDE.md` BEFORE WRITING A GATE.** Seven gates used now;
  two were wrong, and every one since was right because it was calibrated across the attainable
  range and expressed as a bounded or scale-free quantity.
- **An instrument that cannot detect the defect it exists for is the gate trap in disguise, and
  only MUTATION TESTING finds it** (EXP-061's leak detector). Also in `CLAUDE.md`.
- **Confirm completion from COUNTS, never the log.** The log prints one success rate per line;
  reading it to check a run finished contaminates the aggregator you have not written yet.
  EXP-060's write-up had to disclose exactly that; EXP-061 and EXP-062 avoided it.
- **An unreachable tailscale peer is an UNKNOWN**, not a paused job. Use the three-reading test.
- **Price runs from a SAME-worker-count measurement.** Cross-worker-count scalings have come in low
  five times; same-count estimates have held.
- **Worker count should DIVIDE the cell count.** Collapsing two waves into one beat the per-cell
  penalty on EXP-062's re-run and bought the margin that saved it.
- **n >= 12 seeds. Measure the chance floor. No scipy.**

## 5. Pointers

- `experiments/062_depth_frontier/RESULTS.md` — the frontier and the censoring trap
- `experiments/059_memory_depth5/`, `061_noise_matched_recall/` — memory closed with a mechanism
- `experiments/060_flattened_critic_replication/` — the critic replicated
- `docs/retired-instruments.md`, `CLAUDE.md`, `docs/playbooks/remote-experiment-runs.md`
- Vault: `experiment-log.md` (through EXP-062), `road-to-a-solved-cube.md` (Stage 4 correction),
  `progress-tracker.md`, weekly notes 21-23
