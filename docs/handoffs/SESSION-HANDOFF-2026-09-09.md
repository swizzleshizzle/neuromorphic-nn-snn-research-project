# Session Handoff - 2026-09-09 (Week 23) - A RUN IS IN FLIGHT

> **EXP-059 IS RUNNING ON THE LAPTOP.** Dispatched 2026-09-08 23:50 laptop-local (the laptop's
> clock runs a day behind the VPS's; do not read that as a stale timestamp). 72 cells, 6 workers,
> **estimated ~30 h**, so it spans at least one laptop sleep.
>
> **`main` is at `60245b6` and clean. Work sits on branch `exp-059-memory-depth5` (`bbea390`, 5
> commits ahead), unmerged and pushed.**

## 0. RESUMING THE RUN - do this first

**Check whether it finished before anything else:**

```bash
ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\probe_run.ps1 -OutDir "C:\Users\mlgbr\wt-exp053\experiments\059_memory_depth5\outputs"'
ssh -n laptop 'powershell -NoProfile -Command "@(Get-ChildItem -Path \"C:\Users\mlgbr\wt-exp053\experiments\059_memory_depth5\outputs\" -Filter \"exp059_*.json\").Count"'
```

**72 records means done.** Then:

```bash
scp "laptop:C:/Users/mlgbr/wt-exp053/experiments/059_memory_depth5/outputs/exp059_*.json" experiments/059_memory_depth5/outputs/
scp "laptop:C:/Users/mlgbr/wt-exp053/experiments/059_memory_depth5/outputs/*_head.pt" experiments/059_memory_depth5/outputs/
```

**JUDGE PROGRESS BY CPU-HOURS PER WORKER, NEVER BY WALL CLOCK.** EXP-058 slept **26 h of its
39.7 h** in transit and finished correctly. A worker's CPU time divided by completed cells is the
real per-cell cost:

```bash
ssh -n laptop 'powershell -NoProfile -Command "Get-Process | Where-Object { $_.Name -match \"^python\" } | ForEach-Object { $_.Id.ToString() + \" \" + [math]::Round($_.CPU/3600,2) }"'
```

**If it stopped early**, `--skip-existing` makes resuming free and lossless (seeded runs are
byte-identical). Re-dispatch with the same launcher.

**Still to do when it lands:** write `aggregate.py` against the pre-registered contract, produce
`RESULTS.md`, run the suite in chunks, merge `--no-ff`, delete the branch, add the vault row.
**There is no aggregator yet** - that is deliberate, so it can be written from the spec rather than
from the numbers.

## 1. What EXP-059 is, and why the design is shaped this way

**The memory question, at depth 5, at n=24, with a gate that works.** Three arms varying ONLY the
readout, on `exp043_capped_d5`'s config field for field: a working depth-5 policy at **0.3229**,
frozen encoder, already measured at 24 seeds.

| arm | readout | tag |
|---|---|---|
| A, amnesic | `memory_amnesic` | `exp059_amnesic_d5` |
| M, memory | `memory` | `exp059_memory_d5` |
| S, shuffled | `memory_shuffled` | `exp059_shuffled_d5` |

**It exists because EXP-058 was VOID.** That experiment's gate required `mean_n_stored > 10`, a
quantity bounded by episode length, where episodes average 7.76 steps. **Unsatisfiable by
construction**, and it gated on **storing** when the arms differ at the **read** site. Its seeds are
also burned: runs here are byte-identical, so re-running its cells under a corrected gate
reproduces exactly the void records.

**Changing venue to depth 5 fixes both at once**: a different measurement rather than a repeat,
seeds 12-23 fresh, n=24 halving the standard error, and **no encoder manufacturing** because all 24
`exp040_encoder_s*.pt` exist.

### The gate was calibrated BEFORE the spec, and that caught a third mistake

The first instinct was "arm A's recall norm is exactly zero". **Wrong**: `MemoryReadout`'s own
docstring records that 65% of the recall block is a memory-free transform of the current concept.
The gate now measures the read site across its full range, at `bb1efa5`:

| attractor state | `recall_content_cos` |
|---|---|
| empty, nothing stored | **1.000000** (the gate CAN fail) |
| random loaded | 0.9437 |
| **real depth-5 run** | **0.8128** (the gate CAN pass) |
| docstring, independent, 79 steps | 0.802 |

**Condition: arm M below 0.95**, plus arm S's `unshuffled_frac` below 0.20. A cosine is bounded and
scale-free, so it cannot repeat EXP-057's regime-dependence.

### Two limits recorded up front

- **Claim 1 is NOT directional.** EXP-058's unlicensed ordering suggests memory may HURT, so a
  significant **-0.05** is pre-registered as a real finding, not a failed confirmation.
- **n=24 is still only ~30-40% powered at a 0.03 effect**, which is roughly what EXP-058 saw.
  `revisit_rate` is the better-powered instrument and is why Claim 2 is a claim, not a footnote.

## 2. What else changed this session

- **`bbd0` audit done and ticked.** The uniform modal floor is budget-dependent; 0.354 is the
  depth-3 figure. Two wrong citations fixed (EXP-038's spec body, EXP-037's depth-4 Claim 5);
  neither changed a conclusion. The recurrence fix is the budget-by-depth table now in
  `modal_action_fraction`'s docstring, extended to depth 7 (0.3010).
- **`c16a` done.** `requirements.txt` gained `-e .[server]`; without it a fresh checkout could not
  COLLECT `tests/server`, and a collection error fails the whole run.
- **`3ef2` done.** The ssh exit-code guidance now points at the **tailscale last-seen field**, with
  both real strings recorded verbatim.
- **`7741` reworded** into `4f13`: the memory re-ask is NOT cheap. The CLI has no edit command, so
  the old entry is ticked and its text is now misleading in the completed list.
- **The playbook gained the comma trap and the background-kill behaviour**, out of rotating
  handoffs and into the durable doc.

## 3. Open items

1. **Finish EXP-059** (section 0).
2. **Repeat EXP-056 at higher n, ~25 h**, of which 14 h is re-manufacturing encoders. `p 0.0234`
   against `0.025` is thin for a result two experiments lean on.
3. **EXP-055's two leads**: what one epoch builds that helps policy while making `S` worse than
   random, and whether `e2` is a cheaper recipe at 74% of `e10` for a fifth of the cost.
4. **A standing note on the five retired instruments.** Enough instances now to write once.
5. **Vault, needs Michael**: `0576` see the dashboard render once, `0817` decide the Phase 0/1
   progress-tracker checkpoints.

## 4. Standing facts

- **READ "THE GATE-CALIBRATION RULE" IN `CLAUDE.md` BEFORE WRITING A VALIDITY GATE.** Two of three
  experiments that used one got it wrong, both the same shape: a threshold chosen in one regime and
  applied in another. **Compute a gate's attainable range before committing it.**
- **The critic question is CLOSED.** Its benefit is within-episode state-dependence, not
  calibration. Everything without it sits 0.1358-0.1558; the full critic sits at 0.2004.
- **FIVE instruments move against policy quality**: the EXP-033 probe, pretraining move-accuracy,
  the entropy trace, `S`, and `critic_ev`. **Use `revisit_rate` and `optimality`.**
- **Unanimity at p 0.0005 measures an instrument's consistency, not its link to the outcome.**
- **Week 20's spiking-encoder training is the ONE neuromorphic change.**
- **Costs, measured:** depth-7 frozen cell ~2.8 h; depth-6 memory cell 3.37 h; depth-5 memory cell
  **unmeasured, estimated 2.5 h**. More workers buy no wall clock on RL cells.
- **n >= 12 seeds. Measure the chance floor. No scipy.**

## 5. Operational

- **Never pass comma-separated arguments over ssh**; the failure **exits zero**.
- **Long background commands are killed here around 2-3 h** with empty output. The remote run
  survives; only the notification is lost. `| tail` buffers until exit, so a killed piped command
  tells you nothing. Foreground chunks under 600 s are reliable.
- **The Bash default timeout is 120 s and 600 s is a hard ceiling.**
- **Sync the laptop worktree with `sync_repo.ps1 -Repo ... -Branch ...`, never a bare checkout**,
  and remember `exp040_encoder_s*.pt` are NOT tracked in git: all 24 live on the MAIN checkout and
  were copied into the worktree with hash verification.
- **`C:\Users\mlgbr\repo-attic` holds 18+ folders** that nothing prunes.

## 6. Pointers

- `docs/superpowers/specs/2026-09-09-exp059-memory-depth5-design.md` - the live contract
- `experiments/058_memory_reask/RESULTS.md` - the VOID write-up and the EXP-030 trap it reproduced
- `experiments/057_constant_critic/RESULTS.md` - calibration ruled out
- `CLAUDE.md` - the gate-calibration rule; the re-measured test runtimes
- `docs/playbooks/remote-experiment-runs.md` - dispatch, the comma trap, background kills,
  last-seen diagnosis
- Vault: `experiment-log.md` (through EXP-058 plus four dated corrections)
