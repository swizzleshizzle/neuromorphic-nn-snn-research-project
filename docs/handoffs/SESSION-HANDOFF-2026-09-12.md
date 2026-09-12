# Session Handoff - 2026-09-12 (Week 23) - EXP-059 AND EXP-060 BOTH COMPLETE

> **Nothing is running. The laptop is FREE. `main` is at `1c82a33`, clean, no branches, no PRs.**
> **Suite: 591 passed** under `-m "not slow"` (356 + 31 + 204), plus 22 slow-marked deselected.
>
> **Two experiments landed this week and they point in opposite directions:**
> **memory HURTS a working policy**, and **the critic's within-episode structure is real and
> replicates.**

## 1. EXP-059 - episodic memory HURTS

**`M` minus `A` = -0.0954 at p 0.0056**, clearing the pre-registered bar downward and Bonferroni.
Arms: amnesic **0.3138**, memory **0.2183**, shuffled **0.1979**.

**`M` minus `S` is +0.0204 at p 0.4268 - correct memory is indistinguishable from WRONG memory.**
Reading the attractor costs 0.095; the content being right buys nothing measurable. Arm A is not
"no recall" (65% of its recall block is a memory-free transform of the current concept), so
**adding stored content is the harm.**

**A two-arm design would have published a win for the third time.** Mechanism claim did NOT
confirm and its sign runs the wrong way (`revisit_rate` +0.0157, p 0.4450), so this is not the
anti-cycling story. Full write-up: `experiments/059_memory_depth5/RESULTS.md`.

## 2. EXP-060 - EXP-056 REPLICATES, and the effect grew

**`F` minus `B` = -0.0925 at p 0.0156** on seeds 14-23 alone, clearing the bar and Bonferroni,
W-L-T 2-8-0, exact over 1,024 flips. **Larger than the original -0.0646**, which is the opposite of
what regression to the mean predicts for a result selected for significance; paired sd is
near-identical across blocks (0.0891 vs 0.0814).

**The critic conclusion no longer rests on one knife-edge contrast.** Read beside EXP-057 -
calibration alone was worth +0.0088 at p 0.7822 - both halves of "the benefit is within-episode
state-dependence" have now been tested twice.

> [!warning] **THE POOLED n=22 FIGURE (-0.0773, p 0.0005) HAS TWO PROBLEMS, NOT ONE.**
> It inherits **optional stopping** (seeds were added *because* the p-value was marginal), **and it
> pools across a LEVEL SHIFT**: both arms sit 0.05-0.08 lower on the fresh seeds (F 0.0540 vs
> 0.1358, B 0.1465 vs 0.2004). The contrast replicates **because it is paired within seed** - that
> is what a paired design buys - but the blocks are not samples from the same level.
> **Quote Claim 1, not the pooled number. The shift is flagged and NOT explained.**

**Disclosed in its RESULTS.md:** EXP-060's aggregator was written *after* the run, unlike
EXP-059's, because verifying completion meant reading the log tail and ~12 individual cell values
had been seen. Every rule applied was fixed in the spec before dispatch.

## 3. Method lessons worth more than either result

1. **`docs/retired-instruments.md`** - the standing note. All five work as a THRESHOLD and fail as
   a GRADIENT.
2. **An unreachable tailscale peer is an UNKNOWN, not a paused job.** That misread let a dead run
   be reported healthy for 13 hours after Windows Update destroyed EXP-059's first attempt
   (~19 CPU-hours). The playbook's table is corrected with a three-reading test.
3. **Cross-worker-count cost estimates under-price, four instances now.** EXP-060 overran in all
   three phases; EXP-059 in its one. **The 0.16-vs-0.115 s/step ratio itself needs re-measuring**;
   until then treat such an estimate as a floor and add 20-40%.
4. **A gate's pre-registered wording binds even when you find a looser precedent.** EXP-060's
   spec said every-stage where EXP-056's code used any-stage; every-stage was implemented and
   passed. **Noticing a looser precedent is a reason to write the next spec better, not to switch
   mid-analysis.**
5. **Dependency chains are longer than specs assume.** EXP-060 needed an EXP-043 depth-6 baseline
   nobody had costed, found at dispatch. **Check the artifact inventory before pricing a run.**

## 4. Open items

1. **The live scientific thread: WHY does the memory read hurt?** EXP-059's gate ruled out an
   *empty* attractor, far narrower than ruling out a badly scaled or uninformative recall code.
   **This is the most interesting open question on the board.**
2. **The EXP-060 level shift.** Both arms ~0.06 lower on seeds 14-23 with the same recipe and the
   same `selected_lr.json`. Nothing distinguishes seed-set difficulty from anything about the
   2026-09-12 encoder manufacturing run. Cheap to probe: compare the new E1 encoders against the
   old ones on any frozen metric.
3. **EXP-055's two leads**, both needing compute.
4. **Vault `4f13` IS EXP-059 and can be ticked**; `eef1` (dispatch EXP-060) is **done** - both
   Michael's call.
5. **Vault, needs Michael**: `0576` dashboard render, `0817` Phase 0/1 checkpoints.

## 5. Standing facts

- **READ `docs/retired-instruments.md` BEFORE PUTTING AN INSTRUMENT IN A SPEC.**
- **READ "THE GATE-CALIBRATION RULE" IN `CLAUDE.md` BEFORE WRITING A GATE.** Five gates used now;
  two were wrong, and the two most recent were right because they were calibrated across the
  attainable range and expressed as bounded or scale-free quantities.
- **The critic question is CLOSED and now replicated**: within-episode state-dependence.
- **The memory question is ANSWERED at depth 5**: the recall read is harmful and its content's
  correctness is worth nothing measurable.
- **Nothing is durable until a cell completes.** Records are written per cell.
- **n >= 12 seeds. Measure the chance floor. No scipy.**

## 6. Pointers

- `experiments/059_memory_depth5/RESULTS.md`, `experiments/060_flattened_critic_replication/RESULTS.md`
- `docs/retired-instruments.md`, `CLAUDE.md`
- `docs/playbooks/remote-experiment-runs.md` - the corrected last-seen table and cost warning
- Vault: `experiment-log.md`, through EXP-060 plus two gate-calibration addenda
