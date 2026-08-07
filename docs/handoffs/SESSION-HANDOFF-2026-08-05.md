# Session Handoff - 2026-08-05 (Wed) - Week 18 session 2

> **EXP-037 IS RUNNING ON THE LAPTOP.** Dispatched 2026-08-05 around 20:05 ET, roughly
> **4.62 M env steps across 48 runs on 10 workers**. Repo is clean at `022d8b8` and pushed;
> the laptop is on the same commit.
>
> Read `CLAUDE.md`, then this file, then
> `docs/superpowers/specs/2026-08-05-exp037-curriculum-weighting-design.md` for the
> pre-registered contract this run is answering.

## 0. State check

```bash
git log --oneline -1                 # expect 022d8b8 or later
git status --short                   # expect clean
ssh -n laptop 'powershell -NoProfile -Command "(Get-Process | Where-Object { $_.Name -eq \"python3.13\" }).Count"'
```

## 1. Check on EXP-037

```bash
ssh -n laptop 'powershell -NoProfile -Command "$d=\"C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\experiments\037_curriculum_weighting\outputs\"; \"records=\" + (Get-ChildItem $d -Filter *.json).Count + \"/48\"; \"workers=\" + (Get-Process | Where-Object { $_.Name -eq \"python3.13\" }).Count; \"tracebacks=\" + (Select-String -Path (Join-Path $d run.log) -Pattern \"Traceback\").Count"'
```

**48 records expected**, plus 48 `*_head.pt` checkpoints.

> [!warning] Read the RIGHT signals, in this order
> 1. **Worker count alive is not progress.** A live pool can be hung. The signal that
>    separates working from hung is **CPU accumulating between two samples**; EXP-036 needed
>    exactly that check.
> 2. **Record count is not progress either**, because the cells cost different amounts. EXP-036
>    read 82/96 while the remaining 14 were its most expensive runs.
> 3. **Zero records for hours is normal.** Runs only write on completion.
> 4. **An ssh failure is not a dead run.** Check `tailscale status` for the peer's last-seen.
>    The laptop travels; it dropped off the tailnet mid-run on Aug 3-4 and kept computing.

Fetch when done, then run the analysis (already written, do not write it in the morning):

```bash
scp "laptop:C:/Users/mlgbr/Desktop/Projects/neuromorphic-nn-snn-research-project/experiments/037_curriculum_weighting/outputs/*" experiments/037_curriculum_weighting/outputs/
.venv/bin/python experiments/037_curriculum_weighting/aggregate.py
```

`aggregate.py` needs EXP-036's records present at
`experiments/036_generalisation_gap/outputs/` - **they ARE the 25% comparator** and every claim
is paired against them. They are gitignored, so if the directory is empty, re-fetch from the
laptop.

## 2. What EXP-037 asks

One axis: **the share of the episode budget spent at the evaluated depth**, remainder split
equally among the bootstrap stages.

| share | weights | depth-4 schedule | env steps | status |
|---|---|---|---|---|
| 12.5% | `(7,7,7,3)` | 2916/2916/2916/**1252** | 75,008 | new, **the control** |
| 25% | `(1,1,1,1)` | 2500/2500/2500/**2500** | 80,000 | **EXP-036 = 0.1591**, not re-run |
| 50% | `(1,1,1,3)` | 1666/1666/1666/**5002** | 90,008 | new |
| 75% | `(1,1,1,9)` | 833/833/833/**7501** | 100,004 | new |
| 100% | - | direct training | - | **EXP-034 refuted this** |

Plus one depth-6 arm at 50% share against EXP-036's 0.0000.

The two known endpoints are what make this a dose-response with a **predicted interior
optimum** rather than a fishing trip. Full contract in the spec and in `run.py`'s docstring:

1. **Lever?** 50% beats 25% by >= 0.03 with p <= 0.05. The bar is 0.4 sd of EXP-036's measured
   depth-4 spread (0.0739), a 19% relative gain on 0.1591.
2. **Interior optimum?** 50% > 75% confirms; 75% >= 50% means the next experiment is 85/95%.
3. **The control:** 12.5% MUST be worse than 25%. If it is not, success is not tracking share
   and any 50% win needs a different explanation than starvation.
4. **Depth 6:** does anything move it off 0.0000? Still zero means the failure is collapse, not
   starvation, and the lever is EXP-031/032 territory.
5. **Mechanism:** does a gain arrive with modal fraction FALLING?
6. **The confound**, disclosed: holding episodes fixed does not hold compute fixed. The 75% arm
   spends **25% more env steps** than the 25% arm at the same episode budget. A 75% win by a
   margin under 25% is not separable from its extra compute on this design.

## 3. Two traps avoided, one caught at the last moment

**The weighting hack would have been unattributable.** Weights can be faked by repeating a
depth in the curriculum tuple - `(1,2,3,4,4,4)` does give depth 4 half the budget. But
`run_cube_baseline` rebuilds `ShellCubeEnv` per stage with `random.Random(train_seed)`, so
consecutive identical stages **replay the same start-state sequence**. It is not 5,000 fresh
episodes, it is one 1,666-episode sequence looped. Real weights give one continuous stage.

**A 22-minute unit test.** The first end-to-end neutrality test ran at depth 4 with a small
episode budget. Shrinking the training budget does nothing about an evaluation over 333 states
at 11 steps. Moved to depth 3 with `heldout_cap=8`, which still exercises a real multi-stage
curriculum. **When making a test cheap, check which phase actually dominates.**

> [!danger] **The laptop was 12 commits behind and the pull was silently failing.**
> Caught immediately before dispatch. `git status` on the laptop showed 48 **untracked**
> `*_head.pt` files at paths that are now **tracked** in git - because EXP-036's checkpoints
> were committed from the VPS while the laptop still held its own generated copies. Git refuses
> to overwrite untracked files on merge, so `git pull --ff-only` failed and left HEAD at
> `87a965b`. `curriculum_weights` appeared **zero times** in the laptop's source.
>
> The launcher runs its own `git pull` and would have failed the same way, then run old code.
> That would have died loudly on an unexpected keyword rather than silently, but it would still
> have burned a dispatch.
>
> **Resolved** by moving the 48 files to `C:\Users\mlgbr\exp036_heads_bak`, pulling, then
> verifying by hash that all 48 restored-from-git checkpoints are byte-identical to the laptop
> originals. The backup is still there and can be deleted.
>
> **Standing consequence: committing generated artifacts creates a pull conflict on whichever
> machine generated them.** Verify `git rev-list --count HEAD..origin/main` is 0 on the laptop
> before every dispatch, and grep the source for the new symbol. Do not trust a silenced pull.
>
> **IT WILL HAPPEN AGAIN ON THE NEXT DISPATCH, and it is predictable.** EXP-037's 48 head
> checkpoints were committed from the VPS at `034f1a8`, while the laptop still holds its own
> untracked copies at the same paths. **The next `git pull` on the laptop will fail exactly the
> same way.** Clear it first:
>
> ```bash
> ssh -n laptop 'powershell -NoProfile -Command "cd C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project; New-Item -ItemType Directory -Force C:\Users\mlgbr\exp037_heads_bak | Out-Null; Get-ChildItem experiments\037_curriculum_weighting\outputs -Filter *_head.pt | Move-Item -Destination C:\Users\mlgbr\exp037_heads_bak -Force; git pull --ff-only origin main; git rev-list --count HEAD..origin/main"'
> ```
>
> Then confirm the restored files hash-match the originals, as was done for EXP-036. The EXP-036
> backup at `C:\Users\mlgbr\exp036_heads_bak` was verified identical and can be deleted.
>
> **The better fix is to stop hitting this**: either have the laptop's `outputs/` be the only
> copy and never commit from the VPS, or add a `git clean` step to the launcher. Neither is done.

## 4. The 10-worker prediction

EXP-036 ran **16 workers at 920 MB private each**, drove system commit to **48.6 of 50.4 GB
(96%)**, and held utilisation at a measured **43.1%** - the workers spent most of their time
paging. EXP-037 runs **10 workers**: 914 MB each, commit **25.4 of 52.4 GB (49%)** at launch.

> [!success] **PREDICTION CONFIRMED, measured 2026-08-05 over a 421 s interval.**
> ```
> 10 workers:  7.42 effective   74.2% utilisation   commit 49%
> 16 workers:  6.90 effective   43.1% utilisation   commit 96%
> ```
> **Ten workers beat sixteen outright** - 7.5% more throughput from 37.5% fewer processes. The
> laptop is memory-bound, not core-bound, and EXP-036 paid a real penalty for oversubscription
> despite having 22 logical cores.
>
> Written up as a standing rule in `docs/playbooks/remote-experiment-runs.md`: **choose the
> worker count from memory headroom, not core count**, and budget from PRIVATE bytes (~920 MB
> each) rather than working set (~80 MB). The playbook's old 195 MB figure is a working-set
> number and understates by ~4.7x, which is what made 16 look safe.
>
> **Whether fewer than 10 is better again is unmeasured.** Two points do not fix a curve.

Measure it the same way, two CPU samples over a known interval:

```bash
ssh -n laptop 'powershell -NoProfile -Command "$p = Get-Process | Where-Object { $_.Name -eq \"python3.13\" -and $_.CPU -gt 30 }; [math]::Round((($p | Measure-Object CPU -Sum).Sum),1)"'
```

**Revised ETA:** EXP-037 is 4.62 M steps against EXP-036's ~4.46 M, at 7.42 effective workers
versus 6.90, so roughly **22 h** - about the same wall clock for 13% more work.

## 5. Housekeeping done this session

- **Six `neuromorphic` todos seeded** via `secretary`; there were none before, so every open
  item on this project lived only in handoffs. They sync to Michael's phone.
- **The stale Aug 8 calendar milestone was rewritten**, not deleted: it still said "don't rush
  past 1-move scrambles until solve rate >80%". It now carries the real standing and points at
  `road-to-a-solved-cube.md`. **It duplicates the recurring coding block in the same slot** -
  delete it if that is noise.
- **Vault updated** for EXP-036 across four notes: the week-18 note gained its Tuesday session,
  `experiment-log` gained the result and the break-bar callout, `progress-tracker` gained a
  depth-frontier metric line, and `road-to-a-solved-cube` gained a Stage 1 completion block.

## 6. Open

Carried forward, unchanged from 2026-08-03 section 6 except where noted:

1. `cubeNet.ts` B/D orientation - **pinned with `it.fails`, not fixed**. Needs a browser. Remove
   the `.fails` when fixed or it will start erroring. Todo `^f772`.
2. Nothing in the cube dashboard has been seen rendering in a browser. Todo `^0576`.
3. Phase 0 / Phase 1 checkpoints in the vault `progress-tracker` unticked. Needs Michael. Todo `^0817`.
4. `requirements.txt` omits the `[server]` extra. Todo `^c16a`.
5. Playbook: the exit-255 guidance should say to check the tailscale peer last-seen. Todo `^3ef2`.
6. `verify_instrument_neutrality.py` reports a `config` mismatch as one opaque key rather than
   naming the differing sub-key. Cost several minutes on 2026-08-04 to find it was provenance
   (EXP-035's records predate the three seed fields).
7. EXP-025 still has no committed `RESULTS.md`; cite ADR 0001 Amendment 2, not "EXP-025".
8. Live trace streaming during training is a spec, not built.
9. Shuffle-control dilution (`unshuffled_frac` 0.321 at depth 1) if a shuffle control is reused.

## 7. Pointers

- Pre-registration: `docs/superpowers/specs/2026-08-05-exp037-curriculum-weighting-design.md`
- Driver and contract: `experiments/037_curriculum_weighting/run.py`
- Analysis: `experiments/037_curriculum_weighting/aggregate.py`
- Launcher, with the three Windows failure modes: `experiments/037_curriculum_weighting/launch037.ps1`
- What EXP-037 is measured against: `experiments/036_generalisation_gap/RESULTS.md`
- Strategy: vault `Neuromorphic Development/road-to-a-solved-cube.md`
- Previous handoff: `docs/handoffs/SESSION-HANDOFF-2026-08-03.md` (sections 1.6 and 2.5 still apply)
