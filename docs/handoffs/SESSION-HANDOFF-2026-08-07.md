# Session Handoff - 2026-08-07 (Fri) - Week 18 session 3

> [!important] **SUPERSEDED for current state. EXP-038 is now IN FLIGHT.**
> **Start from `docs/handoffs/SESSION-HANDOFF-2026-08-08.md` instead.**
>
> Section 0 says the laptop is idle and section 3 warns the pull trap will bite the next
> dispatch. Both are **no longer true**: EXP-038 was dispatched 2026-08-07 20:15:11 at commit
> `6416379` (~21 h, ETA ~17:15 Sat), and the pull trap was fixed structurally in `470419f` and
> verified against the live 48-file collision.
>
> Section 2's open decision was resolved as **candidate 1, the depth-6 collapse fix**.
> Candidates 2 (the EXP-030 memory re-ask) and 3 (the encoder, vault Stage 2) remain open and
> are carried forward. **Sections 1, 4, 5 and 6 are still accurate and worth reading.**

> **Nothing is in flight. The repo is clean at `ab49f91` and pushed. The laptop is idle.**
>
> Week 18 landed **EXP-036 and EXP-037**, and between them they **close the curriculum as a
> lever**. There is no cheap reallocation left; the next move is architectural or
> credit-assignment.
>
> Read `CLAUDE.md` first, then this file. Strategy is **not** in this repo: it is the vault at
> `300 Efforts/Active/Coding/Neuromorphic Development/road-to-a-solved-cube.md`.

## 0. State check

```bash
git log --oneline -1                 # expect ab49f91
git status --short                   # expect clean
ssh -n laptop 'powershell -NoProfile -Command "(Get-Process | Where-Object { $_.Name -eq \"python3.13\" }).Count"'   # expect 0
```

The laptop travels with Michael and is frequently off the tailnet. **An ssh timeout is not a
problem signal** - check `tailscale status | grep swizzlesduo` for the peer's last-seen before
concluding anything. Full suite is **436 tests**, about 24 min, all green at `ab49f91`.

## 1. What week 18 established

| exp | question | answer |
|---|---|---|
| **036** | Is there a train/held-out gap, and where does the curriculum break? | Breaks at **depth 5**, as pre-registered. Depth 4 works at **0.1591** (51x its floor). Depth 6 is **0.0000** and fails by *collapse*, not weak learning. The depth-3 gap is real (+0.1093, p 0.0059) but **inconclusive** against its own pre-registered 0.15 bar. |
| **037** | Does the curriculum's stage weighting matter? | **Refuted, and not as a null.** Share of budget at the evaluated depth 12.5 / 25 / 50 / 75% gives 0.1535 / **0.1591** / 0.1078 / 0.0921. More budget at the evaluated depth makes things *worse*. Depth 6 at 3x its episodes stayed at **0.0000**, modal fraction rising 0.975 -> 0.982. |

**Standing position:** depth 3 at 0.500 (30k budget) / 0.397 (10k), depth 4 at 0.1591, depth 5
broken, depth 6 collapsed. All on a frozen randomly-initialised brain and a `Linear(64 -> 6)`
head: **390 trainable parameters**.

> [!important] Refuted and CLOSED. Do not revisit without a new reason.
> - **Width** (EXP-033) - the representation is not the first bottleneck.
> - **Volume alone** (EXP-034) - extra episodes without the curriculum are worth -0.003 at p 1.000.
> - **Curriculum stage weighting** (EXP-037) - the equal split is at or near optimal.
> - **Starvation as the explanation for depth 6** (EXP-037) - it is collapse.
>
> The mechanism tying the last two together: **time at the deepest stage drives the policy
> toward a constant action**, because sparse reward there means more updates on mostly-failed
> episodes. The best-performing arm was the *least* collapsed (modal 0.685 vs 0.757).

## 2. THE JOB NEXT - this is the open decision for Friday

**Friday is a 1.5 h study/design block** (7:00-8:30pm ET). Saturday Aug 8 is the 3 h coding
block. The shape that worked twice this week: **design and dispatch on the weeknight, let the
run go overnight**. A 48-run sweep is roughly 20 h at 10 workers.

Candidates, and none is obviously dominant - this is a real choice:

1. **Depth-6 collapse fix.** EXP-032's stabilizers (entropy bonus, advantage normalization) were
   refuted *at depth 3*, where the failure was NOT collapse. At depth 6 it demonstrably is
   (modal 0.982). That is a different question at a different depth and it is now well
   motivated. **Cheapest to design, since the machinery exists.**
2. **The EXP-030 memory re-ask.** The thesis-relevant item, waiting since week 17, and now
   **cheap**: 96 trained heads are committed across EXP-036/037, so no retrain is needed. The
   risk is unchanged - at these depths with a `2d+3` budget, cycling may be rare enough to null
   again. Worth reading EXP-030's three-arm design before repeating it.
3. **Stage 2, unfreeze the encoder.** The vault's week-19 plan and the moment the SNN stops
   being a fixed random projection. **Biggest build, most thesis-relevant**, and EXP-033's probe
   gives a measured success criterion: does the ceiling rise at depths 4-6? Too large for one
   weeknight; would need Saturday too.
4. **Depth 4 interventions generally.** It is the frontier (learning, capped, no gap, no
   collapse) and it is where any of the above would show first.

**My read, for what it is worth:** (1) is the natural next experiment - smallest design, a
sharp pre-registerable question, and it directly follows EXP-037's finding. (3) is the one that
matters most for the project's actual thesis and should not be deferred indefinitely.

## 3. The trap that WILL bite the next dispatch

> [!danger] The laptop's `git pull` will fail again, silently, and it is predictable
> EXP-037's 48 head checkpoints were committed **from the VPS** at `034f1a8`, while the laptop
> still holds its own **untracked** copies at the same paths. Git refuses to overwrite untracked
> files on merge, so the pull fails and leaves the laptop on an old commit. **This exact thing
> happened before the EXP-037 dispatch** - the laptop was 12 commits behind and
> `curriculum_weights` appeared zero times in its source. It was caught only by checking.
>
> Clear it before dispatching:
>
> ```bash
> ssh -n laptop 'powershell -NoProfile -Command "cd C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project; New-Item -ItemType Directory -Force C:\Users\mlgbr\exp037_heads_bak | Out-Null; Get-ChildItem experiments\037_curriculum_weighting\outputs -Filter *_head.pt | Move-Item -Destination C:\Users\mlgbr\exp037_heads_bak -Force; git pull --ff-only origin main; git rev-list --count HEAD..origin/main"'
> ```
>
> **Then verify, every time:** `git rev-list --count HEAD..origin/main` is 0 **and** grep the
> laptop's source for the new symbol you just added. Do not trust a silenced pull.
>
> **The real fix is not done.** Either stop committing `outputs/` from the VPS, or add a
> `git clean` step to the launcher. Worth doing before the next dispatch rather than after.

## 4. Operational facts worth not rediscovering

- **10 workers beat 16**, measured: 7.42 effective at 74.2% utilisation versus 6.90 at 43.1%.
  The laptop is **memory-bound, not core-bound**, despite 22 logical cores. Size the pool from
  memory headroom and budget from **private bytes (~920 MB/worker)**, not working set (~80 MB).
  In `docs/playbooks/remote-experiment-runs.md`.
- **`ssh laptop`, never the fully-qualified hostname.** ssh matches `Host` patterns against what
  you typed, so the long form misses the config block and fails with a `publickey` error that
  looks exactly like a missing `authorized_keys` entry.
- **Estimate wall clock from measured throughput at the concurrency you will actually run.**
  `brain.step` at 90 ms is single-step latency. EXP-035's 153 ms/step was measured at a
  concurrency that never saturated the pool. Both under-estimated by large factors.
- **Seeded runs are NOT reproducible across platforms.** Byte-identity holds within a machine.

## 5. The habit that paid for itself repeatedly

**Measure the instrument on a case whose answer you already know.** Six measurement bugs were
caught across week 18, all the same shape - *a check that cannot distinguish the states it
exists to separate*:

| the check | what it could not see |
|---|---|
| gap threshold 0.05 on one seed | the null arm swings +-0.10 at n=1 |
| "broken = 2x the measured floor" | the floor collapses with depth; at depth 5 it is exactly 0.0000 |
| `brain.step` = 90 ms | throughput under load, off by ~2x |
| summed CPU of *live* workers | exited workers silently leave the sum |
| worker count alive | a hung pool looks identical to a working one |
| record count | cheap cells and expensive cells count the same |

The three caught before costing anything were caught by testing the instrument against a known
answer: the random arm cannot overfit, so its gap **is** the null; a synthetic failing depth-6
**should** read BROKEN; derived cube corners **must** be adjacent across a net border.

**A related failure, logged in EXP-037:** a decision rule written for the outcome you expect can
produce a true-but-misleading verdict when the result goes the other way. `aggregate.py` printed
"INTERIOR OPTIMUM" and "CONTROL HOLDS" on a result that supported neither. Read the ordering,
not just the one comparison the rule names.

## 6. Open debt

Tracked as todos (visible on Michael's phone via `secretary todo list --project neuromorphic`):

1. `^f772` **cubeNet.ts B/D face orientation** - pinned with `it.fails`, not fixed. Needs a
   browser. **Remove the `.fails` when fixed or the test starts erroring.** Note the 2026-08-02
   description was backwards: B is the only face row-major gets *right*; U, R, F, D, L are wrong.
2. `^0576` Nothing in the cube dashboard has been seen rendering in a browser.
3. `^0817` Phase 0 / Phase 1 checkpoints in the vault `progress-tracker` unticked though every
   week beneath them is done. **Needs Michael** - cannot be verified from here.
4. `^c16a` `requirements.txt` omits the `[server]` extra, so a fresh checkout fails
   `tests/server/` collection.
5. `^3ef2` Playbook: the exit-255 guidance should say to check the tailscale peer last-seen.

Not tracked as todos:

6. `verify_instrument_neutrality.py` reports a `config` mismatch as one opaque key rather than
   naming the differing sub-key.
7. EXP-025 still has no committed `RESULTS.md`; cite ADR 0001 Amendment 2, not "EXP-025".
8. Live trace streaming during training is a spec, not built.
9. The Aug 8 calendar milestone was rewritten with the real standing but still **duplicates the
   recurring coding block in the same slot**. Delete it if that is noise.

## 7. Pointers

- Results: `experiments/03{6,7}_*/RESULTS.md`
- Pre-registrations: `docs/superpowers/specs/2026-08-0{3,5}-*.md`
- Analysis, re-runnable any time: `experiments/03{6,7}_*/aggregate.py`
  (EXP-037's needs EXP-036's records present - **they are the 25% comparator**, and they are
  gitignored, so re-fetch from the laptop if `outputs/` is empty)
- Remote runs: `docs/playbooks/remote-experiment-runs.md`
- **Strategy: vault `Neuromorphic Development/road-to-a-solved-cube.md`**
- Week log: vault `Weekly Notes/week-18-generalisation-and-depth.md`
- Previous handoffs: `SESSION-HANDOFF-2026-08-05.md` (sections 3-5), `-08-03.md` (sections 1.6, 2.5)
