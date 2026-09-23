# Reproducibility audit (week 25, 2026-09-22)

**Why now:** Phase 4's stated deliverable is a **public release**. A release makes a claim about
reproduction that had never been tested, and `docs/phase3-honest-assessment.md` grades
documentation as **Met** partly on that basis. This audit measures what a fresh clone on a
different machine actually reproduces.

> **THE ANSWER, in one line.** ==**Published numbers reproduce by RE-EVALUATING the tracked
> checkpoints. They do NOT reproduce by retraining.**== Both halves are measured below, and the
> second half applies to every seeded run in the project, not just to encoder pretraining.

## 1. Retraining is not portable, and not by a little

Two independent probes, on two machines, comparing weights byte for byte.

| probe | what it trains | laptop (made the originals) | this VPS (different x86) |
|---|---|---|---|
| **E0 encoder** (EXP-040 phase 1) | the pretrained sensory encoder | **12 of 12 byte-identical** | **0 of 2** |
| **EXP-036 depth 3** | a policy head, **no pretrained encoder at all** | (originals) | **0 of 1** |

**The differences are structural, not last-bit float noise:**

| probe | parameters matching | max abs delta | cosine |
|---|---|---|---|
| E0 encoder, seed 0 | 1,909 / 26,824 (**7.1%**) | 2.38 | **0.771** |
| EXP-036 head, seed 0 | **0 / 390 (0%)** | 3.60 | **0.524** |

==**EXP-036 is the load-bearing probe** because it takes no pretrained encoder: its brain is
frozen at random init, so the only inputs are code and a seed.== It still fails to reproduce.
So this is **not** a quirk of encoder pretraining. **No seeded training run in this project
reproduces off the machine it was run on.**

**The regenerated artifacts are VALID, just not THE artifacts.** A regenerated E0 encoder scores
move-accuracy **0.4346** against the original's **0.4373**, inside the seed spread
**0.4300-0.4373**. So retraining from scratch would plausibly reproduce the **findings** while
reproducing none of the **published numbers**. Those are different claims.

## 2. Re-evaluating a tracked checkpoint IS portable

`.gitignore` tracks `*_head.pt` for a stated reason: it is *"what makes 'never retrain to
re-evaluate' true from a fresh checkout"*. **That reasoning is now measured and it holds.**

EXP-036 depth 3, published checkpoints loaded and re-evaluated on both machines:

| field | this VPS | laptop | agree? |
|---|---|---|---|
| `success_rate` | 0.26666666666666666 | 0.26666666666666666 | **yes, full repr** |
| `mean_steps` | 3.75 | 3.75 | **yes** |
| `optimality` | 0.8 | 0.8 | **yes** |
| `eval_revisit_rate` | 0.4298245614035088 | 0.4298245614035088 | **yes** |
| `greedy_modal_action_frac` | 0.75259259259259**26** | 0.75259259259259**25** | **1 ULP** |

Seed 1 agreed on **every** field including that one; seeds 0 and 2 differ by 1-2 ULP in it alone.
It is a mean over per-episode fractions, so its summation order is the one thing that moves.

==**Every headline metric reproduces exactly.**== That is a real, defensible guarantee.

## 3. What the artifacts situation was, and is

| artifact | before | after |
|---|---|---|
| E0 pretrained encoders (24) | **untracked**, laptop-only, 12 of 24 on the VPS | **tracked**, 2.7 MB |
| EXP-043 depth-6 heads, seeds 14-23 (10) | untracked, laptop-only | **tracked** |
| EXP-047 fine-tuned heads, seeds 14-23 (10) | untracked, laptop-only | **tracked** |
| EXP-047 fine-tuned **encoders**, seeds 14-23 | already tracked | unchanged |

The E0 encoders were excluded on the documented ground that they are "reproducible from
EXP-040's phase 1". **That ground is true on the laptop and false anywhere else**, which section
1 measures, so they are now tracked. **Fifteen experiments load them frozen** (EXP-041 through
EXP-066).

The 20 seed-14-23 heads were EXP-060's dependency chain, generated at its dispatch and never
added while their sibling seeds were tracked - a missed `git add`, not a policy decision.

> **A first reading of this looked worse than it was**, and the correction is worth keeping: the
> EXP-047 **encoders** for those seeds were already tracked. That is the expensive artifact, a
> ~12 h fine-tuning run. Only the cheap heads were missing. The difference between "a
> replication is unreproducible" and "some checkpoints were untidy" is large and the first
> reading was wrong.

## 4. Documentation defects found and fixed

- **EXP-040's `RESULTS.md` claimed its `*_encoder_*.pt` files were tracked. Zero were** - 36 head
  checkpoints were tracked and no encoder weights. Anyone following that file to reproduce would
  have looked for artifacts that were not in the checkout. Corrected with a dated note.
- **`.gitignore`'s reasoning now carries the measurement** rather than the superseded claim.

## 5. The guarantee Phase 4 may make

**Say this:**

> Every published number can be reproduced from a fresh clone by loading the tracked checkpoints
> and re-evaluating. Retraining from a seed reproduces the findings but not the numbers, because
> training is not bit-reproducible across machines.

**Do not say** "seeded runs are byte-identical" without qualification. That is true **within a
machine** - where it has been verified repeatedly and is a genuine correctness check on the
seeding discipline - and **false across machines**.

## 6. Tools, all committed

| tool | what it answers |
|---|---|
| `experiments/040_pretrained_encoder_policy/verify_regeneration.py` | is encoder pretraining reproducible, here vs there |
| `scripts/verify_e2e_reproduction.py` | does retraining a published cell reproduce its checkpoint |
| `scripts/verify_eval_portability.py` | does re-evaluating a tracked checkpoint reproduce its metrics |
| `scripts/wt-switch-branch.ps1` | switch the laptop worktree without losing untracked checkpoints |

**Run the first three on any machine that will host the public release**, before the release
claims anything.

## 7. What this audit did NOT check

- **Whether the findings survive retraining.** Section 1 argues they plausibly would, from one
  encoder's move-accuracy sitting inside the seed spread. That is an argument, not a measurement,
  and measuring it means re-running experiments end to end on a second machine.
- **Any experiment other than EXP-036 and EXP-040 phase 1.** Two probes, chosen because one has
  no encoder dependency and the other is the dependency itself.
- **Non-x86 or GPU hosts.** Both machines here are x86 CPUs.
- **The `dashboard/` JS toolchain**, which has its own lockfile and was not exercised.
