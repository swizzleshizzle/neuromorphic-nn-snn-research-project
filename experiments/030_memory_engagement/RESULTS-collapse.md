# EXP-030 follow-up: was the memory null measured on a collapsed policy?

> **Provenance:** 144 records (4 arms x depths 1-3 x seeds 0-11), 600 episodes per run, 2x2 cube,
> identical configuration to EXP-030 except for two added instrument fields. Run 2026-08-01 on the
> laptop `SwizzlesDuo` (Intel Core Ultra 9 185H, 16 cores / 22 threads) over SSH from the VPS with
> `--workers 16`, wall clock 3h14m (21:22:56 to 00:37:07 local). Records in
> `experiments/030_memory_engagement/outputs_instrumented/` (gitignored). Instrument added on branch
> `week17-collapse-instrument`, commit `0017053`. Regenerate with the command at the bottom.

## The question

The 2026-07-30 handoff recorded that while capturing a depth-3 trace for the dashboard, **the policy
played `F'` nine times in a row**. `F'` has order 4, so that is an absorbing 4-cycle. If the policy
has collapsed to a constant action then it is not reading its input at all, and no change to the
feature vector could alter behaviour, which would make EXP-030's memory null **a fact about a
degenerate policy rather than about memory**.

That observation was n = 1, from the recorder's own training run. This tests it at n = 12 per arm.

## Instrument

Two readouts, because the `F'` observation was made under the GREEDY policy while `entropy_beta=0.0`
is a TRAINING setting, and those can fail independently:

| field | meaning | collapse | uniform reference |
|---|---|---|---|
| `greedy_modal_action_frac` | fraction of a rollout spent on its single most-common action, averaged per episode | 1.000 | 0.354 at a 9-step budget, 0.429 at 5 |
| `mean_train_entropy` | per-episode policy entropy during training | -> 0 | log 6 = 1.792 |

The uniform references were measured over 20,000 simulated rollouts before the implementation
existed; the finished code then measured 0.357 on the real random-policy path.

## Neutrality check: the instrument did not perturb the experiment

The instrument consumes no randomness, so re-running the identical configuration must reproduce every
pre-existing field exactly. Verified record by record with `scripts/verify_instrument_neutrality.py`,
comparing all 144 new records against EXP-030's originals with only the two new fields and
`config.out_dir` excluded:

```
compared: 144    identical: 144    differing: 0
PASS: every pre-existing field reproduced exactly.
```

**Nothing below can be attributed to the code change.**

## The confound, addressed before the result

`greedy_modal_action_frac` is inflated by early termination: an episode solved in one move has a modal
fraction of 1.0 by construction. So the metric is only meaningful where episodes actually run the
budget.

| depth | mean success | fraction of episodes hitting the full budget | corr(modal_frac, success_rate) |
|---|---|---|---|
| 1 | 0.826 | 0.174 | **+0.254** |
| 2 | 0.299 | 0.701 | -0.920 |
| 3 | 0.010 | **0.990** | -0.547 |

**Depth 1 is uninformative and its numbers are discarded**: 83% of episodes end early, and the weak
positive correlation is the signature of exactly the artifact described. At depths 2 and 3 the
correlation is *negative*, so the metric is not being inflated there; collapsed policies simply
succeed less. **At depth 3, 99% of episodes run all 9 steps, so the measurement is clean.**

## Result: the policy is collapsed

`greedy_modal_action_frac`, mean over 12 seeds (standard deviation):

| depth | concept | memory | shuffled | amnesic |
|---|---|---|---|---|
| 1 *(discarded, see above)* | 0.997 (0.010) | 0.986 (0.033) | 0.978 (0.046) | 0.983 (0.022) |
| 2 | 0.825 (0.119) | 0.853 (0.124) | 0.882 (0.122) | 0.858 (0.118) |
| **3** | **0.932** (0.124) | **0.959** (0.111) | **0.998** (0.007) | **0.994** (0.020) |

Seeds at or above 0.95, out of 12:

| depth | concept | memory | shuffled | amnesic |
|---|---|---|---|---|
| 2 | 2 | 3 | 5 | 3 |
| **3** | **9** | **10** | **12** | **11** |

Depth-3 concept arm, per seed, sorted:

```
0.684  0.719  0.789  0.996  0.996  1.000  1.000  1.000  1.000  1.000  1.000  1.000
```

**Seven of twelve seeds are at exactly 1.000**: one action, every step, every episode, for the entire
9-step budget. Three more are within 0.004 of it. Against a uniform floor of 0.354.

## Independent corroboration from EXP-030's own numbers

EXP-030 reported that at depth 3, **7 of 12 seeds sat at exactly 0.667 greedy revisit rate**. That
number is derivable from constant-action play: a quarter turn on a 2x2 cube has order 4, so a 9-step
rollout visits 10 states of which 4 are unique, giving `(10 - 4) / 9 = 0.667` exactly.

**The same seven seeds, arrived at by two instruments that share no code.** The revisit rate was
already reporting the collapse in EXP-030; nobody had the second measurement needed to read it that way.

## Training entropy agrees, and has no episode-length confound

Depth 3, mean over 12 seeds, against the log 6 = 1.792 ceiling:

| arm | entropy | % of maximum |
|---|---|---|
| concept | 0.541 | 30% |
| memory | 0.322 | 18% |
| shuffled | 0.307 | 17% |
| amnesic | 0.430 | 24% |

Entropy is averaged per step, so early termination does not inflate it. It confirms the greedy finding
from a direction the confound cannot reach.

## What this does and does not establish

**Established.** At depth 3 the trained policy is effectively constant-action in 9 to 12 of 12 seeds
depending on arm. EXP-030's depth-3 results therefore say nothing about memory: a policy that ignores
its input cannot respond to a change in its input. This is consistent with `CubeConfig`'s defaults
(`entropy_beta=0.0`, `normalize_advantages=False`), the same collapse EXP-025 fixed once.

**Depth 2 is partially affected, and that matters more.** Modal fraction is 0.825 to 0.882 with 2 to 5
seeds fully collapsed. **Depth 2 is where EXP-030's headline +10.8 point primary effect lived**, so
that comparison rests on a policy that is degenerate in a minority of seeds and healthy in the rest.
It is weakened, not voided.

**Not established.** This says nothing about whether memory would help a healthy policy. It removes
the depth-3 evidence from the ledger rather than reversing it. EXP-030's central claim, that the
`memory` vs `memory_amnesic` comparison is a clean null (+1.2 points, p = 0.91 at depth 2), is
untouched by this at depth 2 and unsupported at depth 3.

**What it does not excuse.** The mechanism null stands on its own at depth 2: cycles were abundant and
the memory arm revisited slightly more than concept at every depth.

## Lead for the next experiment

Re-run engagement with `entropy_beta > 0` and `normalize_advantages=True` before drawing any further
conclusion about memory, and gate on `greedy_modal_action_frac` at depth 3 falling well below 0.95
before the memory arms are allowed to launch. That gate is the analogue of EXP-030's revisit gate: it
should be pre-registered as a stop condition, because a memory result measured on a collapsed policy is
not interpretable no matter how clean its statistics look.

## Regenerate

```powershell
.venv\Scripts\python.exe -u experiments\030_memory_engagement\run.py `
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 16 --skip-gate `
    --out-dir experiments\030_memory_engagement\outputs_instrumented
.venv\bin\python scripts\verify_instrument_neutrality.py `
    experiments\030_memory_engagement\outputs experiments\030_memory_engagement\outputs_instrumented
```

`--skip-gate` is correct for this re-run specifically: the revisit gate was already read and passed in
EXP-030 (greedy 0.089 / 0.327 / 0.604). It instruments a decided configuration rather than re-deciding it.
