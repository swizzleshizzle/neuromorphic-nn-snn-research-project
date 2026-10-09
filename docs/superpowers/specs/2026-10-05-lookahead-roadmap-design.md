# Look-ahead roadmap, and EXP-070: search on the networks we already have

**Written 2026-10-05, week 27.** Design agreed with Michael in session `week27`. This spec does two
things: it fixes the **direction** of the project from here (section 1 and section 4), and it
specifies the **first experiment** in full (sections 2 and 3). Stages after EXP-070 are roadmap,
not specification; each gets its own spec when its turn comes.

---

## 1. Why the direction changed

**The goal is a spiking network that solves a cube, not a release.** Michael stated on 2026-10-05
that Phase 4's "public release" was an artifact of the original media plan and is dropped. The
report (`docs/writeup/report.md`) continues as documentation, on a parallel track (section 5).

**Where the solver stands.** Held-out success with the current recipe:

| depth | success | source |
|---|---|---|
| 4 | 0.5351 | EXP-042 (capped arm) |
| 5 | 0.3412 | EXP-043 |
| 6 | ~0.32 | EXP-046 |
| 7 | 0.2004 | EXP-053 arm B |
| 8 | 0.0783 | EXP-062 |
| 9 | 0.0163 (at the floor) | EXP-062 |

Only **159,120 of 3,674,160 states (4.3%)** lie within 8 moves of solved. The modal distance of a
random 2x2 is 11 and the maximum is 14 (6-move quarter-turn set). Weighting the table above by the
shell sizes, the current solver handles a uniformly random cube **under 1% of the time, about
0.7%**, even counting every state within 3 moves of solved as solved and every state beyond depth
9 as unsolved.

> **CORRECTION 2026-10-07.** This table first gave depth 4 as ~0.84 and depth 5 as ~0.69, citing
> EXP-047. Those were EXP-047's **probe accuracies** after fine-tuning (0.8396 and 0.6850, its
> probe-depth table), not policy success; EXP-047 measured success at depth 6 only. The rows now
> cite the committed success rates. The random-cube figure was re-weighted with them: 0.70%,
> against 0.72% from the old rows, so its conclusion stands but "well under" overstated it. Shell
> sizes: depths 0 to 6 from `road-to-a-solved-cube`, depths 8 and 9 from the EXP-062 spec, depth 7
> as the remainder of the 159,120 states within 8 moves. Nothing in EXP-070 or EXP-071 used these
> two rows.

**Why more of the same will not get there.** The budget law (EXP-044/045/046, held out of sample
by EXP-062) prices success at about 0.22 per decade of training spend. Depth 11 costs about 33 days
of compute per seed. The law describes the **current reactive policy**: one forward pass, one move,
no look-ahead. A policy that must be right roughly 11 times in a row compounds its per-move error.
Nothing in the law constrains a policy that searches.

**What is on the policy path.** The sensory region (192 neurons) feeding a 390-parameter linear
head. 318 of 510 neurons are off-path. Memory as a recall input to that head hurt (EXP-059/061/063);
the neuromodulatory bus bought nothing (EXP-053 arm G); the trained five-region motor pathway
collapses (EXP-064/066).

**What this design is aiming at.** Split the skill into a **general procedure** (imagine moves,
compare where they lead, choose) and a **learned judgement** (how close does this position look).
The procedure transfers to any cube; the judgement is learned from experience. This is the shape
of the strongest published cube solvers (DeepCubeA: a learned cost-to-go plus search) and the
closest realistic reading of "learn the process, not the cube".

## 2. EXP-070: does look-ahead on the existing networks move the frontier?

### 2.1 Question

If the networks already trained by the current recipe are allowed to imagine 1 to 3 moves ahead
with the true simulator, do they solve deeper cubes than when they react one move at a time? And
does the **network's judgement** contribute, or does blind search of the same size do as well?

### 2.2 What is loaded (nothing is trained)

| depth | head | encoder | seeds |
|---|---|---|---|
| 7 | `experiments/053_neuromod_stage3/outputs/exp053_critic_d7_regionalized_d7_s*_sig0.0_head.pt` | E1, frozen: `experiments/047_encoder_finetuning/outputs/exp047_ft_d6_lr0.0001_regionalized_d6_s*_sig0.0_encoder.pt` | the 12 EXP-053 seeds |
| 8 | `experiments/062_depth_frontier/outputs/exp062_frontier_d8_regionalized_d8_s*_sig0.0_head.pt` | E1, as above | the 12 EXP-062 seeds |
| 9 | `experiments/062_depth_frontier/outputs/exp062_frontier_d9_regionalized_d9_s*_sig0.0_head.pt` | E1, as above | the 12 EXP-062 seeds |

The held-out set per seed is the one `split_shell` produced for the original run (capped at 200).
Re-evaluating tracked checkpoints is the one operation the reproducibility audit showed is portable
across machines, and every comparison below is **paired on the same checkpoint**, so the seed
effect (sd 0.0906, `docs/seed-effect.md`) contributes exactly zero noise to any contrast.

Pre-flight: confirm the seed lists and that the E1 encoder for each seed is the one the head was
trained against (by config record, not by filename pattern).

### 2.3 How a move is chosen with look-ahead `k`

1. From the current state, enumerate every move sequence of length `k` with the pure simulator
   `apply_move`: `n**k` leaves for `n = env.action_space.n` (never a literal 6).
2. If any prefix of any sequence reaches solved, play the first move of the shortest such
   sequence (ties broken by lowest action index, so the rule is deterministic).
3. Otherwise score each sequence and play the first move of the best one. Re-plan from the new
   state on every real move.

**Scorers.**

- **P (policy-guided):** the sum along the sequence of the head's log-probability of each move at
  the state where it would be taken. All states at one tree level are encoded in ONE batched
  `SensoryCortex` call. The sensory region resets its LIF state on every forward
  (`SensoryCortex.forward` calls `self.reset()`), so imagined states cannot corrupt the real one.
  Pre-flight check: assert that the greedy action on the real state is identical before and after
  scoring a tree.
- **E (endgame only):** no scoring. Play the greedy move exactly as G would, but with the same
  `k`-step goal test in step 2. This isolates what the goal test buys for free, so that P can be
  measured against reflex-plus-endgame rather than against chance.
- **R (random-guided):** each sequence gets an independent uniform score from a generator seeded per
  (seed, state, step). Same goal test, same `k`, same tree. A reported floor, not a primary
  contrast: at depths 8 and 9 a random walk almost never comes within `k` of solved, so P > R is
  near-certain before any cell runs and would only restate that the network beats chance.

**The encoding is stochastic, and that has to be handled.** `encode_cube` produces Poisson spikes
from the `generator` that `evaluate_states` threads through `greedy_action`. If imagined states
drew from that stream, every real-move encoding after the first tree would differ from G's, and
P versus E would be confounded by encoding noise. So **imagined states draw from a dedicated
generator** seeded per (seed, state, real step), and **real states draw from the evaluation stream
exactly as in G**. Consequence: while P, E and G play the same real moves, they see bit-identical
real encodings, and they diverge only where a decision differs.

**Budgets.** The real-move budget stays `max_steps_for(d) = 2d+3`, exactly as published, so
results compare to the published tables. Imagined moves do not count against it. The **node
budget** (states the tree touches per real move) is identical between P and R at the same `k` by
construction. E touches the same tree for the goal test but scores none of it.

**No 2x2-only shortcuts inside the procedure.** The BFS distance table is used only to construct
the held-out shells and to report results. It is never read by any scorer or by the goal test.

### 2.4 Arms

| arm | k | scorer | role |
|---|---|---|---|
| **G** | 0 | greedy argmax, as published | **validity gate** |
| **E1, E2, E3** | 1, 2, 3 | greedy move, goal test only | what the endgame check buys |
| **P2, P3** | 2, 3 | policy-guided | under test |
| **R1, R2, R3** | 1, 2, 3 | random-guided | reported floor |

9 arms x 3 depths x 12 seeds = 324 cells. No retraining, so cells are evaluations.

> **AMENDMENT 2026-10-06, before any cell has run: P1 is dropped.** At `k = 1` the P score of a
> sequence is the root's log-probability of its one move, so P1 picks the greedy move whenever the
> goal test does not fire: **P1 is E1 by construction**, and a P1-versus-E1 contrast is 0 on every
> cell before it runs. Found while writing the implementation plan. The primary cell (depth 8,
> k = 3) is unaffected.

### 2.5 Pre-registered claims

**Gate 0 (validity), two parts.**

- **(a) Determinism.** Re-running any cell reproduces its record byte for byte. Always required.
- **(b) Match to the published numbers.** Compared on **integer solved counts per cell**, not
  float repr (the audit itself found a derived mean 1 ULP off). This is a real check, not a
  formality: the portability audit covered EXP-036 and EXP-040 only, and it re-evaluated with a
  fresh generator seeded at the train seed, whereas EXP-053/062 evaluated with whatever stream
  state training left behind. The **pre-flight** (section 2.6) establishes which of two forms
  applies, and the spec is amended with the result, dated, **before** the full launch:
  - if a fresh-generator re-evaluation reproduces the published counts, Gate 0(b) is exact match
    on all 36 (depth, seed) cells;
  - if it does not, the published counts are not reproducible by re-evaluation for these
    experiments, G's re-evaluated numbers become the reference, and Gate 0(b) is replaced by
    "G's mean is within the binomial interval of the published mean at each depth". The
    difference is reported in `RESULTS.md` either way.

If Gate 0 fails, the harness is wrong and **no claim below may be read**.

**Gate 1 (resolution).** The primary contrast is P against E. It can only come out either way if
the pair is not stuck on a shared floor or ceiling (EXP-064: two arms at 0.0000 give a contrast of
0 by construction). For each (depth, k): if both `E_k` and `P_k` mean success are below **0.02**,
or both above **0.98**, that cell's verdict is **UNRESOLVED**, returned by the aggregator as a
verdict of its own (the EXP-068 rule). E needs one network call per real move and R needs none,
so **both are measured during the pre-flight on the real held-out shells**, before any P cell
exists. The pre-flight result for each (depth, k) is written into this spec, dated, before launch.

**Claim 1 (primary): scoring the look-ahead beats reflex plus endgame.** One primary cell:
**depth 8, k = 3.** mean(P3 - E3) > 0 by an exact paired sign-flip permutation test over all
`2**12` flips, one-sided, p < 0.05. Verdict: CONFIRMED, REFUTED (point estimate <= 0), NOT
SIGNIFICANT, or UNRESOLVED (Gate 1). The other (depth, k) cells of the same contrast are
**secondary** (k = 2 at depths 8 and 9, k = 3 at depth 9): reported with their p-values, read as a pattern, never as independent
confirmations. One honest expectation, recorded in advance: summed log-probability usually picks
the greedy first move, so P and E may differ little. The contrast exists to show exactly that if
it is true.

**Claim 2 (practical): look-ahead moves the frontier.** At depth 9, P3 mean success >= 0.10,
against G. Reported as a practical gain, never as evidence about the network on its own: the
decomposition is G -> E (what the goal test gives) -> P (what scoring adds). Read it only with
Claim 1.

**Claim 3 (mechanism): look-ahead loops.** Report `revisit_rate` for every arm. No threshold is
pre-registered; this measurement exists to size the job a memory component would have later.

R is reported at every (depth, k) as the floor for "search without a network". Depth 7 is
reported for all arms as the bridge to EXP-053, without its own claim.

### 2.6 Cost, and what is measured before launch

Per real move, a `k = 3` tree is 6 + 36 + 216 = 258 states, encoded as three batched calls.
Estimate: minutes per (depth, seed, arm) on the laptop, not hours. Runs bank one JSON record per
cell, so a Windows Update restart costs at most one in-flight wave.

**Pre-flight, in order, before the full launch. Each result is written into this spec, dated:**

1. **Gate 0(b) form:** re-evaluate G for all 36 (depth, seed) cells and compare solved counts
   with the published records. Decides which form of Gate 0(b) applies.
2. **Gate 1 calibration:** run E1 to E3 and R1 to R3 at all depths. Records each (depth, k)
   cell's E level, so a reader can see in advance which cells can resolve.
3. **Calibration wave:** one seed, P2 and P3, depth 8. Its wall-clock replaces the estimate above.

These pre-flight numbers are for E, R and G only. No P number exists until the gates are fixed.

> **PRE-LAUNCH AMENDMENT 2026-10-06, written after pre-flight steps 1 and 2 and BEFORE any P
> number exists.** Laptop (SwizzlesDuo), main at `0af516b`, 252 records.
>
> **Gate 0(b) form: WILSON.** A fresh-generator re-evaluation does NOT reproduce the published
> per-cell counts: **27 of 36 cells differ** (largest: depth 8 seed 3, published 22/200,
> re-evaluated 8/200). Cause: evaluation is stochastic (Poisson spike encoding on the eval
> generator) and the published records evaluated on the stream state training left behind.
> Pooled per depth, each re-evaluated mean lies inside the Wilson 95% interval of the published
> one:
>
> | depth | published | interval | re-evaluated G |
> |---|---|---|---|
> | 7 | 481/2400 = 0.2004 | [0.1849, 0.2169] | 496/2400 = 0.2067 |
> | 8 | 188/2400 = 0.0783 | [0.0682, 0.0898] | 165/2400 = 0.0688 |
> | 9 | 39/2400 = 0.0163 | [0.0119, 0.0221] | 41/2400 = 0.0171 |
>
> Depth 8 clears by 0.0006. That is not a near-miss to repair: the pooled Wilson check is weak by
> construction. **The real evidence that the harness is right is the unit test pinning G to
> `evaluate_states` at full float repr** (`test_mode_g_reproduces_evaluate_states_exactly`).
> Consequence for every claim: **G re-evaluated, not the published table, is the reference.**
>
> **Gate 0(a): PASS.** G0, E3 and R3 at depth 8 seed 0 re-run into a separate directory are
> identical in every field except `wall_s`, which is a wall clock and is excluded by definition.
>
> **Gate 1 calibration (E and R levels, mean of 12 seeds):**
>
> | depth | G0 | E1 | E2 | E3 | R1 | R2 | R3 |
> |---|---|---|---|---|---|---|---|
> | 7 | 0.2067 | 0.2179 | 0.2150 | 0.2321 | 0.0000 | 0.0008 | 0.0025 |
> | 8 | 0.0688 | 0.0708 | 0.0750 | 0.0838 | 0.0000 | 0.0004 | 0.0013 |
> | 9 | 0.0171 | 0.0179 | 0.0200 | 0.0200 | 0.0000 | 0.0004 | 0.0008 |
>
> The primary cell's E3 (0.0838) is clear of the 0.02 floor, so Claim 1 can resolve. Depth 9's
> E2 and E3 sit exactly at 0.0200: those secondary cells resolve only if P lifts them, and
> otherwise come back UNRESOLVED, which is the verdict the gate exists to return. E is not
> monotone in k at depth 7 (E1 0.2179 > E2 0.2150): once E's moves diverge from G's, the two see
> different encoding draws, so per-arm levels carry that noise. R confirms random search is near
> zero at every depth.
>
> **What the primary contrast can detect (the EXP-067 rule: state a threshold's power in the regime
> it runs in).** The G0-versus-published differences are pure encoding noise on identical
> checkpoints and states. At depth 8 their per-seed sd is **0.0244**, so the se of a mean over 12
> seeds is **0.0071**, and a one-sided test at alpha 0.05 has about 80% power at a true effect of
> **0.0175** (normal approximation, (1.645 + 0.84) x se). This is an UPPER bound on the noise in
> P3 - E3, because P and E share the real stream while their moves agree. **So Claim 1 can see a
> look-ahead benefit of roughly 0.02 success or more at depth 8, and will likely miss a smaller
> one.** A NOT SIGNIFICANT verdict is therefore "below about 0.02", not "zero".
>
> **Cost.** A G/E/R cell takes about 250 s at 3 workers and 760 to 1640 s at 20 (contention, not
> work). The calibration wave (P2 and P3, depth 8, seed 0) runs next, for wall-clock only, and its
> success counts are part of the full P dataset, read only by the aggregator.

### 2.7 Outputs

`experiments/070_lookahead_existing/`: `run.py`, `aggregate.py` (verdict function encodes Gate 0,
Gate 1 and all four Claim 1 verdicts), per-cell records, `RESULTS.md` with provenance.

## 3. Tests EXP-070 needs before any cell runs

Each must fail against the bug it names (mutation-check them):

- With `k = 0` the harness takes the exact code path of `evaluate_states`' greedy rollout
  (Gate 0 depends on it).
- The goal test finds a solve at exactly depth `k` and not `k + 1` (catches an off-by-one in tree
  depth).
- Scoring a tree leaves the real state's encoding and greedy action unchanged (catches state leaking
  between imagined and real steps).
- Scoring a tree does not advance the evaluation generator: E and G produce bit-identical
  trajectories on a state where the goal test never fires (catches imagined states drawing from
  the real stream).
- The batched concept for a state equals the concept `evaluate_states` feeds the head for that
  state under the same generator draw (catches the batched path and the evaluation path
  disagreeing, which a leak test alone would not see).
- E's chosen move equals G's whenever the goal test does not fire (catches E silently scoring).
- R's scores are independent of the checkpoint (catches the scorer accidentally reading the head).
- Neither scorer nor goal test can reach the BFS provider (catches an oracle leak; e.g. run the
  search with the provider replaced by an object that raises on access).
- Action-space width comes from the environment (run the tree builder on a stub env with 12
  actions).

## 4. The roadmap after EXP-070

Each stage has a decision point; the next stage starts only on its answer.

| stage | what | decision before moving on |
|---|---|---|
| **1. EXP-070** | look-ahead on existing networks (section 2) | Does P beat E (reflex plus endgame) at depth 8? If P adds nothing over E, the policy head is not a usable judge and stage 2 decides whether the critic is. |
| **2** | retrain at one depth with the critic saved; test whether it ranks the `n` children of a state correctly (BFS distance as the yardstick only); then critic-guided search | Must beat stage 1's P at the same `k`. If the critic cannot rank siblings, go straight to 3. **Pre-flight:** `*_critic.pt` files are written by every critic run (`critic_filename`) but none are tracked; check the laptop worktree for EXP-053/062 critics before retraining. |
| **3** | train the judge from its own look-ahead: the target for a state is one move plus the best child's value (approximate value iteration, the DeepCubeA shape). No distance table, no solutions. | The target is uniformly random scrambles at depth 11 and beyond. |
| **Reshape** | sensory plus decision only, wider | A width comparison on its own, never bundled with a search change. |
| **Memory** | a visited/explored store inside the search, measured by `revisit_rate` | Only once a search loop exists. Spiking implementation is optional and will be stated as such. |
| **B** | the network predicts the effect of a move (a learned world model) in place of `apply_move`, building on EXP-040's inverse-dynamics pretraining | After stage 3, when there is a working judge to plug it into. |
| **3x3** | same recipe, larger environment (54 facelets, 12 or 18 moves, no enumerable BFS table) | Gated on random 2x2 scrambles being solved. A 3x3's corners are a 2x2, so a 2x2 skill can become one component. |

> **OUTCOMES, dated.**
> - **2026-10-06, stage 1 (EXP-070):** the policy head as judge adds below about 0.01 at depth 8
>   (NOT SIGNIFICANT). The reflex loops on about half its moves.
> - **2026-10-06, stage 2 (EXP-071, depth 7):** the critic ranks above chance (R1, R3 pass on 12/12
>   seeds) but critic-guided search is REFUTED (C3 0.0729 vs E3 0.2321; C3 - P3 -0.1725). By this
>   table's rule the roadmap moves to stage 3. In the same experiment the no-revisit rule was
>   CONFIRMED (G0V +0.0279, p 0.0115), and the best measured arm is P3V at **0.3217** (secondary,
>   +0.0762 over P3). The memory row below has therefore been tested in its simplest, non-spiking
>   form and works.
> - **2026-10-07, EXP-072 (depths 8 and 9):** P3V holds where the reflex fails. Both primaries
>   CONFIRMED: depth 8 P3V 0.1496 vs P3 0.0854 (+0.0642, p 0.0005); depth 9 P3V 0.0629 vs G0
>   0.0171 (+0.0458, p 0.0002, off the floor). Still search scaffolding on frozen networks; a
>   random 2x2 (usually 11 moves out) remains out of reach.
> - **2026-10-09, stage 3 (EXP-074, superseding EXP-073):** a judge trained by value iteration from
>   its own look-ahead, never from answers, reading all 192 sensory neurons. Both primaries
>   CONFIRMED at depth 9, on 12 seeds and on the 10-seed sensitivity line: J3V-W 0.1933 vs P3V
>   0.0629 (+0.1304, p 0.0002, 12/12 seeds) and vs the 64-unit judge J3V-A 0.1292 (+0.0642,
>   p 0.0010). Depth 7 J3V-W 0.7908. Exploratory depth 11, the typical random scramble: J3V-W
>   0.1013 vs P3V 0.0308. Readout width was a measured bottleneck. States at distance 11, the
>   typical random scramble, are solved about one time in ten (exploratory, no claim).

**Standing rules for every stage:** the random-guided control; a matched node budget; at least 12
paired seeds; no distance table and no action-count literal inside the system; laptop runs bank
per cell.

**What this roadmap does not claim.** It does not claim that search makes the network
neuromorphic in its learning rule (training remains surrogate-gradient backprop), nor that the
learned weights will transfer from 2x2 to 3x3. What is meant to transfer is the recipe.

## 5. Parallel track: the report

A separate session, in its own git worktree, fills `docs/writeup/report.md` section by section
from committed `RESULTS.md` files only. It drops "public release" from the report's framing and
adds a short section 9, "The road from here", pointing at this spec. It does not edit `src/`,
`experiments/`, or `CLAUDE.md`, so it cannot collide with the build track.
