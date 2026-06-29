# EXP-025: Head Capacity Probe (MLP head vs linear head)

Date: 2026-06-29
Status: Approved, ready for implementation planning

## Motivation

EXP-024 established that the v1 grid-world policy **navigates rather than memorizes**
(held-out goals generalize), but is **capped at ~30-50% held-out success**. The
suspected bottleneck is representational: a trainable `nn.Linear` head reads a *frozen*
64-dim sensory concept, so the policy can only express the linearly-decodable slice of
goal-conditioned navigation. Four of five brain regions are frozen spectators and the
memory path is bypassed (ADR-0001, amended).

The curriculum/calendar labels Week 13 "Grid Scaling + Obstacles." Per EXP-024 that is
the wrong lever: a harder task worsens the representational cap rather than addressing it.
The honest next step is to engage the brain's learnable capacity. The cheapest, most
decisive first probe is to swap the readout, not the task.

## Question and verdict logic

Does a nonlinear readout on the frozen sensory concept lift the ~30-50% held-out cap?

- **MLP clears the linear band** -> the **linear head** was the limiter. There is more
  signal in the frozen concept than a linear map can extract. Cheap capacity win.
- **MLP lands inside the linear band** -> the **frozen encoder** is the wall. The signal
  is neither linearly nor shallow-nonlinearly decodable from the frozen concept.
  Escalate to step 2: unfreeze / pre-train the sensory region (gated on this verdict).

"Clears the band" is defined as: the MLP's mean held-out success rate exceeds the linear
head's `mean + spread` across seeds, in at least the sparse regime. The comparison is
**paired**: at a fixed seed both heads see the identical train/held-out goal split.

## Scope

In scope:
- A pluggable policy head (linear default unchanged, plus a one-hidden-layer MLP).
- A reusable `head_type` option threaded through the existing generalization harness.
- An EXP-025 runner that sweeps both heads across both reward regimes and multiple seeds.
- An aggregator that produces a paired evidence table and a written verdict.

Out of scope (explicitly parked):
- **No encoder unfreezing.** That is the next step, gated on this probe's verdict.
- **No dashboard work.** An MLP head still runs a frozen brain, so regions stay
  spectators; the honest-dashboard "real specialization" payoff belongs to the unfreeze
  step, not here.
- **No grid scaling / obstacles.** The curriculum collision set aside above.
- **No width/depth sweep** in the baseline probe. Escalate to a small width sweep only
  if the width-128 MLP lands ambiguously on the linear band.

## Statistical rigor

The cap (~30-50%) is noisy, so a single-seed comparison could move ~15 points on luck
alone. The verdict therefore requires a multi-seed, paired comparison:

- Seeds: `{0, 1, 2, 3, 4}` (5 seeds).
- For each `(head_type, regime, seed)` run `run_generalization` and record train and
  held-out success, mean steps, optimality.
- Report mean +/- spread per `(head_type, regime)`. Declare a lift only if the MLP band
  clears the linear band per the rule above.

## MLP head shape

- Linear (baseline, unchanged): `nn.Linear(64 -> 4)`.
- MLP (probe): `nn.Sequential(nn.Linear(64 -> 128), nn.ReLU(), nn.Linear(128 -> 4))`.

Rationale: the minimal nonlinear readout. ~9k params, clearly distinct from linear, and
small enough not to trivially memorize 24 train goals. If a one-hidden-layer MLP does not
lift the cap, deeper readouts will not either, so this width is a sufficient probe.

## Components and the single library change

The only library change is making the head pluggable. Concept/action dims are read from
the brain (`brain.content == 64`, `brain.n_actions == 4`), not hardcoded.

`src/neuromorphic/training/reinforce.py`:
- `make_policy_head(brain, head_type="linear", hidden=128) -> nn.Module`
  - `"linear"`: returns today's `nn.Linear(brain.content, brain.n_actions)`. This is the
    byte-identical default, preserving EXP-023/EXP-024 behavior.
  - `"mlp"`: returns `nn.Sequential(Linear(brain.content, hidden), ReLU,
    Linear(hidden, brain.n_actions))`.
- Widen the `head: nn.Linear` type hints to `nn.Module` where they appear
  (`action_distribution`, `greedy_action`, `policy_parameters`, `train_episode`). No
  behavioral change: every call site uses only `head(concept_rate)` and
  `head.parameters()`, both satisfied by any `nn.Module`.

`src/neuromorphic/training/generalization.py`:
- Add `head_type: str = "linear"` and `hidden: int = 128` to `GenConfig`.
- Pass them through to `make_policy_head` in `run_generalization`.
- `tag` already namespaces output filenames; the runner folds `head_type` into the tag
  (e.g. `shaped_linear`, `sparse_mlp`), so EXP-024 output names are never collided with.

No new training loop, no change to the REINFORCE update, no encoder changes.

## Experiment runner and data flow

`experiments/025_head_capacity/run.py`:
- Sweep: `{linear, mlp} x {shaped, sparse} x seeds {0..4}` = 20 runs.
- Each run calls `run_generalization` with `out_dir =
  experiments/025_head_capacity/outputs/` and `tag = f"{regime}_{head_type}_seed{seed}"`,
  writing per-run `..._<tag>_metrics.csv` and `..._<tag>_summary.json`. Folding the seed
  into the tag means no run overwrites another, so all 20 per-run artifacts persist for
  debugging. The aggregated `025_summary.json` is the artifact of record.
- Aggregator: collects the 20 summaries' train/held-out success (plus steps, optimality)
  into `experiments/025_head_capacity/outputs/025_summary.json` and a markdown evidence
  table grouped by `(head_type, regime)` with mean +/- spread.

Determinism and pairing: every seed flows through `torch.manual_seed`, the
`GridWorldEnv(goal_seed=seed)`, the harness `torch.Generator(seed)`, and
`split_goals(..., seed)`. At a fixed seed, linear and mlp therefore train and evaluate on
the identical goal partition -> the comparison is paired.

## Testing

Extend `tests/training/` (keep the existing 202-test suite green):
- `make_policy_head("linear")` returns an `nn.Linear` with shape `(n_actions, content)`
  (unchanged from today).
- `make_policy_head("mlp")` returns a 2-layer module whose forward on a `[content]` input
  yields `[n_actions]`; `.parameters()` is non-empty and optimizer-steppable.
- A 1-episode `train_episode` with the MLP head runs end to end and changes head params.
- Determinism: same `(seed, head_type)` -> identical `run_generalization` summary (guards
  the paired-comparison claim).

## Deliverable

- Raw per-run CSV/JSON plus the aggregated `025_summary.json` and evidence table.
- `experiments/025_head_capacity/FINDINGS.md`: the verdict (head-limited vs
  encoder-limited), the paired evidence table (held-out success, linear band vs MLP band,
  both regimes), and the recommended Week-13 next step.
- A one-paragraph amendment to `docs/adr/0001-multi-region-training-strategy.md` **iff**
  the verdict is decisive.

## Risks and mitigations

- **MLP overfits 24 train goals** (high train, flat held-out): this is itself an informative
  signal (capacity exists but does not generalize from the frozen code); report it, do not
  hide it. Width 128 on 64-dim input is modest, mitigating gross memorization.
- **Result lands on the band** (ambiguous): the planned escalation is a small width sweep
  `{32, 64, 128, 256}` in the sparse regime before concluding; flagged, not silent.
- **Accidental EXP-024 regression**: the `"linear"` default must stay byte-identical;
  the determinism test and unchanged output for `head_type="linear"` guard this.
