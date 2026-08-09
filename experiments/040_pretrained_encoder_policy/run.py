# experiments/040_pretrained_encoder_policy/run.py
"""EXP-040 driver: does a raised representational ceiling become a better policy?

Vault Stage 2, second increment. EXP-039 built the encoder; this asks whether REINFORCE can
use it.

EXP-039 measured what inverse-model pretraining does to the linear probe:

    depth        frozen   pretrained   facelets (the linear ceiling ON the observation)
    4            0.447      0.786        0.742
    5            0.406      0.660        0.618
    6            0.344      0.575        0.488          all p 0.0005, W-L 12-0

The pretrained encoder beats the raw-facelet ceiling at 4, 5 and 6 - which width provably
cannot do (concept@512 reached 0.638 at depth 4).

BUT A PROBE IS NOT A POLICY. EXP-033 Finding 2, measured in this repo: at depth 3 an oracle
probe on the frozen concept supported 0.48 success while the actual REINFORCE policy managed
0.22. Less than half of what the representation supported was extracted. So whether the raised
ceiling converts is genuinely open, and it is the whole question here.

It also matters more than its size suggests, because EXP-038 closed the other side of the
ledger: trainer stabilizers, curriculum weighting, width, volume and starvation are all
refuted. THE ENCODER IS THE ONLY LIVE LEVER LEFT.

EXACTLY ONE VARIABLE CHANGES vs EXP-036. The pretrained encoder is FROZEN during RL, so the
trainable parameter count stays at the same 390 (`Linear(64 -> 6)`) as every cube experiment
since EXP-029. The only difference is which weights the frozen encoder holds. Fine-tuning
end-to-end would confound "a better representation" with "the encoder kept learning during
RL", and those need separating before they are combined.

PRE-REGISTERED CONTRACT, committed before any number exists. Full version and the reasoning
behind each threshold: docs/superpowers/specs/2026-08-09-exp040-pretrained-encoder-policy-design.md

  All comparisons PAIRED per seed against EXP-036, same twelve seeds, same machine, exact
  permutation over all 2**12 = 4096 sign flips.

  1. PRIMARY, DEPTH 4. CONFIRMED needs >= +0.05 at p <= 0.05 against EXP-036's 0.1591.
     Depth 4 is the POWERED arm - the only depth where the policy currently works, so the only
     one where an improvement is measurable. Depths 5 and 6 are resolution-bound at 1/200 per
     seed, which EXP-038 just showed makes a null uninterpretable. The bar is 0.68 sd of
     EXP-036's measured spread and deliberately harder than EXP-037's 0.03, because EXP-039's
     probe effect was enormous and a marginal policy gain is not what it predicts.

  2. DOES THE BREAK POINT MOVE? EXP-036's rule for "working" is >= 2x floor AND >= 0.10
     absolute; floors are 0.0000 (d5) and 0.0008 (d6), so the bar is 0.10. Depth 5 reaching
     0.10 would be the most consequential cube result to date - it has been broken since
     EXP-036 and nothing has moved it.

  3. MECHANISM. Report modal fraction and entropy against EXP-036's 0.685/0.779/0.975. A gain
     arriving WITH modal fraction falling is more trustworthy than one that does not; EXP-038
     showed collapse is a symptom, so reduced collapse should FOLLOW a gain rather than cause
     it. Read modal WITH entropy - EXP-035, EXP-037 and EXP-038 each produced a different
     relationship between the two.

  4. THE NULL IS PRE-COMMITTED AND INFORMATIVE. If Claim 1 refutes while EXP-039's probe
     result stands, the finding is that the representation was never the binding constraint on
     the policy - the readout or the learning signal is. That is EXP-033 Finding 2 writ large
     and the strongest evidence yet for Stage 3 (a value function on the idle `neuromod`
     pathway). Report it as a positive redirection, not a disappointing null.

THE CONTROL THAT IS NOT OPTIONAL: pretraining EXCLUDES the RL held-out states - the UNION of
`split_shell`'s eval side at depths 4, 5 and 6 for that seed, since one encoder serves all
three - and drops a pair if EITHER endpoint is in it, because `s'` passes through the same
encoder as `s`. This is a DIFFERENT split from EXP-039's, which excluded the probe's
`split_states` partition. EXP-039's encoders are therefore not reusable here even in principle;
re-pretraining is required by the design rather than by an oversight.

No baseline arms are re-run. EXP-036 measured all three on this machine with these seeds:
depth 4 = 0.1591, depth 5 = 0.0396, depth 6 = 0.0000.

Run (repo root):
    .venv/bin/python -u experiments/040_pretrained_encoder_policy/run.py \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 10
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.cube_baseline import (
    CubeConfig,
    curriculum_schedule,
    max_steps_for,
    record_filename,
    run_cube_baseline,
    shell_states,
    split_shell,
)
from neuromorphic.training.encoder_pretrain import (
    PretrainConfig,
    build_pairs,
    save_encoder,
    train_inverse_model,
)

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent

EPISODES = 10000
DEPTHS = [4, 5, 6]
PRETRAIN_DEPTHS = [1, 2, 3, 4, 5, 6]     # pair sources, as in EXP-039
HELDOUT_CAP = 200
HELDOUT_FRAC = 0.25

# EXP-039's calibrated values, selected by the pretraining objective and not by the probe.
PRETRAIN = dict(epochs=40, batch_size=256, lr=3e-3)

EXP036 = {4: 0.1591, 5: 0.0396, 6: 0.0000}   # the arms this is measured against


def curriculum_for(depth: int) -> tuple[int, ...]:
    return tuple(range(1, depth + 1))


def cell_tag(depth: int) -> str:
    """`record_filename` encodes tag/arm/depth/seed/sigma and NOT `encoder_state_path`, so the
    arm must live in the tag or these records would collide with EXP-036's."""
    return f"exp040_pre_d{depth}"


def encoder_path(out_dir: Path, seed: int) -> Path:
    return out_dir / f"exp040_encoder_s{seed}.pt"


def rl_heldout_union(provider, seed: int) -> set:
    """Every state the POLICY will be evaluated on, across all three depths.

    One encoder per seed serves depths 4, 5 and 6, so the exclusion set is the union of their
    held-out sides. `split_shell` is keyed on `split_seed`, which falls back to `seed`.
    """
    held = set()
    for d in DEPTHS:
        _, eval_states, _ = split_shell(
            shell_states(provider, d), d, seed=seed,
            heldout_cap=HELDOUT_CAP, heldout_frac=HELDOUT_FRAC,
        )
        held.update(eval_states)
    return held


def pretrain_one(seed: int, out_dir: Path) -> dict:
    """Pretrain and serialise one encoder. Returns its provenance row."""
    torch.set_num_threads(1)
    t0 = time.time()
    provider = ExactBFSDistance(max_depth=max(PRETRAIN_DEPTHS) + 1)

    states = []
    for d in PRETRAIN_DEPTHS:
        states.extend(shell_states(provider, d))

    forbidden = rl_heldout_union(provider, seed)
    pairs = build_pairs(states, forbidden=forbidden)
    # The control, asserted rather than trusted.
    assert not (forbidden & {p[0] for p in pairs}), "RL held-out state leaked in as a source"
    assert not (forbidden & {p[2] for p in pairs}), "RL held-out state leaked in as a successor"

    cfg = PretrainConfig(seed=seed, **PRETRAIN)
    result = train_inverse_model(pairs, cfg)
    save_encoder(result.sensory, encoder_path(out_dir, seed))

    return {
        "seed": seed, "n_pairs": len(pairs), "n_forbidden": len(forbidden),
        "move_accuracy": result.final_accuracy,
        "seconds": round(time.time() - t0, 1),
    }


def sweep_configs(seeds, out_dir: Path) -> list[CubeConfig]:
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=cell_tag(depth),
            depth=depth, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=curriculum_for(depth),        # equal weights: EXP-037
            entropy_beta=0.0, normalize_advantages=False,   # EXP-038 closed these
            encoder_state_path=str(encoder_path(out_dir, seed)),
            max_depth=max(6, depth), out_dir=out_dir,
        )
        for depth in DEPTHS
        for seed in seeds
    ]


def env_steps(depth: int) -> int:
    sched = curriculum_schedule(curriculum_for(depth), EPISODES, None)
    return sum(n * max_steps_for(d) for d, n in sched)


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip RL cells whose record already exists, so an interrupted run "
                         "resumes instead of recomputing")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- phase 1: one pretrained encoder per seed ----
    todo = [s for s in args.seeds if not encoder_path(args.out_dir, s).exists()]
    print(f"EXP-040 phase 1: pretraining {len(todo)} encoder(s) "
          f"({len(args.seeds) - len(todo)} already present), {args.workers} workers", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(pretrain_one, s, args.out_dir): s for s in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                print(f"  {i}/{len(todo)} seed {row['seed']}: {row['n_pairs']} pairs, "
                      f"{row['n_forbidden']} excluded, move-acc {row['move_accuracy']:.3f}, "
                      f"{row['seconds']}s", flush=True)
    for s in args.seeds:
        if not encoder_path(args.out_dir, s).exists():
            raise SystemExit(f"encoder for seed {s} missing after phase 1")

    # ---- phase 2: the policy runs ----
    configs = sweep_configs(args.seeds, args.out_dir)
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        dupes = {n for n in names if names.count(n) > 1}
        raise SystemExit(f"record filename collision, cells would overwrite: {sorted(dupes)[:5]}")

    if args.skip_existing:
        before = len(configs)
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]
        if before != len(configs):
            print(f"  skipping {before - len(configs)} already-recorded cell(s)")

    total = sum(env_steps(c.depth) for c in configs)
    print(f"\nEXP-040 phase 2: {len(configs)} runs, {args.workers} workers")
    for depth in DEPTHS:
        n = sum(1 for c in configs if c.depth == depth)
        print(f"  d{depth}: {n} runs, {env_steps(depth):,} steps/run, "
              f"vs EXP-036's {EXP036[depth]}")
    print(f"  total {total:,} env steps")
    print("  ONE VARIABLE vs EXP-036: which weights the FROZEN encoder holds. The head is")
    print("  still Linear(64 -> 6), 390 trainable parameters.\n", flush=True)

    if not configs:
        print("nothing to do.")
        return

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            print(f"  {i}/{len(configs)}", flush=True)

    print(f"\ndone. one record + one head checkpoint per run in {args.out_dir}")
    print("Run aggregate.py for the pre-registered verdicts.")


if __name__ == "__main__":
    main()
