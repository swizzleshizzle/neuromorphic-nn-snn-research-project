# experiments/050_objective_vs_gradient/extend_pretrain.py
"""Phase 1: another 40 epochs of inverse-model pretraining from E0, warm-started -> E0+.

THE STEP-MATCHED CONTROL for EXP-047. REINFORCE calls `optimizer.step()` once per episode, so
EXP-047 applied 10,000 encoder updates. This applies ceil(66192/256)*40 = 10,360 - a 1.036x match
that nobody chose the size of; it falls out of configs fixed in EXP-040 and EXP-047 for unrelated
reasons.

The parallel with EXP-047 is exact:

    EXP-047 -> E1   E0 + fresh POLICY  head + 10,000 REINFORCE updates
    this    -> E0+  E0 + fresh INVERSE head + 10,360 supervised updates

Same starting encoder, same rough update count, fresh task head both times. Only the objective
differs. `train_inverse_model` already accepts `sensory=` for warm-start, so nothing in the
library changes.

> THE MATCH FAVOURS THIS CONTROL, deliberately: each update here sees 256 pairs against RL's ~15
> environment steps, so it gets ~17x more data per step AND a clean supervised gradient. A
> control handed the advantage that still loses makes the conclusion stronger.

Run (repo root):
    .venv/bin/python -u experiments/050_objective_vs_gradient/extend_pretrain.py --workers 10
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.cube_baseline import shell_states, split_shell
from neuromorphic.training.encoder_pretrain import (
    PretrainConfig,
    build_pairs,
    load_encoder,
    save_encoder,
    train_inverse_model,
)

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
E0_DIR = Path("experiments/040_pretrained_encoder_policy/outputs")

# EXP-040's phase-1 configuration, reproduced EXACTLY. Changing any of it would make E0+ a
# different intervention from "more of the same objective".
_EXP040_PATH = HERE.parent / "040_pretrained_encoder_policy" / "run.py"
_spec = importlib.util.spec_from_file_location("exp040_run", _EXP040_PATH)
exp040 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp040)

PRETRAIN = dict(exp040.PRETRAIN)          # epochs=40, batch_size=256, lr=3e-3
PRETRAIN_DEPTHS = list(exp040.PRETRAIN_DEPTHS)
RL_ENCODER_UPDATES = 10_000               # what EXP-047 applied, for the banner


def e0_path(seed: int) -> Path:
    return E0_DIR / f"exp040_encoder_s{seed}.pt"


def e0plus_path(out_dir: Path, seed: int) -> Path:
    return out_dir / f"exp050_encoder_plus_s{seed}.pt"


def extend_one(seed: int, out_dir: Path) -> dict:
    """40 more epochs from this seed's E0, with EXP-040's exclusions unchanged."""
    torch.set_num_threads(1)
    t0 = time.time()
    provider = ExactBFSDistance(max_depth=max(PRETRAIN_DEPTHS) + 1)

    states = []
    for d in PRETRAIN_DEPTHS:
        states.extend(shell_states(provider, d))

    # The SAME held-out exclusions EXP-040 used, via its own function. If E0+ were trained on
    # states the policy is later evaluated on, arm F would be measuring leakage rather than
    # pretraining, and it would beat arm B for the wrong reason.
    forbidden = exp040.rl_heldout_union(provider, seed)
    pairs = build_pairs(states, forbidden=forbidden)
    assert not (forbidden & {p[0] for p in pairs}), "RL held-out state leaked in as a source"
    assert not (forbidden & {p[2] for p in pairs}), "RL held-out state leaked in as a successor"

    sensory = load_encoder(e0_path(seed))
    before = sensory.fc1.weight.detach().clone()

    # `seed` is reused so the data order matches EXP-040's own run; the fresh InverseModel head
    # is what makes this "another round" rather than a resumed one - the same shape as EXP-047,
    # which also began from E0 with a fresh task head.
    cfg = PretrainConfig(seed=seed, **PRETRAIN)
    result = train_inverse_model(pairs, cfg, sensory=sensory)
    save_encoder(result.sensory, e0plus_path(out_dir, seed))

    drift = (result.sensory.fc1.weight - before).norm().item() / before.norm().item()
    steps = -(-len(pairs) // PRETRAIN["batch_size"]) * PRETRAIN["epochs"]
    return {"seed": seed, "n_pairs": len(pairs), "encoder_updates": steps,
            "move_accuracy": result.final_accuracy, "rel_drift": drift,
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for s in args.seeds:
        if not e0_path(s).exists():
            raise SystemExit(f"missing E0 encoder {e0_path(s)}")
    todo = [s for s in args.seeds if not e0plus_path(args.out_dir, s).exists()]

    print(f"EXP-050 phase 1: {len(todo)} encoder(s) to extend, "
          f"{len(args.seeds) - len(todo)} already present, {args.workers} workers")
    print(f"  {PRETRAIN}, exclusions over depths {PRETRAIN_DEPTHS} (EXP-040's, unchanged)")
    print(f"  step-matched control: ~10,360 encoder updates vs EXP-047's "
          f"{RL_ENCODER_UPDATES:,} - 1.036x")
    print(f"  the control sees ~17x more data per update. It is FAVOURED, on purpose.\n",
          flush=True)
    if not todo:
        print("nothing to do.")
        return
    if args.dry_run:
        print(f"  --dry-run: {len(todo)} encoder(s) NOT started.")
        return

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extend_one, s, args.out_dir): s for s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            print(f"  {i}/{len(todo)} seed {r['seed']}: {r['encoder_updates']:,} updates, "
                  f"move-acc {r['move_accuracy']:.3f}, drift {r['rel_drift']*100:.2f}%, "
                  f"{r['seconds']}s", flush=True)

    for s in args.seeds:
        if not e0plus_path(args.out_dir, s).exists():
            raise SystemExit(f"encoder for seed {s} missing after phase 1")
    print(f"\ndone in {time.time() - t0:.0f}s. {len(args.seeds)} E0+ encoder(s) available.")


if __name__ == "__main__":
    main()
