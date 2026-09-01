"""EXP-055 phase 2: EXP-054's sequence-sensitivity, measured on the left-edge encoders.

FREE - 60 encoders took 8.8 s in EXP-054, because `concept_rates` is batched. Runs on the VPS
and must never touch the laptop.

The statistic is imported unchanged from EXP-054's module. Only the iteration is new: that
experiment's driver hardcodes its own arms' encoder paths and cannot be pointed here.

`S_cross` is reported beside `S` per EXP-054's amendment: `S` includes the within-shell term
and is therefore partly clustering, so a change in `S` that is not matched in `S_cross` is not
a change in graded distance structure. `level` is the collapse control.

Run (repo root), after phase 1:
    .venv/bin/python -u experiments/055_pretraining_left_edge/measure_s.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from neuromorphic.analysis.sequence_sensitivity import (
    DEPTHS,
    N_PER_SHELL,
    sensitivity_from_similarity,
    sequence_sensitivity,
    sim_from_record,
)
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.encoder_pretrain import load_encoder

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent

_PRETRAIN_PATH = HERE / "pretrain_left_edge.py"
_spec = importlib.util.spec_from_file_location("exp055_pretrain", _PRETRAIN_PATH)
_pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pre)
encoder_path = _pre.encoder_path
EPOCH_ARMS = _pre.EPOCH_ARMS
SEEDS = _pre.SEEDS


def record_name(epochs: int, seed: int) -> str:
    return f"exp055_S_e{epochs}_s{seed}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="+", default=list(EPOCH_ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells = [(e, s) for e in args.epochs for s in args.seeds]
    missing = [str(encoder_path(args.out_dir, e, s)) for e, s in cells
               if not encoder_path(args.out_dir, e, s).exists()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} encoder(s), first {missing[:3]}. Run phase 1 first."
        )

    print(f"EXP-055 phase 2: S over {len(cells)} encoders, shells {DEPTHS}, "
          f"up to {N_PER_SHELL} states each")
    print("  the statistic trains NOTHING and this does not touch the laptop.\n", flush=True)
    if args.dry_run:
        print(f"  --dry-run: {len(cells)} cell(s) NOT measured.")
        return

    # One bounded build, reused. Depth 6 is 11,913 states, about 0.04 s.
    provider = ExactBFSDistance(max_depth=max(DEPTHS))

    for i, (e, s) in enumerate(cells, 1):
        sensory = load_encoder(encoder_path(args.out_dir, e, s), seed=s)
        result = sequence_sensitivity(sensory, provider, seed=s)
        sim = sim_from_record(result["sim"])
        record = {
            "epochs": e,
            "seed": s,
            "S": result["S"],
            "S_cross": sensitivity_from_similarity(sim, min_separation=1),
            "level": sum(result["sim"].values()) / len(result["sim"]),
            "sim": result["sim"],
            "n_by_shell": result["n_by_shell"],
        }
        (args.out_dir / record_name(e, s)).write_text(json.dumps(record), encoding="utf-8")
        print(f"  {i}/{len(cells)}  e{e} s{s}  S={result['S']:+.4f}", flush=True)

    print(f"\ndone. S records in {args.out_dir}.")


if __name__ == "__main__":
    main()
