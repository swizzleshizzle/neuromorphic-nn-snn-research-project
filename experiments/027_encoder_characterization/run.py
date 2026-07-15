"""EXP-027 driver: cross-region encoder characterization + causal dropout-on-navigation.

Component A (always): per-seed cross-region decodability matrix (sensory concept vs every
other Brain region, paired win-fraction on displacement + optimal-action) plus concept
geometry, via ``characterize_seed``.

Component B (--dropout): mint one trained checkpoint per seed (pretrain_sensory=True,
shaping=True), reload it, and run the random/top-k/bottom-k concept-dropout navigation
curve on the held-out goals from that same run.

Run (repo root, venv active):
    .venv/Scripts/python.exe experiments/027_encoder_characterization/run.py --seeds 0 1
    .venv/Scripts/python.exe experiments/027_encoder_characterization/run.py --seeds 0 1 --dropout
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.training.checkpoints import load_trained
from neuromorphic.training.generalization import GenConfig, run_generalization

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent

_probe_spec = importlib.util.spec_from_file_location("exp027_probe", HERE / "probe.py")
probe_mod = importlib.util.module_from_spec(_probe_spec)
_probe_spec.loader.exec_module(probe_mod)

_agg_spec = importlib.util.spec_from_file_location("exp027_agg", HERE / "aggregate.py")
aggregate_mod = importlib.util.module_from_spec(_agg_spec)
_agg_spec.loader.exec_module(aggregate_mod)

_dropout_spec = importlib.util.spec_from_file_location("exp027_dropout", HERE / "dropout_eval.py")
dropout_mod = importlib.util.module_from_spec(_dropout_spec)
_dropout_spec.loader.exec_module(dropout_mod)

DROPOUT_KS = [2, 4, 8, 16, 32]


def _run_component_a(seed, *, grid_n, pretrain_epochs):
    torch.set_num_threads(1)
    return probe_mod.characterize_seed(seed, grid_n=grid_n, pretrain_epochs=pretrain_epochs)


def _run_component_b(seed, *, grid_n, episodes, out_dir):
    """Mint a trained checkpoint for this seed, reload it, run the concept-dropout curve
    on the held-out goals from the same generalization run."""
    torch.set_num_threads(1)
    ckpt_path = out_dir / f"027_ck_{seed}.pt"
    cfg = GenConfig(
        seed=seed, episodes=episodes, shaping=True, pretrain_sensory=True,
        checkpoint_path=str(ckpt_path), size=grid_n, tag=f"exp027_seed{seed}", out_dir=out_dir,
    )
    summary = run_generalization(cfg)
    brain, head = load_trained(str(ckpt_path), grid_n=grid_n, seed=seed)
    heldout_goals = [tuple(g) for g in summary["heldout_goals"]]
    curve = dropout_mod.dropout_curve(brain, head, heldout_goals, grid_n=grid_n, ks=DROPOUT_KS)
    return seed, curve


def _format_dropout_md(dropout_by_seed: dict) -> str:
    """Markdown: mean held-out success across seeds, per mode x k."""
    if not dropout_by_seed:
        return "(no dropout data - run with --dropout)\n"
    seeds = sorted(dropout_by_seed)
    modes = ("random", "top", "bottom")
    ks = sorted(dropout_by_seed[seeds[0]]["random"])
    lines = ["| k | " + " | ".join(modes) + " |", "| --- | " + " | ".join("---" for _ in modes) + " |"]
    for k in ks:
        cells = []
        for mode in modes:
            vals = [dropout_by_seed[s][mode][k] for s in seeds]
            cells.append(f"{sum(vals) / len(vals):.0%}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(description="EXP-027 encoder characterization")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    p.add_argument("--grid-n", type=int, default=5)
    p.add_argument("--pretrain-epochs", type=int, default=200)
    p.add_argument("--episodes", type=int, default=600,
                   help="policy-training episodes for the Component B checkpoint")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--dropout", action="store_true",
                   help="also run Component B (mints one checkpoint per seed)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = HERE / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, len(args.seeds)))

    print(f"Component A: characterizing {len(args.seeds)} seeds across {workers} workers ...",
          flush=True)
    per_seed = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_run_component_a, seed, grid_n=args.grid_n,
                     pretrain_epochs=args.pretrain_epochs): seed
            for seed in args.seeds
        }
        for done, fut in enumerate(as_completed(futures), 1):
            seed = futures[fut]
            res = fut.result()
            per_seed.append(res)
            sen = res["regions"]["sensory"]
            print(f"[A {done}/{len(args.seeds)}] seed {seed}: sensory displacement_r2="
                  f"{sen['displacement_r2']:.3f} optimal_action_acc={sen['optimal_action_acc']:.3f}",
                  flush=True)
    per_seed.sort(key=lambda r: r["seed"])

    agg = aggregate_mod.aggregate_regions(per_seed)
    matrix_md = aggregate_mod.format_matrix(agg)
    print("\n=== region x target decodability (paired win-fraction, sensory > region) ===\n"
          + matrix_md, flush=True)

    dropout_by_seed = {}
    if args.dropout:
        print(f"\nComponent B: minting {len(args.seeds)} checkpoints + dropout curves ...",
              flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_run_component_b, seed, grid_n=args.grid_n,
                         episodes=args.episodes, out_dir=out_dir): seed
                for seed in args.seeds
            }
            for done, fut in enumerate(as_completed(futures), 1):
                seed, curve = fut.result()
                dropout_by_seed[seed] = curve
                print(f"[B {done}/{len(args.seeds)}] seed {seed} dropout curve done", flush=True)

    dropout_md = _format_dropout_md(dropout_by_seed)

    summary = {
        "config": {"seeds": args.seeds, "grid_n": args.grid_n,
                   "pretrain_epochs": args.pretrain_epochs, "dropout": args.dropout},
        "per_seed": per_seed,
        "aggregate": agg,
        "dropout": dropout_by_seed,
    }
    (out_dir / "027_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "027_matrix.md").write_text(matrix_md + "\n")
    (out_dir / "027_dropout.md").write_text(dropout_md)

    if args.dropout:
        print("\n=== dropout curve (mean held-out success across seeds) ===\n" + dropout_md,
              flush=True)


if __name__ == "__main__":
    main()
