"""EXP-028 driver: sensory-code ablation dose-response.

Mint 12 pretrained+frozen sensory encoders once, then for each (operator, dose, seed)
reload the cached encoder and re-train ONLY the linear policy head against the ablated
concept, measuring held-out navigation success. Operators: gaussian noise, unit-drop
random, unit-drop top-k (most-important-first, importance from that encoder).

Run (repo root, venv active):
    .venv/Scripts/python.exe experiments/028_sensory_ablation/run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.analysis.ablate import AblationSpec
from neuromorphic.analysis.probes import region_rate_matrix, task_targets, unit_importance
from neuromorphic.training.generalization import GenConfig, run_generalization
from neuromorphic.training.checkpoints import load_trained
from neuromorphic.training.pretrain import enumerate_states, split_states

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent

_agg_spec = importlib.util.spec_from_file_location("exp028_agg", HERE / "aggregate.py")
aggregate_mod = importlib.util.module_from_spec(_agg_spec)
_agg_spec.loader.exec_module(aggregate_mod)

GAUSS_DOSES = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
DROP_DOSES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]


def mint_encoder(seed, *, grid_n, out_dir, episodes, pretrain_epochs) -> str:
    torch.set_num_threads(1)
    out_dir = Path(out_dir)
    ck = out_dir / f"028_enc_{seed}.pt"
    cfg = GenConfig(
        seed=seed, episodes=episodes, shaping=True, pretrain_sensory=True,
        pretrain_epochs=pretrain_epochs, checkpoint_path=str(ck), size=grid_n,
        tag=f"exp028_mint_seed{seed}", out_dir=out_dir,
    )
    run_generalization(cfg)
    return str(ck)


def importance_order(ckpt_path, *, grid_n) -> list[int]:
    """Most-important concept-unit ranking for the cached encoder (train states)."""
    brain, _ = load_trained(ckpt_path, grid_n=grid_n)
    tr, _ = split_states(enumerate_states(grid_n), frac_heldout=0.2, seed=0)
    X = region_rate_matrix(brain, tr, region_key="sensory", signal_key="concept",
                           width=brain.content, recall=False, T=brain.T,
                           generator=torch.Generator().manual_seed(0))
    order = unit_importance(X, task_targets(tr, grid_n)["displacement"])
    return [int(i) for i in order]


def run_cell(seed, operator, dose, *, ckpt_path, order, grid_n, episodes) -> dict:
    torch.set_num_threads(1)
    if operator == "gaussian":
        spec, ab_order = AblationSpec("gaussian", dose=dose, seed=seed), None
    elif operator == "unitdrop_random":
        spec, ab_order = AblationSpec("unitdrop", dose=dose, mode="random", seed=seed), None
    elif operator == "unitdrop_top":
        spec, ab_order = AblationSpec("unitdrop", dose=dose, mode="top", seed=seed), order
    else:
        raise ValueError(f"unknown operator {operator!r}")
    cfg = GenConfig(
        seed=seed, episodes=episodes, shaping=True, size=grid_n,
        load_encoder_path=ckpt_path, ablation=spec, ablation_order=ab_order,
        tag=f"exp028_{operator}_d{dose}_s{seed}", out_dir=Path(ckpt_path).parent,
    )
    summary = run_generalization(cfg)
    return {"operator": operator, "dose": dose, "seed": seed,
            "heldout_success": summary["eval"]["heldout"]["success_rate"]}


def _cells_for_seed(seed):
    out = []
    for d in GAUSS_DOSES:
        out.append(("gaussian", d, seed))
    for d in DROP_DOSES:
        out.append(("unitdrop_random", d, seed))
        out.append(("unitdrop_top", d, seed))
    return out


def parse_args():
    p = argparse.ArgumentParser(description="EXP-028 sensory-code ablation")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    p.add_argument("--grid-n", type=int, default=5)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--pretrain-epochs", type=int, default=200)
    p.add_argument("--workers", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = HERE / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, len(args.seeds)))

    print(f"Minting {len(args.seeds)} encoders across {workers} workers ...", flush=True)
    ckpts, orders = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(mint_encoder, s, grid_n=args.grid_n, out_dir=out_dir,
                          episodes=args.episodes, pretrain_epochs=args.pretrain_epochs): s
                for s in args.seeds}
        for fut in as_completed(futs):
            s = futs[fut]
            ckpts[s] = fut.result()
            print(f"  minted encoder seed {s}", flush=True)
    for s in args.seeds:
        orders[s] = importance_order(ckpts[s], grid_n=args.grid_n)

    jobs = [job for s in args.seeds for job in _cells_for_seed(s)]
    print(f"Sweeping {len(jobs)} ablation cells across {workers} workers ...", flush=True)
    cells = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_cell, s, op, d, ckpt_path=ckpts[s], order=orders[s],
                          grid_n=args.grid_n, episodes=args.episodes): (op, d, s)
                for (op, d, s) in jobs}
        for done, fut in enumerate(as_completed(futs), 1):
            cells.append(fut.result())
            if done % 12 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] cells done", flush=True)

    curve = aggregate_mod.aggregate_curve(cells)
    curve_md = aggregate_mod.format_curve(curve)
    (out_dir / "028_curve.md").write_text(curve_md)
    (out_dir / "028_summary.json").write_text(json.dumps(
        {"config": {"seeds": args.seeds, "grid_n": args.grid_n, "episodes": args.episodes},
         "cells": cells, "curve": curve}, indent=2))
    print("\n=== ablation dose-response (mean held-out success across seeds) ===\n" + curve_md,
          flush=True)


if __name__ == "__main__":
    main()
