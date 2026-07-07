"""EXP-026 - sensory pre-training (engage the encoder).

Two-stage test of ADR-0001 Amendment 2 (the frozen encoder is the cap):
  Stage 1 - pre-train the sensory encoder to decode goal-relative displacement; gate on the
            held-out displacement error vs a random-encoder reference.
  Stage 2 - freeze the pre-trained encoder, train the linear policy head, and compare held-out
            navigation success against the EXP-025 random-encoder band (shaped 23%, sparse 27%).

Run (repo root, venv active):
    .venv/Scripts/python.exe experiments/026_sensory_pretrain/run.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch  # noqa: E402

from neuromorphic.brain import Brain
from neuromorphic.training.generalization import GenConfig, run_generalization
from neuromorphic.training.pretrain import pretrain_sensory

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
_agg_spec = importlib.util.spec_from_file_location(
    "exp025_aggregate", HERE.parent / "025_head_capacity" / "aggregate.py"
)
aggregate_mod = importlib.util.module_from_spec(_agg_spec)
_agg_spec.loader.exec_module(aggregate_mod)


def build_configs(seeds, episodes, out_dir):
    """Linear head, pretrain_sensory=True, {shaped, sparse} x seeds; tags suffixed _pt."""
    configs = []
    for shaping in (True, False):
        regime = "shaped" if shaping else "sparse"
        for seed in seeds:
            configs.append(GenConfig(
                seed=seed, episodes=episodes, shaping=shaping, head_type="linear",
                pretrain_sensory=True, tag=f"{regime}_linear_seed{seed}_pt", out_dir=out_dir,
            ))
    return configs


def _run_one(cfg):
    torch.set_num_threads(1)
    return run_generalization(cfg)


def _random_reference(seed):
    """Stage-1 reference: can a linear readout decode displacement from a RANDOM encoder?"""
    torch.set_num_threads(1)
    brain = Brain(grid_n=5, seed=seed)
    info = pretrain_sensory(
        brain.sensory, grid_n=5, epochs=200, lr=1e-3, seed=seed,
        generator=torch.Generator().manual_seed(seed), freeze_encoder=True,
    )
    return seed, info["heldout_disp_error"]


def parse_args():
    p = argparse.ArgumentParser(description="EXP-026 sensory pre-training")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--workers", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = HERE / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = build_configs(args.seeds, args.episodes, out_dir)
    workers = max(1, min(args.workers, len(configs)))

    print(f"running {len(configs)} pretrain configs across {workers} workers ...", flush=True)
    summaries = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_one, cfg): cfg for cfg in configs}
        for done, fut in enumerate(as_completed(futures), 1):
            cfg = futures[fut]
            summaries.append(fut.result())
            print(f"[{done}/{len(configs)}] {cfg.tag} done", flush=True)

    # Stage 1 gate: pre-trained vs random-encoder held-out displacement error, per seed
    print("\n=== Stage 1: displacement decode error (held-out states) ===", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rand_ref = dict(ex.map(_random_reference, args.seeds))
    pt_err = {}
    for s in summaries:
        seed = s["config"]["seed"]
        pt_err.setdefault(seed, s["pretrain"]["heldout_disp_error"])
    for seed in args.seeds:
        print(f"  seed {seed}: pretrained {pt_err[seed]:.3f}  vs  random {rand_ref[seed]:.3f}",
              flush=True)

    # Stage 2 table: held-out navigation success (reuse the EXP-025 aggregator)
    agg = aggregate_mod.aggregate(summaries)
    table = aggregate_mod.format_table(agg)
    agg_json = {f"{head}|{regime}": v for (head, regime), v in agg.items()}
    (out_dir / "026_summary.json").write_text(json.dumps(
        {"stage2": agg_json, "stage1": {"pretrained": pt_err, "random": rand_ref}}, indent=2))
    (out_dir / "026_table.md").write_text(table + "\n")
    print("\n=== Stage 2: held-out navigation success ===\n" + table, flush=True)


if __name__ == "__main__":
    main()
