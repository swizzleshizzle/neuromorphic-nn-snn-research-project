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


def build_configs(seeds, episodes, out_dir, n_heldout=10):
    """Linear head, BOTH arms (pretrain on/off), {shaped, sparse} x seeds.

    Paired de-noise design: the pre-trained-encoder arm (tag ``_pt``) and the random-encoder
    baseline arm (tag ``_rand``) run at the same seeds / held-out size so the comparison is
    apples-to-apples at higher n.
    """
    configs = []
    for pretrain in (True, False):
        arm = "pt" if pretrain else "rand"
        for shaping in (True, False):
            regime = "shaped" if shaping else "sparse"
            for seed in seeds:
                configs.append(GenConfig(
                    seed=seed, episodes=episodes, shaping=shaping, head_type="linear",
                    n_heldout=n_heldout, pretrain_sensory=pretrain,
                    tag=f"{regime}_linear_seed{seed}_{arm}", out_dir=out_dir,
                ))
    return configs


def _comparison_table(agg_pt, agg_rand):
    """Arm-aware Stage-2 table: pretrained vs random held-out success, per regime."""
    lines = [
        "| regime | arm | n | heldout mean | heldout spread | train mean |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for regime in ("shaped", "sparse"):
        key = ("linear", regime)
        for arm_label, agg in (("pretrained", agg_pt), ("random", agg_rand)):
            m = agg.get(key)
            if m is None:
                continue
            lines.append(
                f"| {regime} | {arm_label} | {m['n']} | {m['heldout_mean']:.0%} | "
                f"{m['heldout_spread']:.0%} | {m['train_mean']:.0%} |"
            )
    return "\n".join(lines)


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
    p.add_argument("--n-heldout", type=int, default=10,
                   help="held-out goal cells for the navigation eval (bigger = tighter estimate)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = HERE / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = build_configs(args.seeds, args.episodes, out_dir, n_heldout=args.n_heldout)
    workers = max(1, min(args.workers, len(configs)))

    print(f"running {len(configs)} configs (pretrained + random arms) across {workers} workers ...",
          flush=True)
    summaries = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_one, cfg): cfg for cfg in configs}
        for done, fut in enumerate(as_completed(futures), 1):
            cfg = futures[fut]
            summaries.append(fut.result())
            print(f"[{done}/{len(configs)}] {cfg.tag} done", flush=True)

    pt_summaries = [s for s in summaries if s["config"]["pretrain_sensory"]]
    rand_summaries = [s for s in summaries if not s["config"]["pretrain_sensory"]]

    # Stage 1 gate: pretrained vs random-encoder held-out displacement error, per seed
    print("\n=== Stage 1: displacement decode error (held-out states) ===", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rand_ref = dict(ex.map(_random_reference, args.seeds))
    pt_err = {}
    for s in pt_summaries:
        pt_err.setdefault(s["config"]["seed"], s["pretrain"]["heldout_disp_error"])
    for seed in args.seeds:
        print(f"  seed {seed}: pretrained {pt_err[seed]:.3f}  vs  random {rand_ref[seed]:.3f}",
              flush=True)

    # Stage 2: arm-aware held-out success (aggregate each arm separately, then compare)
    agg_pt = aggregate_mod.aggregate(pt_summaries)
    agg_rand = aggregate_mod.aggregate(rand_summaries)
    table = _comparison_table(agg_pt, agg_rand)
    (out_dir / "026_summary.json").write_text(json.dumps({
        "stage2": {
            "pretrained": {f"{h}|{r}": v for (h, r), v in agg_pt.items()},
            "random": {f"{h}|{r}": v for (h, r), v in agg_rand.items()},
        },
        "stage1": {"pretrained": pt_err, "random": rand_ref},
        "config": {"seeds": args.seeds, "n_heldout": args.n_heldout, "episodes": args.episodes},
    }, indent=2))
    (out_dir / "026_table.md").write_text(table + "\n")
    print("\n=== Stage 2: held-out navigation success (pretrained vs random) ===\n" + table,
          flush=True)


if __name__ == "__main__":
    main()
