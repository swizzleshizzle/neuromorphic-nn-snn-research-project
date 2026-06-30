"""EXP-025 - head capacity probe (MLP head vs linear head).

Sweeps {linear, mlp} x {shaped, sparse} x seeds, paired by seed (identical goal split
per seed), and aggregates held-out success into a markdown evidence table. The brain stays
frozen (ADR-0001); only the policy head changes.

Run (repo root, venv active):
    .venv/Scripts/python.exe experiments/025_head_capacity/run.py
    .venv/Scripts/python.exe experiments/025_head_capacity/run.py --seeds 0 1 2 3 4 --episodes 600
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch  # noqa: E402

from neuromorphic.training.generalization import GenConfig, run_generalization

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
_agg_spec = importlib.util.spec_from_file_location("exp025_aggregate", HERE / "aggregate.py")
aggregate_mod = importlib.util.module_from_spec(_agg_spec)
_agg_spec.loader.exec_module(aggregate_mod)


def build_configs(seeds: list[int], episodes: int, out_dir: Path) -> list[GenConfig]:
    """The 2 heads x 2 regimes x seeds sweep, each tagged uniquely by regime/head/seed."""
    configs = []
    for head_type in ("linear", "mlp"):
        for shaping in (True, False):
            regime = "shaped" if shaping else "sparse"
            for seed in seeds:
                configs.append(GenConfig(
                    seed=seed, episodes=episodes, shaping=shaping,
                    head_type=head_type, hidden=128,
                    tag=f"{regime}_{head_type}_seed{seed}", out_dir=out_dir,
                ))
    return configs


def _run_one(cfg: GenConfig) -> dict:
    """Worker: one config in its own process, pinned single-threaded for clean scaling."""
    torch.set_num_threads(1)
    return run_generalization(cfg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-025 head capacity probe")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--workers", type=int, default=14,
                   help="parallel processes (runs are independent and single-threaded)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = HERE / "outputs"
    configs = build_configs(args.seeds, args.episodes, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, len(configs)))
    print(f"running {len(configs)} configs across {workers} workers ...", flush=True)
    summaries = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_one, cfg): cfg for cfg in configs}
        for done, fut in enumerate(as_completed(futures), 1):
            cfg = futures[fut]
            summaries.append(fut.result())
            print(f"[{done}/{len(configs)}] {cfg.tag} done", flush=True)

    agg = aggregate_mod.aggregate(summaries)
    table = aggregate_mod.format_table(agg)
    out_dir.mkdir(parents=True, exist_ok=True)
    # tuple keys are not JSON-serializable; join into "head|regime" strings
    agg_json = {f"{head}|{regime}": v for (head, regime), v in agg.items()}
    (out_dir / "025_summary.json").write_text(json.dumps(agg_json, indent=2))
    (out_dir / "025_table.md").write_text(table + "\n")
    print("\n" + table, flush=True)


if __name__ == "__main__":
    main()
