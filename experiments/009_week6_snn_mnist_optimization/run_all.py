"""Drive all 3 variants and emit comparison artifacts.

Usage:
    python run_all.py                  # train all 3 fresh + write artifacts
    python run_all.py --tables-only    # skip training; rebuild tables from
                                       # already-saved per-variant checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from neuromorphic.config import load_config

VARIANT_YAMLS = ["v1_mlp_tuned.yaml", "v2_cnn_baseline.yaml", "v3_cnn_tuned.yaml"]


# ---- formatting helpers --------------------------------------------------

def _fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s"


def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}K"
    return str(n)


# ---- table writers -------------------------------------------------------

def write_comparison_md(results: list[dict], path: Path) -> None:
    """Pipe-table format matching the design doc's preview."""
    cols = ["Variant", "Arch", "Params", "num_steps", "beta",
            "Epochs", "Train time", "Test acc"]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in results:
        lines.append("| " + " | ".join([
            r["variant"],
            r["arch"],
            _fmt_params(r["num_params"]),
            str(r["num_steps"]),
            f'{r["beta"]:.2f}',
            str(r["epochs"]),
            _fmt_time(r["train_seconds"]),
            f'{r["final_test_acc"] * 100:.2f}%',
        ]) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_csv(results: list[dict], path: Path) -> None:
    fieldnames = ["variant", "arch", "num_params", "num_steps", "beta",
                  "epochs", "train_seconds", "final_test_acc"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})


# ---- result loading from saved checkpoints -------------------------------

def _result_from_checkpoint(yaml_path: Path) -> dict:
    """Rebuild a metrics dict from a previously-saved checkpoint.

    Used by ``--tables-only`` so we don't have to re-train to regenerate
    tables after a writer/format change.
    """
    config = load_config(yaml_path)
    ckpt_path = Path(config.viz_output_dir) / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run training first."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    return {
        "variant": cfg["run_name"],
        "arch": cfg["arch"],
        "num_params": _params_from_state(ckpt["model_state"]),
        "num_steps": cfg["num_steps"],
        "beta": cfg["beta"],
        "epochs": cfg["epochs"],
        "train_seconds": ckpt["train_seconds"],
        "final_test_acc": ckpt["final_accuracy"],
        "checkpoint_path": str(ckpt_path.resolve()),
    }


def _params_from_state(state_dict: dict) -> int:
    return sum(t.numel() for t in state_dict.values())


# ---- driver --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-only", action="store_true",
                        help="Skip training, rebuild tables from existing checkpoints.")
    args = parser.parse_args()

    out_dir = HERE / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tables_only:
        results = [_result_from_checkpoint(HERE / y) for y in VARIANT_YAMLS]
    else:
        from run import train  # local import so --tables-only doesn't need viz dep
        results = []
        for y in VARIANT_YAMLS:
            print(f"\n===== Training {y} =====")
            metrics = train(load_config(HERE / y))
            results.append(metrics)

    # Tables
    write_comparison_md(results, out_dir / "comparison.md")
    write_comparison_csv(results, out_dir / "comparison.csv")
    print(f"\nWrote {out_dir / 'comparison.md'}")
    print(f"Wrote {out_dir / 'comparison.csv'}")

    # Best checkpoint copy + sidecar
    best = max(results, key=lambda r: r["final_test_acc"])
    shutil.copy(best["checkpoint_path"], out_dir / "best_checkpoint.pt")
    with (out_dir / "best_checkpoint.json").open("w", encoding="utf-8") as f:
        json.dump({
            "variant": best["variant"],
            "final_test_acc": best["final_test_acc"],
            "config_summary": {
                "arch": best["arch"],
                "num_steps": best["num_steps"],
                "beta": best["beta"],
                "epochs": best["epochs"],
                "num_params": best["num_params"],
            },
        }, f, indent=2)
    print(f"\nBest: {best['variant']} @ {best['final_test_acc'] * 100:.2f}%")
    print(f"Copied to {out_dir / 'best_checkpoint.pt'}")

    print("\nFinal comparison:")
    print((out_dir / "comparison.md").read_text())


if __name__ == "__main__":
    main()
