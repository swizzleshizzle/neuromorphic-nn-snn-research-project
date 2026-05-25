"""Run both variants, write comparison artifacts, run reload-and-verify gate.

Two-variant driver for exp 011. Imports ``train`` from ``run.py``.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neuromorphic.config import ExperimentConfig, load_config
from neuromorphic.utils import get_device

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run import train, _full_test_eval  # noqa: E402
from models import SequentialSNN  # noqa: E402


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def write_comparison_md(results: list[dict], path: Path) -> None:
    """Pipe-table format. Columns: Variant | Hidden | Params | Epochs | Train time | Test acc | Gap."""
    ff_acc = next(r["final_test_acc"] for r in results if not r["recurrent"])
    lines = [
        "| Variant | Hidden layer | Params | Epochs | Train time | Test acc | Gap vs feedforward |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        hidden = "snn.RLeaky(all_to_all=True)" if r["recurrent"] else "snn.Leaky"
        gap = (r["final_test_acc"] - ff_acc) * 100.0
        gap_str = f"{gap:+.2f}%" if r["recurrent"] else "0.00%"
        lines.append(
            f"| `{r['variant']}` | {hidden} | {r['num_params']:,} | {r['epochs']} | "
            f"{_fmt_time(r['train_seconds'])} | {r['final_test_acc']*100:.2f}% | {gap_str} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_csv(results: list[dict], path: Path) -> None:
    ff_acc = next(r["final_test_acc"] for r in results if not r["recurrent"])
    rows = []
    for r in results:
        rows.append({
            "variant": r["variant"],
            "recurrent": r["recurrent"],
            "hidden_layer": "rleaky_all_to_all" if r["recurrent"] else "leaky",
            "num_params": r["num_params"],
            "epochs": r["epochs"],
            "train_seconds": r["train_seconds"],
            "final_test_acc": r["final_test_acc"],
            "gap_vs_feedforward": r["final_test_acc"] - ff_acc,
        })
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def reload_gate(best_result: dict, best_config: ExperimentConfig) -> None:
    """§8 gate #4: reload best checkpoint, re-evaluate, assert within +/-0.1%."""
    device = get_device()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    test_set = datasets.MNIST(best_config.data_root, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=best_config.batch_size, shuffle=False, drop_last=False)

    ckpt = torch.load(best_result["checkpoint_path"], map_location=device)
    net = SequentialSNN(
        num_inputs=best_config.num_inputs,
        hidden_size=best_config.hidden_size,
        num_outputs=best_config.num_outputs,
        beta=best_config.beta,
        threshold=best_config.threshold,
        reset_mechanism=best_config.reset_mechanism,
        num_steps=best_config.num_steps,
        readout_window=best_config.readout_window,
        recurrent=best_config.recurrent,
    ).to(device)
    net.load_state_dict(ckpt["model_state"])

    _, reload_acc = _full_test_eval(net, test_loader, device)
    delta = abs(reload_acc - best_result["final_test_acc"])
    assert delta <= 0.001, (
        f"reload gate FAILED: recorded {best_result['final_test_acc']:.4f}, "
        f"reloaded {reload_acc:.4f}, delta {delta:.4f}"
    )
    print(f"[reload gate] recorded={best_result['final_test_acc']:.4f} "
          f"reloaded={reload_acc:.4f} delta={delta:.5f} OK")


def main() -> None:
    yamls = ["recurrent.yaml", "feedforward.yaml"]
    configs = [load_config(HERE / y) for y in yamls]

    results = []
    for c in configs:
        print("\n" + "=" * 72)
        print(f"Training variant: {c.run_name}")
        print("=" * 72)
        results.append(train(c))

    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    write_comparison_md(results, out / "comparison.md")
    write_comparison_csv(results, out / "comparison.csv")

    best = max(results, key=lambda r: r["final_test_acc"])
    best_config = next(c for c in configs if c.run_name == best["variant"])
    shutil.copy(best["checkpoint_path"], out / "best_checkpoint.pt")
    ff_acc = next(r["final_test_acc"] for r in results if not r["recurrent"])
    (out / "best_checkpoint.json").write_text(json.dumps({
        "variant": best["variant"],
        "final_test_acc": best["final_test_acc"],
        "gap_vs_feedforward": best["final_test_acc"] - ff_acc,
        "config_summary": {
            "recurrent": best["recurrent"],
            "num_steps": best["num_steps"],
            "hidden_size": best["hidden_size"],
            "beta": best["beta"],
            "epochs": best["epochs"],
            "num_params": best["num_params"],
        },
    }, indent=2), encoding="utf-8")

    reload_gate(best, best_config)

    print("\nSummary")
    for r in results:
        print(f"  {r['variant']:>12s}  acc={r['final_test_acc']*100:.2f}%  "
              f"params={r['num_params']:,}  train={_fmt_time(r['train_seconds'])}")
    print(f"\nBest: {best['variant']} @ {best['final_test_acc']*100:.2f}%  "
          f"(gap vs feedforward: {(best['final_test_acc']-ff_acc)*100:+.2f}%)")
    print(f"Artifacts in {out}/")


if __name__ == "__main__":
    main()
