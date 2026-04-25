"""
Week 3 Session 2 — experiment driver.

Runs all 8 configs sequentially, writes per-epoch results to results.csv,
and produces a training-loss plot for comparison.

Usage:
    python run_experiments.py

If you want to run a single config for quick iteration:
    python run_experiments.py --only baseline
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, works headless
import matplotlib.pyplot as plt

from train import train_one_config


# --- The matrix ------------------------------------------------------------
# Same seed, same epochs, same batch size across all runs.
# Only the three axes (arch, optimizer, lr) change. This is the whole point.

COMMON = {
    "epochs": 5,
    "batch_size": 64,
    "seed": 42,
}

EXPERIMENTS = [
    # Run 1 — baseline (replicate Week 1)
    {"run_id": "01_baseline_mlp_sgd_lr0.01",    "arch": "baseline_mlp", "optimizer": "sgd",  "lr": 0.01,   **COMMON},
    # Run 2 — undercapacity
    {"run_id": "02_tiny_mlp_sgd_lr0.01",        "arch": "tiny_mlp",     "optimizer": "sgd",  "lr": 0.01,   **COMMON},
    # Run 3 — optimizer swap (MLP + Adam at Adam's default LR)
    {"run_id": "03_baseline_mlp_adam_lr0.001",  "arch": "baseline_mlp", "optimizer": "adam", "lr": 0.001,  **COMMON},
    # Run 4 — LR too high (deliberate failure)
    {"run_id": "04_baseline_mlp_sgd_lr1.0",     "arch": "baseline_mlp", "optimizer": "sgd",  "lr": 1.0,    **COMMON},
    # Run 5 — LR too low (undertrained)
    {"run_id": "05_baseline_mlp_sgd_lr0.0001",  "arch": "baseline_mlp", "optimizer": "sgd",  "lr": 0.0001, **COMMON},
    # Run 6 — arch swap (CNN, SGD to match baseline optimizer)
    {"run_id": "06_simple_cnn_sgd_lr0.01",      "arch": "simple_cnn",   "optimizer": "sgd",  "lr": 0.01,   **COMMON},
    # Run 7 — "modern" config: CNN + Adam
    {"run_id": "07_simple_cnn_adam_lr0.001",    "arch": "simple_cnn",   "optimizer": "adam", "lr": 0.001,  **COMMON},
    # Run 8 — Adam at too-high LR (does Adam's adaptivity save it?)
    {"run_id": "08_simple_cnn_adam_lr0.01",     "arch": "simple_cnn",   "optimizer": "adam", "lr": 0.01,   **COMMON},
]


# --- CSV writing -----------------------------------------------------------

CSV_COLUMNS = [
    "run_id", "arch", "optimizer", "lr", "seed",
    "epoch", "train_loss", "train_acc", "test_loss", "test_acc",
    "wall_time_sec", "diverged",
]

def write_results_csv(all_results: list, path: Path):
    """One row per epoch per run. Wall time is repeated on every row of a given run
    — redundant but keeps the CSV shape regular."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in all_results:
            for epoch_row in result["epoch_history"]:
                writer.writerow({
                    "run_id": result["run_id"],
                    "arch": result["arch"],
                    "optimizer": result["optimizer"],
                    "lr": result["lr"],
                    "seed": result["seed"],
                    "epoch": epoch_row["epoch"],
                    "train_loss": epoch_row["train_loss"],
                    "train_acc": epoch_row["train_acc"],
                    "test_loss": epoch_row["test_loss"],
                    "test_acc": epoch_row["test_acc"],
                    "wall_time_sec": result["wall_time_sec"],
                    "diverged": result["diverged"],
                })


def write_summary_table(all_results: list):
    """Print a compact summary table to stdout."""
    print("\n" + "=" * 90)
    print(f"{'Run':<32} {'Arch':<14} {'Opt':<6} {'LR':<8} {'Test Acc':<10} {'Time (s)':<10}")
    print("=" * 90)
    for r in all_results:
        acc_str = "DIVERGED" if r["diverged"] else f"{r['final_test_acc']:.4f}"
        print(
            f"{r['run_id']:<32} {r['arch']:<14} {r['optimizer']:<6} "
            f"{r['lr']:<8} {acc_str:<10} {r['wall_time_sec']:<10.1f}"
        )
    print("=" * 90)


# --- Plotting --------------------------------------------------------------

def plot_training_curves(all_results: list, out_path: Path):
    """Two-panel plot: train loss (left) and test accuracy (right), all 8 runs overlaid."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    for result in all_results:
        epochs = [row["epoch"] for row in result["epoch_history"]]
        train_losses = [row["train_loss"] for row in result["epoch_history"]]
        test_accs = [row["test_acc"] for row in result["epoch_history"]]

        label = f"{result['run_id'][:2]} {result['arch']}/{result['optimizer']}/lr={result['lr']}"
        if result["diverged"]:
            label += " [DIVERGED]"

        ax_loss.plot(epochs, train_losses, marker="o", label=label, alpha=0.8)
        ax_acc.plot(epochs, test_accs, marker="o", label=label, alpha=0.8)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train Loss")
    ax_loss.set_title("Training Loss per Epoch")
    ax_loss.set_yscale("log")  # makes the diverged/high-LR runs visible on same axis
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=7, loc="best")

    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Test Accuracy")
    ax_acc.set_title("Test Accuracy per Epoch")
    ax_acc.set_ylim(0, 1)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(fontsize=7, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Plot saved to {out_path}")


# --- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", type=str, default=None,
        help="Substring match on run_id to run a single config (e.g. '01' or 'cnn_adam')"
    )
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in experiments if args.only in e["run_id"]]
        if not experiments:
            print(f"No experiments match '--only {args.only}'")
            return
        print(f"Running {len(experiments)} experiment(s) matching '{args.only}'")

    all_results = []
    for i, config in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Starting {config['run_id']}")
        print(f"  arch={config['arch']}  opt={config['optimizer']}  lr={config['lr']}")
        result = train_one_config(config)
        all_results.append(result)

    # Write outputs
    results_path = Path("results.csv")
    write_results_csv(all_results, results_path)
    print(f"\nResults written to {results_path}")

    plot_path = Path("training_curves.png")
    plot_training_curves(all_results, plot_path)

    write_summary_table(all_results)


if __name__ == "__main__":
    main()