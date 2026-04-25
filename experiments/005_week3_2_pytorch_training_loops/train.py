"""
Week 3 Session 2 — single-config training function.

train_one_config(config) runs one experiment end-to-end:
seed -> model -> optimizer -> train loop -> eval -> save checkpoint -> return metrics.

All runs in the experiment matrix go through this function. That's the whole point:
if the matrix is a controlled comparison, only the config dict should differ between runs.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import get_mnist_loaders
from models import BaselineMLP, TinyMLPMNIST, SimpleCNN


# --- Lookup tables ---------------------------------------------------------
# Map config string names to actual classes/factories. Keeps configs serializable.

ARCH_REGISTRY = {
    "baseline_mlp": BaselineMLP,
    "tiny_mlp": TinyMLPMNIST,
    "simple_cnn": SimpleCNN,
}


def _build_optimizer(name: str, params, lr: float):
    """Return the requested optimizer. Keep the SGD momentum constant across configs
    so 'SGD vs Adam' is a clean comparison, not 'SGD+momentum=X vs Adam'."""
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name!r}")


# --- Metrics ---------------------------------------------------------------

@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return (loss, accuracy) on the given loader. No grad, model in eval mode."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")  # sum so we can average over samples
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


# --- The main training function --------------------------------------------

def train_one_config(config: dict) -> dict:
    """
    Run one complete training experiment described by `config`.

    Required config keys:
        run_id    : str, used for checkpoint filename and CSV row id
        arch      : str, one of ARCH_REGISTRY keys
        optimizer : str, 'sgd' or 'adam'
        lr        : float
        epochs    : int
        batch_size: int
        seed      : int

    Returns a dict with:
        run_id, arch, optimizer, lr, seed, device,
        final_train_loss, final_train_acc, final_test_loss, final_test_acc,
        wall_time_sec, diverged (bool),
        epoch_history: list of dicts (one per epoch) with train/test loss and accuracy.
    """
    # --- Seeding: first thing, before any model/optimizer construction ---
    # If you set the seed outside this function, run order affects results.
    # This is the experimental-discipline point of the whole session.
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    train_loader, test_loader = get_mnist_loaders(batch_size=config["batch_size"])

    # --- Model + optimizer + loss ---
    model_cls = ARCH_REGISTRY[config["arch"]]
    model = model_cls().to(device)
    optimizer = _build_optimizer(config["optimizer"], model.parameters(), config["lr"])
    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    epoch_history = []
    diverged = False
    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)

            # Catch divergence (NaN/Inf loss) without crashing — this is expected
            # for Config 4 (SGD at LR=1.0). Log it and break out of this config.
            if not torch.isfinite(loss):
                diverged = True
                break

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_samples += labels.size(0)

        if diverged:
            # Fill remaining epoch slots with NaN so the CSV shape is uniform
            for _ in range(epoch, config["epochs"] + 1):
                epoch_history.append({
                    "epoch": _,
                    "train_loss": float("nan"),
                    "train_acc": float("nan"),
                    "test_loss": float("nan"),
                    "test_acc": float("nan"),
                })
            break

        train_loss = running_loss / running_samples
        train_acc = running_correct / running_samples
        test_loss, test_acc = _evaluate(model, test_loader, device)

        epoch_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })

        print(
            f"  [{config['run_id']}] epoch {epoch}/{config['epochs']} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f}"
        )

    wall_time = time.time() - start_time

    # --- Checkpoint (skip if diverged — no meaningful weights to save) ---
    if not diverged:
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f"{config['run_id']}.pt"
        torch.save(model.state_dict(), ckpt_path)

    # --- Assemble result dict ---
    if diverged:
        final_train_loss = float("nan")
        final_train_acc = float("nan")
        final_test_loss = float("nan")
        final_test_acc = float("nan")
    else:
        final_train_loss = epoch_history[-1]["train_loss"]
        final_train_acc = epoch_history[-1]["train_acc"]
        final_test_loss = epoch_history[-1]["test_loss"]
        final_test_acc = epoch_history[-1]["test_acc"]

    result = {
        "run_id": config["run_id"],
        "arch": config["arch"],
        "optimizer": config["optimizer"],
        "lr": config["lr"],
        "seed": config["seed"],
        "device": str(device),
        "final_train_loss": final_train_loss,
        "final_train_acc": final_train_acc,
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
        "wall_time_sec": wall_time,
        "diverged": diverged,
        "epoch_history": epoch_history,
    }

    status = "DIVERGED" if diverged else f"test_acc={final_test_acc:.4f}"
    print(f"  [{config['run_id']}] done in {wall_time:.1f}s — {status}")

    return result


if __name__ == "__main__":
    # Quick smoke test: run the baseline for 1 epoch to confirm everything wires up.
    smoke_config = {
        "run_id": "smoke_test",
        "arch": "baseline_mlp",
        "optimizer": "sgd",
        "lr": 0.01,
        "epochs": 1,
        "batch_size": 64,
        "seed": 42,
    }
    result = train_one_config(smoke_config)
    print("\nSmoke test result:")
    for k, v in result.items():
        if k != "epoch_history":
            print(f"  {k}: {v}")