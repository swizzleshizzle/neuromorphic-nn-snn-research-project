"""Smoke test — proves the scaffold works end-to-end.

What this verifies
------------------
- YAML config loads into the dataclass without errors
- Seed control works (set_seed)
- Device detection works (CUDA picked up if available, CPU otherwise)
- Model instantiation works (BaselineMLP from src/neuromorphic/models/)
- DataLoader works (MNIST downloads + iterates)
- Training loop runs one epoch end-to-end without crashing
- Tracker initializes (W&B + TensorBoard, with graceful fallback if not installed)
- Metrics get logged
- Checkpoint gets saved

If this script runs cleanly to "Smoke test passed.", the scaffold is healthy
and you can build real experiments on top of it.

Usage
-----
    # From the repo root, with the venv active:
    python experiments/001_smoke_test/run.py --config experiments/001_smoke_test/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure src/ is importable regardless of where we're invoked from.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neuromorphic.config import load_config, parse_cli_overrides
from neuromorphic.tracking import ExperimentTracker
from neuromorphic.utils import get_device, set_seed


# --- A minimal model defined inline for the smoke test only.
# Real models will live in src/neuromorphic/models/ once Phase 1 starts.
class BaselineMLP(nn.Module):
    def __init__(self, hidden_dims: list[int]) -> None:
        super().__init__()
        dims = [784] + hidden_dims + [10]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


def get_mnist_loaders(batch_size: int, data_root: str, num_workers: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        total_loss += criterion(logits, labels).item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    # --- Config ---
    config_path, overrides = parse_cli_overrides()
    config = load_config(config_path, overrides)

    print(f"=== Smoke Test: {config.run_name} ===")
    print(f"  arch={config.arch}  optimizer={config.optimizer}  lr={config.lr}  epochs={config.epochs}")

    set_seed(config.seed)
    device = get_device("auto")
    print(f"  device={device}")

    # --- Data ---
    train_loader, test_loader = get_mnist_loaders(
        batch_size=config.batch_size,
        data_root=config.data_root,
        num_workers=config.num_workers,
    )
    print(f"  train batches={len(train_loader)}  test batches={len(test_loader)}")

    # --- Model + optimizer ---
    model = BaselineMLP(hidden_dims=config.hidden_dims).to(device)
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    criterion = nn.CrossEntropyLoss()

    # --- Tracker ---
    with ExperimentTracker(config) as tracker:
        tracker.log_config()

        # --- Training loop ---
        global_step = 0
        for epoch in range(1, config.epochs + 1):
            model.train()
            running_loss, running_correct, running_samples = 0.0, 0, 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                running_correct += (logits.argmax(dim=1) == labels).sum().item()
                running_samples += labels.size(0)
                global_step += 1

                if batch_idx % 100 == 0:
                    tracker.log_metric("train_loss_batch", loss.item(), step=global_step)

            train_loss = running_loss / running_samples
            train_acc = running_correct / running_samples
            test_loss, test_acc = evaluate(model, test_loader, device)

            print(
                f"  epoch {epoch}/{config.epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}"
            )
            tracker.log_metrics({
                "train_loss_epoch": train_loss,
                "train_acc_epoch": train_acc,
                "test_loss_epoch": test_loss,
                "test_acc_epoch": test_acc,
            }, step=epoch)

        # --- Checkpoint ---
        ckpt_dir = Path(config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{config.run_name}.pt"
        torch.save({
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": config.epochs,
            "test_acc": test_acc,
        }, ckpt_path)
        print(f"  checkpoint saved to {ckpt_path}")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
