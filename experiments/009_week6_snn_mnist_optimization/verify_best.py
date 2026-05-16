"""Reload the best checkpoint and re-evaluate on the full MNIST test set.

Asserts the reproduced accuracy matches the value recorded in
``best_checkpoint.json`` within ±0.1%. Implements verification gate §9 #3
of the design spec.

Usage:
    python verify_best.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from neuromorphic.utils import get_device
from models import FeedforwardSNN, SpikingCNN, encode_rate

OUT_DIR = HERE / "outputs"
TOLERANCE = 0.001  # 0.1%


def _build_model_from_dict(cfg: dict) -> torch.nn.Module:
    if cfg["arch"] == "feedforward_snn":
        return FeedforwardSNN(
            num_inputs=cfg["num_inputs"],
            hidden_dims=tuple(cfg["hidden_dims"]),
            num_outputs=cfg["num_outputs"],
            beta=cfg["beta"],
            threshold=cfg["threshold"],
            reset_mechanism=cfg["reset_mechanism"],
            num_steps=cfg["num_steps"],
        )
    if cfg["arch"] == "spiking_cnn":
        return SpikingCNN(
            num_outputs=cfg["num_outputs"],
            beta=cfg["beta"],
            threshold=cfg["threshold"],
            reset_mechanism=cfg["reset_mechanism"],
            num_steps=cfg["num_steps"],
        )
    raise ValueError(f"Unknown arch in checkpoint: {cfg['arch']!r}")


def _prep_input(data: torch.Tensor, arch: str) -> torch.Tensor:
    return data.view(data.size(0), -1) if arch == "feedforward_snn" else data


def main():
    device = get_device()
    meta = json.loads((OUT_DIR / "best_checkpoint.json").read_text())
    expected_acc = float(meta["final_test_acc"])
    print(f"Best variant: {meta['variant']!r}, expected_acc={expected_acc*100:.2f}%")

    ckpt = torch.load(
        OUT_DIR / "best_checkpoint.pt", map_location=device, weights_only=False
    )
    cfg = ckpt["config"]

    net = _build_model_from_dict(cfg).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])
    test_set = datasets.MNIST(cfg["data_root"], train=False, download=False,
                              transform=transform)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"],
                             shuffle=False, drop_last=False)

    arch = cfg["arch"]
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data = _prep_input(data.to(device), arch)
            targets = targets.to(device)
            spk_in = encode_rate(data, num_steps=cfg["num_steps"], gain=cfg["gain"])
            spk_out, _ = net(spk_in)
            _, predicted = spk_out.sum(dim=0).max(1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    reloaded_acc = total_correct / total_samples
    diff = abs(reloaded_acc - expected_acc)
    print(f"Reloaded acc: {reloaded_acc*100:.2f}%  ({total_correct}/{total_samples})")
    print(f"Diff vs. recorded: {diff*100:.3f}%  (tolerance: {TOLERANCE*100:.1f}%)")
    if diff > TOLERANCE:
        raise SystemExit(
            f"VERIFY FAILED: reloaded {reloaded_acc:.4f} vs recorded "
            f"{expected_acc:.4f} (|diff|={diff:.4f} > {TOLERANCE:.4f})"
        )
    print("VERIFY OK")


if __name__ == "__main__":
    main()
