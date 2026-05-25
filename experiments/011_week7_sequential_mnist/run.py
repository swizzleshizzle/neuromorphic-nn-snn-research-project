"""Single-variant trainer for the sequential MNIST experiment (exp 011).

Importable as ``train(config) -> dict`` for the run_all driver, or invoke
the CLI for ad-hoc single-variant runs:

    python run.py --config recurrent.yaml
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neuromorphic.config import ExperimentConfig, load_config, parse_cli_overrides
from neuromorphic.tracking import ExperimentTracker
from neuromorphic.utils import get_device, set_seed
from neuromorphic.viz import spike_raster, training_curve

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from models import SequentialSNN  # noqa: E402


def _get_dataloaders(batch_size: int, data_root: str):
    transform = transforms.Compose([
        transforms.ToTensor(),                           # [1, 28, 28], values in [0, 1]
        transforms.Normalize((0,), (1,)),                # no-op normalize (kept for parity with weeks 5/6)
    ])
    train_set = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


def _full_test_eval(net, loader, device) -> tuple[float, float]:
    net.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            spk_out = net(data)
            logits = net.readout(spk_out)
            loss_sum += loss_fn(logits, target).item() * data.size(0)
            correct += (logits.argmax(dim=1) == target).sum().item()
            total += data.size(0)
    net.train()
    return loss_sum / total, correct / total


def _run_verification_gates(net, loader, device, expected_params: int) -> None:
    """Pre-training gates §8 #1, #2, #3 (spec docs/design/2026-05-23-sequential-mnist.md)."""
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)
    spk_out = net(data)
    assert spk_out.shape == (28, data.size(0), 10), (
        f"gate 1 (forward shape): got {tuple(spk_out.shape)}, want (28, {data.size(0)}, 10)"
    )

    n = sum(p.numel() for p in net.parameters())
    assert abs(n - expected_params) <= 100, (
        f"gate 2 (param count): got {n}, want ~{expected_params} (+/-100)"
    )

    logits = net.readout(spk_out)
    loss = nn.CrossEntropyLoss()(logits, target).item()
    target_loss = math.log(10)
    assert abs(loss - target_loss) <= 0.3, (
        f"gate 3 (initial loss): got {loss:.3f}, want {target_loss:.3f} +/- 0.3"
    )
    print(f"[gates] shape={tuple(spk_out.shape)} params={n:,} initial_loss={loss:.3f}")


def _render_hidden_raster(net, loader, device, save_path: Path) -> None:
    """One test image, render hidden-layer spikes across all 28 steps."""
    data, _ = next(iter(loader))
    image = data[:1].to(device)  # [1, 1, 28, 28]
    net.eval()
    hidden_rec = []
    with torch.no_grad():
        from models import row_encode
        sequence = row_encode(image)
        if net.recurrent:
            spk1, mem1 = net.lif1.init_rleaky()
        else:
            mem1 = net.lif1.init_leaky()
        for t in range(28):
            cur1 = net.fc1(sequence[t])
            if net.recurrent:
                spk1, mem1 = net.lif1(cur1, spk1, mem1)
            else:
                spk1, mem1 = net.lif1(cur1, mem1)
            hidden_rec.append(spk1)
    net.train()
    spk_hidden = torch.stack(hidden_rec, dim=0).cpu()  # [28, 1, 128]
    fig, ax = spike_raster(spk_hidden)
    variant = "recurrent" if net.recurrent else "feedforward"
    ax.set_title(f"{variant} hidden-layer spikes, 1 test image, T=28")
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def train(config: ExperimentConfig) -> dict:
    """Train one variant. Returns metrics dict + writes per-variant artifacts."""
    assert config.sequential and config.num_steps == 28, (
        f"config invariants: sequential={config.sequential}, num_steps={config.num_steps}"
    )

    set_seed(config.seed)
    device = get_device()

    train_loader, test_loader = _get_dataloaders(config.batch_size, config.data_root)

    net = SequentialSNN(
        num_inputs=config.num_inputs,
        hidden_size=config.hidden_size,
        num_outputs=config.num_outputs,
        beta=config.beta,
        threshold=config.threshold,
        reset_mechanism=config.reset_mechanism,
        num_steps=config.num_steps,
        readout_window=config.readout_window,
        recurrent=config.recurrent,
    ).to(device)

    expected_params = 21514 if config.recurrent else 5002
    _run_verification_gates(net, train_loader, device, expected_params)

    tracker = ExperimentTracker(config)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    loss_hist, test_loss_hist, test_acc_hist = [], [], []
    global_step = 0
    t0 = time.perf_counter()

    for epoch in range(config.epochs):
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            net.train()
            spk_out = net(data)
            logits = net.readout(spk_out)
            loss = loss_fn(logits, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            loss_hist.append(loss.item())
            tracker.log_metric("train/loss", loss.item(), step=global_step)

            if (i % config.log_interval) == 0:
                test_loss, test_acc = _full_test_eval(net, test_loader, device)
                test_loss_hist.append(test_loss)
                test_acc_hist.append(test_acc)
                tracker.log_metrics(
                    {"test/loss": test_loss, "test/acc": test_acc},
                    step=global_step,
                )
                print(f"epoch {epoch} iter {i:>4d}  train_loss={loss.item():.4f}  test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

            global_step += 1

    train_seconds = time.perf_counter() - t0
    final_test_loss, final_test_acc = _full_test_eval(net, test_loader, device)
    tracker.log_metric("test/final_accuracy", final_test_acc, step=global_step)

    out_dir = HERE / "outputs" / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save({
        "model_state": net.state_dict(),
        "config": config.to_dict(),
        "loss_hist": loss_hist,
        "test_loss_hist": test_loss_hist,
        "test_acc_hist": test_acc_hist,
        "final_accuracy": final_test_acc,
        "train_seconds": train_seconds,
        "num_params": sum(p.numel() for p in net.parameters()),
    }, checkpoint_path)

    history = {
        "train_loss": loss_hist,
        "test_loss": test_loss_hist,
        "test_acc": test_acc_hist,
    }
    fig, ax = training_curve(history, log_interval=config.log_interval)
    ax.set_title(f"{config.run_name} — final test acc {final_test_acc*100:.2f}%")
    fig.savefig(out_dir / "training_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    try:
        _render_hidden_raster(net, test_loader, device, out_dir / "hidden_raster.png")
    except Exception as e:
        print(f"hidden_raster.png render skipped: {e!r}")

    tracker.finish()

    return {
        "variant": config.run_name,
        "recurrent": config.recurrent,
        "num_params": sum(p.numel() for p in net.parameters()),
        "num_steps": config.num_steps,
        "hidden_size": config.hidden_size,
        "beta": config.beta,
        "epochs": config.epochs,
        "train_seconds": train_seconds,
        "final_test_acc": final_test_acc,
        "checkpoint_path": str(checkpoint_path),
    }


if __name__ == "__main__":
    config_path, overrides = parse_cli_overrides()
    config = load_config(config_path, overrides)
    metrics = train(config)
    print(metrics)
