"""Single-variant trainer for the MNIST SNN optimization experiment.

Importable as ``train(config) -> dict`` for the run_all driver, or invoke
the CLI for ad-hoc single-variant runs:

    python run.py --config v1_mlp_tuned.yaml
"""

from __future__ import annotations

import json
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
from neuromorphic.viz import training_curve

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from models import FeedforwardSNN, SpikingCNN, encode_rate  # noqa: E402


# ---- internal helpers ----------------------------------------------------

def _build_model(config: ExperimentConfig) -> nn.Module:
    if config.arch == "feedforward_snn":
        return FeedforwardSNN(
            num_inputs=config.num_inputs,
            hidden_dims=tuple(config.hidden_dims),
            num_outputs=config.num_outputs,
            beta=config.beta,
            threshold=config.threshold,
            reset_mechanism=config.reset_mechanism,
            num_steps=config.num_steps,
        )
    if config.arch == "spiking_cnn":
        return SpikingCNN(
            num_outputs=config.num_outputs,
            beta=config.beta,
            threshold=config.threshold,
            reset_mechanism=config.reset_mechanism,
            num_steps=config.num_steps,
        )
    raise ValueError(f"Unknown arch: {config.arch!r}")


def _get_dataloaders(batch_size: int, data_root: str):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])
    train_set = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


def _prep_input(data: torch.Tensor, arch: str) -> torch.Tensor:
    """MLP wants ``[B, 784]``; CNN wants ``[B, 1, 28, 28]`` (untouched)."""
    if arch == "feedforward_snn":
        return data.view(data.size(0), -1)
    return data


def _full_test_eval(net, loader, device, num_steps, gain, arch) -> tuple[float, float]:
    """Returns ``(avg_loss, accuracy)`` over the full loader."""
    net.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, targets in loader:
            data = _prep_input(data.to(device), arch)
            targets = targets.to(device)
            spk_in = encode_rate(data, num_steps=num_steps, gain=gain)
            spk_out, mem_out = net(spk_in)
            for step in range(num_steps):
                total_loss += loss_fn(mem_out[step], targets).item()
            _, predicted = spk_out.sum(dim=0).max(1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
    return total_loss / len(loader), total_correct / total_samples


# ---- public API ----------------------------------------------------------

def train(config: ExperimentConfig) -> dict:
    """Train one variant. Returns a metrics dict and writes per-variant artifacts.

    Side effects (in order):
        1. Builds model from ``config.arch``.
        2. Starts an ExperimentTracker (W&B + TensorBoard per the config).
        3. Trains for ``config.epochs``, timed with ``time.perf_counter``.
        4. Full-test-set eval.
        5. Saves ``checkpoint.pt`` to ``config.viz_output_dir``.
        6. Renders ``training_curve.png`` via ``neuromorphic.viz.training_curve``.
        7. Closes the tracker and returns.

    Returns:
        Dict with keys: variant, arch, num_params, num_steps, beta, epochs,
        train_seconds, final_test_acc, checkpoint_path.
    """
    set_seed(config.seed)
    device = get_device()

    out_dir = Path(config.viz_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = _get_dataloaders(config.batch_size, config.data_root)

    net = _build_model(config).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[{config.run_name}] arch={config.arch} | params={n_params:,} | "
          f"device={device} | epochs={config.epochs} | beta={config.beta} | "
          f"num_steps={config.num_steps}")

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    tracker = ExperimentTracker(config).start()

    loss_hist: list[float] = []
    test_loss_hist: list[float] = []
    test_acc_hist: list[float] = []
    global_step = 0
    start_time = time.perf_counter()

    for epoch in range(config.epochs):
        net.train()
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = _prep_input(data.to(device), config.arch)
            targets = targets.to(device)

            spk_in = encode_rate(data, num_steps=config.num_steps, gain=config.gain)
            spk_out, mem_out = net(spk_in)

            loss_val = torch.zeros(1, device=device)
            for step in range(config.num_steps):
                loss_val = loss_val + loss_fn(mem_out[step], targets)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            loss_hist.append(loss_val.item())
            tracker.log_metric("train/loss", loss_val.item(), step=global_step)

            if global_step % config.log_interval == 0:
                net.eval()
                with torch.no_grad():
                    td, tt = next(iter(test_loader))
                    td = _prep_input(td.to(device), config.arch)
                    tt = tt.to(device)
                    tsk_in = encode_rate(td, num_steps=config.num_steps, gain=config.gain)
                    tsk_out, tm_out = net(tsk_in)
                    test_loss = sum(
                        loss_fn(tm_out[s], tt).item() for s in range(config.num_steps)
                    )
                    _, test_pred = tsk_out.sum(dim=0).max(1)
                    test_acc = (test_pred == tt).float().mean().item()
                test_loss_hist.append(test_loss)
                test_acc_hist.append(test_acc)
                tracker.log_metrics(
                    {"test/loss": test_loss, "test/accuracy": test_acc},
                    step=global_step,
                )
                print(f"  epoch {epoch} iter {batch_idx} (step {global_step}): "
                      f"train_loss={loss_val.item():.2f} | "
                      f"test_acc={test_acc*100:.1f}%")
                net.train()

            global_step += 1

    train_seconds = time.perf_counter() - start_time

    final_loss, final_acc = _full_test_eval(
        net, test_loader, device, config.num_steps, config.gain, config.arch
    )
    print(f"[{config.run_name}] DONE: final_test_acc={final_acc*100:.2f}% | "
          f"train_time={train_seconds:.1f}s")
    tracker.log_metric("test/final_accuracy", final_acc, step=global_step)

    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": net.state_dict(),
            "config": config.to_dict(),
            "loss_hist": loss_hist,
            "test_loss_hist": test_loss_hist,
            "test_acc_hist": test_acc_hist,
            "final_accuracy": final_acc,
            "train_seconds": train_seconds,
        },
        ckpt_path,
    )

    history = {
        "train_loss": loss_hist,
        "test_loss": test_loss_hist,
        "test_acc": test_acc_hist,
    }
    fig, ax = training_curve(history, log_interval=config.log_interval)
    ax.set_title(f"{config.run_name} — final test acc {final_acc*100:.2f}%")
    fig.savefig(out_dir / "training_curve.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    tracker.finish()

    return {
        "variant": config.run_name,
        "arch": config.arch,
        "num_params": n_params,
        "num_steps": config.num_steps,
        "beta": config.beta,
        "epochs": config.epochs,
        "train_seconds": train_seconds,
        "final_test_acc": final_acc,
        "checkpoint_path": str(ckpt_path.resolve()),
    }


def main():
    config_path, overrides = parse_cli_overrides()
    config = load_config(config_path, overrides)
    metrics = train(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
