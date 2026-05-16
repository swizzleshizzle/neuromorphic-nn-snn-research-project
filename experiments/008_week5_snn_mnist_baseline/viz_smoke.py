"""Render every neuromorphic.viz function against the week-5 SNN checkpoint.

This is the human-eyeball test mandated by the parent spec §7.2 — pytest
asserts the contract, this script asserts that the contract produces
visually sensible plots on real data.

Usage:
    python experiments/008_week5_snn_mnist_baseline/viz_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Make the existing run2.py importable so we can reconstruct the network.
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run2 import FeedforwardSNN, encode_rate  # noqa: E402

from neuromorphic.config import load_config  # noqa: E402
from neuromorphic.utils import get_device, set_seed  # noqa: E402
from neuromorphic.viz import (  # noqa: E402
    membrane_trace,
    population_rate,
    psth,
    spike_raster,
    training_curve,
    weight_heatmap,
    weight_histogram,
)


def _capture_full_forward(net, spk_in):
    """Mirror the helper from week-5 plots.py — record spikes from every layer."""
    net.eval()
    with torch.no_grad():
        mem1, mem2, mem3 = (net.lif1.init_leaky(),
                            net.lif2.init_leaky(),
                            net.lif3.init_leaky())
        s1, s2, s3, m3 = [], [], [], []
        for step in range(net.num_steps):
            cur1 = net.fc1(spk_in[step]); spk1, mem1 = net.lif1(cur1, mem1)
            cur2 = net.fc2(spk1);          spk2, mem2 = net.lif2(cur2, mem2)
            cur3 = net.fc3(spk2);          spk3, mem3 = net.lif3(cur3, mem3)
            s1.append(spk1); s2.append(spk2); s3.append(spk3); m3.append(mem3)
    return (torch.stack(s1), torch.stack(s2),
            torch.stack(s3), torch.stack(m3))


def main():
    config = load_config(HERE / "config.yaml")
    set_seed(config.seed)
    device = get_device()

    out_dir = HERE / "outputs" / "snn_mnist_baseline" / "viz_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = HERE / "outputs" / "snn_mnist_baseline" / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Week-5 checkpoint missing at {ckpt_path}. Run run2.py first."
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    net = FeedforwardSNN(
        num_inputs=config.num_inputs,
        hidden_dims=tuple(config.hidden_dims),
        num_outputs=config.num_outputs,
        beta=config.beta,
        threshold=config.threshold,
        reset_mechanism=config.reset_mechanism,
        num_steps=config.num_steps,
    ).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    print(f"Loaded week-5 net, final test acc was {ckpt['final_accuracy']*100:.2f}%")

    # ---- Fetch one batch of MNIST for the spike/membrane plots ----
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])
    test_set = datasets.MNIST(config.data_root, train=False, download=False,
                              transform=transform)
    loader = DataLoader(test_set, batch_size=8, shuffle=True)
    data, targets = next(iter(loader))
    data_flat = data.view(data.size(0), -1).to(device)
    spk_in = encode_rate(data_flat, num_steps=config.num_steps, gain=config.gain)
    spk1, spk2, spk3, mem3 = _capture_full_forward(net, spk_in)
    print(f"sample target={targets[0].item()}, "
          f"pred={spk3[:, 0, :].sum(dim=0).argmax().item()}")

    # ---- Render every plot ----
    plots = []

    fig, ax = spike_raster(spk_in, sample_idx=0)
    ax.set_title("input-layer raster (784 neurons, sample 0)")
    plots.append(("01_spike_raster_input.png", fig))

    fig, ax = spike_raster(spk1, sample_idx=0)
    ax.set_title("hidden-1 raster (1000 neurons, sample 0)")
    plots.append(("02_spike_raster_hidden1.png", fig))

    fig, ax = membrane_trace(mem3, sample_idx=0, threshold=config.threshold)
    ax.set_title(f"output membrane traces (target={targets[0].item()})")
    plots.append(("03_membrane_trace_output.png", fig))

    fig, ax = weight_histogram(net.fc1.weight)
    plots.append(("04_weight_histogram_fc1.png", fig))

    fig, ax = weight_heatmap(net.fc3.weight)
    plots.append(("05_weight_heatmap_fc3.png", fig))

    history = {
        "train_loss": ckpt["loss_hist"],
        "test_loss":  ckpt["test_loss_hist"],
        "test_acc":   ckpt["test_acc_hist"],
    }
    fig, ax = training_curve(history, log_interval=config.log_interval)
    ax.set_title("week-5 training history")
    plots.append(("06_training_curve.png", fig))

    fig, ax = population_rate(spk2, smoothing_window=3)
    ax.set_title("hidden-2 population rate")
    plots.append(("07_population_rate_hidden2.png", fig))

    fig, ax = psth(spk3, bin_size=5)
    ax.set_title("output-layer PSTH")
    plots.append(("08_psth_output.png", fig))

    for filename, fig in plots:
        path = out_dir / filename
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")

    print(f"\nAll 8 plots saved under {out_dir}/")


if __name__ == "__main__":
    main()
