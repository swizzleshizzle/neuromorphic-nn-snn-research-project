"""experiments/008_week5_snn_mnist_baseline/plots.py

Generate the four required visualizations from the saved checkpoint:
  1. Spike raster plot (input + hidden + output for one sample)
  2. Membrane potential traces (selected output neurons)
  3. Training loss curve
  4. Confusion matrix

Usage:
    python plots.py --config .\config.yaml
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn

from neuromorphic.config import load_config, parse_cli_overrides
from neuromorphic.utils import set_seed, get_device

# Import the network + encoder from the training script so we can rebuild
# the model from the saved state_dict. Adjust the import if your file is
# named differently.
from run2 import FeedforwardSNN, encode_rate


# ==============================================================================
# Plot helpers
# ==============================================================================

def plot_spike_raster(spk_in, spk1, spk2, spk3, target, prediction, save_path):
    """Four-panel raster: input spikes, two hidden layers, output spikes.

    Each panel is a [time x neuron] scatter where dots mark spikes.
    spk_*: tensors of shape [num_steps, num_neurons] (already batched out to one sample).
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    titles = [
        f"Input layer (rate-coded pixels) — 784 neurons",
        f"Hidden layer 1 — 1000 neurons (showing 200 most active)",
        f"Hidden layer 2 — 1000 neurons (showing 200 most active)",
        f"Output layer — 10 neurons (target={target}, predicted={prediction})",
    ]

    # For dense layers, show 200 most active neurons so plot is readable
    def select_active(spk_tensor, max_neurons=200):
        if spk_tensor.shape[1] <= max_neurons:
            return spk_tensor, np.arange(spk_tensor.shape[1])
        spike_counts = spk_tensor.sum(dim=0).cpu().numpy()
        top_idx = np.argsort(spike_counts)[-max_neurons:]
        return spk_tensor[:, top_idx], top_idx

    layers = [spk_in, spk1, spk2, spk3]
    max_per_layer = [200, 200, 200, 10]

    for ax, layer_spk, title, max_n in zip(axes, layers, titles, max_per_layer):
        selected, _ = select_active(layer_spk, max_n)
        times, neurons = torch.where(selected.cpu() > 0)
        ax.scatter(times.numpy(), neurons.numpy(), s=2, c="black")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Neuron index")
        ax.set_xlim(-0.5, layer_spk.shape[0] - 0.5)

    axes[-1].set_xlabel("Time step")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_membrane_traces(mem_out, target, prediction, save_path):
    """Output-layer membrane traces — one line per class.

    mem_out: [num_steps, num_classes]. Highlight target and prediction.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    num_steps, num_classes = mem_out.shape
    time = np.arange(num_steps)

    for c in range(num_classes):
        trace = mem_out[:, c].cpu().numpy()
        if c == target and c == prediction:
            color, lw, label = "green", 2.5, f"Class {c} (target ✓ predicted)"
        elif c == target:
            color, lw, label = "blue", 2.5, f"Class {c} (target)"
        elif c == prediction:
            color, lw, label = "red", 2.5, f"Class {c} (predicted, wrong)"
        else:
            color, lw, label = "gray", 0.8, None
        ax.plot(time, trace, color=color, linewidth=lw, label=label, alpha=0.9 if lw > 1 else 0.4)

    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Threshold")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Membrane potential $U[t]$")
    ax.set_title(f"Output-layer membrane traces (sample with target={target})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_loss_curve(loss_hist, test_loss_hist, log_interval, save_path):
    """Training and test loss curves over iterations."""
    fig, ax = plt.subplots(figsize=(12, 6))

    train_iters = np.arange(len(loss_hist))
    ax.plot(train_iters, loss_hist, color="C0", alpha=0.5, label="Train loss (per iter)")

    # Smooth train loss with rolling mean for readability
    window = max(1, len(loss_hist) // 50)
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(loss_hist, kernel, mode="valid")
        smoothed_iters = train_iters[window - 1:]
        ax.plot(smoothed_iters, smoothed, color="C0", linewidth=2, label=f"Train loss (rolling mean, window={window})")

    test_iters = np.arange(0, len(test_loss_hist) * log_interval, log_interval)
    ax.plot(test_iters, test_loss_hist, color="C1", linewidth=2, marker="o", markersize=4, label="Test loss (per eval)")

    ax.set_xlabel("Iteration (gradient step)")
    ax.set_ylabel("Loss (cross-entropy summed over time)")
    ax.set_title("Training & test loss curves")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_confusion_matrix(preds, targets, num_classes, save_path):
    """Confusion matrix as a heatmap with annotations."""
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets, preds):
        cm[t.item(), p.item()] += 1

    cm_np = cm.numpy()
    cm_normalized = cm_np.astype(float) / cm_np.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_normalized, cmap="Blues", aspect="auto")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix (test set, n={len(targets)})\n"
                 f"Diagonal = correct, off-diagonal = errors")

    # Annotate cells with counts; bold the diagonal
    for i in range(num_classes):
        for j in range(num_classes):
            count = cm_np[i, j]
            color = "white" if cm_normalized[i, j] > 0.5 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, str(count), ha="center", va="center",
                    color=color, fontsize=9, fontweight=weight)

    plt.colorbar(im, ax=ax, label="Row-normalized fraction")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")

    # Print per-class accuracy text summary
    print("\nPer-class accuracy:")
    for i in range(num_classes):
        per_class_acc = cm_np[i, i] / cm_np[i].sum() * 100
        print(f"  Class {i}: {per_class_acc:.2f}% ({cm_np[i, i]}/{cm_np[i].sum()})")


# ==============================================================================
# Capture intermediate activations from a single sample (for raster plot)
# ==============================================================================

def capture_full_forward(net, spk_in):
    """Run a single sample through the network, recording spikes from every layer.

    spk_in: [num_steps, 1, num_inputs]
    Returns: spk_in_flat, spk1_rec, spk2_rec, spk3_rec, mem3_rec
             each [num_steps, num_neurons_in_that_layer]
    """
    net.eval()
    with torch.no_grad():
        mem1 = net.lif1.init_leaky()
        mem2 = net.lif2.init_leaky()
        mem3 = net.lif3.init_leaky()

        spk1_rec, spk2_rec, spk3_rec, mem3_rec = [], [], [], []

        for step in range(net.num_steps):
            cur1 = net.fc1(spk_in[step])
            spk1, mem1 = net.lif1(cur1, mem1)
            cur2 = net.fc2(spk1)
            spk2, mem2 = net.lif2(cur2, mem2)
            cur3 = net.fc3(spk2)
            spk3, mem3 = net.lif3(cur3, mem3)

            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

    # Squeeze the batch dim (which is 1) to get [time, neurons]
    return (
        spk_in.squeeze(1),
        torch.stack(spk1_rec).squeeze(1),
        torch.stack(spk2_rec).squeeze(1),
        torch.stack(spk3_rec).squeeze(1),
        torch.stack(mem3_rec).squeeze(1),
    )


# ==============================================================================
# Main
# ==============================================================================

def main():
    config_path, overrides = parse_cli_overrides()
    config = load_config(config_path, overrides)

    set_seed(config.seed)
    device = get_device()

    out_dir = Path(config.viz_output_dir)
    checkpoint_path = out_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run training first.")

    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Rebuild network from config + state dict
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
    print(f"Loaded model, final test accuracy from training: {ckpt['final_accuracy']*100:.2f}%")

    # ---- Plot 3: Loss curve ----
    print("\nPlot 3 of 4: Training loss curve")
    plot_loss_curve(
        ckpt["loss_hist"],
        ckpt["test_loss_hist"],
        config.log_interval,
        out_dir / "03_loss_curve.png",
    )

    # ---- Plot 4: Confusion matrix ----
    print("\nPlot 4 of 4: Confusion matrix")
    plot_confusion_matrix(
        ckpt["all_preds"],
        ckpt["all_targets"],
        num_classes=config.num_outputs,
        save_path=out_dir / "04_confusion_matrix.png",
    )

    # ---- Plots 1 & 2: Need one sample run through the network ----
    # Get one test sample
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])
    test_set = datasets.MNIST(config.data_root, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    sample_data, sample_target = next(iter(test_loader))
    sample_data_flat = sample_data.view(1, -1).to(device)
    sample_target = sample_target.item()

    # Encode and capture full forward pass
    spk_in_one = encode_rate(sample_data_flat, num_steps=config.num_steps, gain=config.gain)
    spk_in_rec, spk1_rec, spk2_rec, spk3_rec, mem3_rec = capture_full_forward(net, spk_in_one)
    prediction = spk3_rec.sum(dim=0).argmax().item()

    print(f"\nSample sample target={sample_target}, predicted={prediction}")

    # ---- Plot 1: Spike raster ----
    print("\nPlot 1 of 4: Spike raster")
    plot_spike_raster(
        spk_in_rec, spk1_rec, spk2_rec, spk3_rec,
        target=sample_target, prediction=prediction,
        save_path=out_dir / "01_spike_raster.png",
    )

    # ---- Plot 2: Membrane traces ----
    print("\nPlot 2 of 4: Output membrane traces")
    plot_membrane_traces(
        mem3_rec, target=sample_target, prediction=prediction,
        save_path=out_dir / "02_membrane_traces.png",
    )

    print(f"\nAll 4 plots saved to {out_dir}/")


if __name__ == "__main__":
    main()