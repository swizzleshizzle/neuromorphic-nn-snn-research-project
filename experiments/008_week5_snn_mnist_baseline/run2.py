"""experiments/008_week5_snn_mnist_baseline/run.py

Phase 1, Step 1.2 (Saturday hands-on): first feedforward SNN on MNIST.
Rate-coded spike input, 784 -> 1000 -> 1000 -> 10, surrogate-gradient training.

Usage:
    python run2.py --config .\config.yaml
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn

from neuromorphic.config import load_config, parse_cli_overrides
from neuromorphic.tracking import ExperimentTracker
from neuromorphic.utils import set_seed, get_device


# ==============================================================================
# Spike encoder — rate coding via Bernoulli sampling
# ==============================================================================

def encode_rate(data: torch.Tensor, num_steps: int, gain: float = 1.0) -> torch.Tensor:
    """Rate-code a batch of normalized images as Bernoulli spike trains.

    Each pixel becomes an independent Bernoulli spike train over `num_steps` time
    steps. Probability of a spike at any time step = (gain * pixel_intensity),
    clipped to [0, 1].

    SCADA bridge: this is exactly how you'd simulate a discrete-event sensor
    that fires with probability proportional to an analog reading. Each scan
    cycle, draw fresh randomness; emit an event if the random draw is below
    the analog setpoint.

    Args:
        data: shape [batch_size, features], values in [0, 1].
        num_steps: number of time steps to simulate.
        gain: multiplier on pixel intensity before clipping. 1.0 = pixel
              value IS the firing probability.

    Returns:
        Tensor of shape [num_steps, batch_size, features], values in {0.0, 1.0}.
    """
    probs = torch.clamp(data * gain, min=0.0, max=1.0)
    probs_expanded = probs.unsqueeze(0).expand(num_steps, -1, -1)
    spikes = (torch.rand_like(probs_expanded) < probs_expanded).float()
    return spikes


# ==============================================================================
# Network — feedforward SNN, 784 -> 1000 -> 1000 -> 10
# ==============================================================================

class FeedforwardSNN(nn.Module):
    """Three-layer feedforward spiking neural network for MNIST.

    Each Linear layer is followed by a Leaky neuron (snn.Leaky).
    Forward pass iterates over `num_steps` time steps; returns spikes and
    membrane potentials from the output layer across all time steps.
    """

    def __init__(
        self,
        num_inputs: int = 784,
        hidden_dims: tuple = (1000, 1000),
        num_outputs: int = 10,
        beta: float = 0.95,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        num_steps: int = 25,
    ):
        super().__init__()
        self.num_steps = num_steps

        self.fc1 = nn.Linear(num_inputs, hidden_dims[0])
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)

        self.fc3 = nn.Linear(hidden_dims[1], num_outputs)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)

    def forward(self, spk_in: torch.Tensor):
        """Forward pass over time.

        Args:
            spk_in: [num_steps, batch_size, num_inputs] — rate-coded input spikes.

        Returns:
            (spk_out_rec, mem_out_rec) — each [num_steps, batch_size, num_outputs]
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_out_rec = []
        mem_out_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(spk_in[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk_out_rec.append(spk3)
            mem_out_rec.append(mem3)

        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)


# ==============================================================================
# Helpers
# ==============================================================================

def get_dataloaders(batch_size: int, data_path: str = "./data"):
    """Standard MNIST loaders. Same transform as Tutorial 5."""
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])
    train_set = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


def compute_loss_over_time(mem_rec: torch.Tensor, targets: torch.Tensor,
                           loss_fn: nn.Module) -> torch.Tensor:
    """Sum cross-entropy loss over all time steps (Tutorial 5 scheme)."""
    loss_val = torch.zeros(1, device=mem_rec.device)
    for step in range(mem_rec.shape[0]):
        loss_val = loss_val + loss_fn(mem_rec[step], targets)
    return loss_val


def evaluate(net: nn.Module, loader: DataLoader, device: torch.device,
             encode_fn, num_steps: int, gain: float):
    """Full eval over a loader. Returns (loss, accuracy, all_preds, all_targets)."""
    net.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in loader:
            data = data.view(data.size(0), -1).to(device)
            targets = targets.to(device)

            spk_in = encode_fn(data, num_steps=num_steps, gain=gain)
            spk_out, mem_out = net(spk_in)

            batch_loss = 0.0
            for step in range(num_steps):
                batch_loss += loss_fn(mem_out[step], targets).item()
            total_loss += batch_loss

            _, predicted = spk_out.sum(dim=0).max(1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            all_preds.append(predicted.cpu())
            all_targets.append(targets.cpu())

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, torch.cat(all_preds), torch.cat(all_targets)


# ==============================================================================
# Main
# ==============================================================================

def main():
    # ---- Setup ----
    config_path, overrides = parse_cli_overrides()
    config = load_config(config_path, overrides)

    set_seed(config.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: {config.run_name} | epochs={config.epochs} | "
          f"batch={config.batch_size} | num_steps={config.num_steps}")

    # ---- Data ----
    train_loader, test_loader = get_dataloaders(config.batch_size, config.data_root)
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    # ---- Model + optimizer + loss ----
    net = FeedforwardSNN(
        num_inputs=config.num_inputs,
        hidden_dims=tuple(config.hidden_dims),
        num_outputs=config.num_outputs,
        beta=config.beta,
        threshold=config.threshold,
        reset_mechanism=config.reset_mechanism,
        num_steps=config.num_steps,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()

    # ---- Tracking ----
    tracker = ExperimentTracker(config)

    # ---- Training loop ----
    loss_hist = []
    test_loss_hist = []
    test_acc_hist = []
    global_step = 0

    for epoch in range(config.epochs):
        net.train()
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(device)
            targets = targets.to(device)

            # 1. Encode pixels as rate-coded spikes
            spk_in = encode_rate(data, num_steps=config.num_steps, gain=config.gain)

            # 2. Forward pass through SNN
            spk_out, mem_out = net(spk_in)

            # 3. Loss summed over time
            loss_val = compute_loss_over_time(mem_out, targets, loss_fn)

            # 4. Backward + step
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            loss_hist.append(loss_val.item())
            tracker.log_metric("train/loss", loss_val.item(), step=global_step)

            # Periodic eval
            if global_step % config.log_interval == 0:
                net.eval()
                with torch.no_grad():
                    _, train_pred = spk_out.sum(dim=0).max(1)
                    train_acc = (train_pred == targets).float().mean().item()

                    test_data, test_targets = next(iter(test_loader))
                    test_data = test_data.view(test_data.size(0), -1).to(device)
                    test_targets = test_targets.to(device)
                    test_spk_in = encode_rate(test_data, num_steps=config.num_steps,
                                              gain=config.gain)
                    test_spk_out, test_mem_out = net(test_spk_in)
                    test_loss = compute_loss_over_time(test_mem_out, test_targets,
                                                       loss_fn).item()
                    _, test_pred = test_spk_out.sum(dim=0).max(1)
                    test_acc = (test_pred == test_targets).float().mean().item()

                test_loss_hist.append(test_loss)
                test_acc_hist.append(test_acc)
                tracker.log_metric("test/loss", test_loss, step=global_step)
                tracker.log_metric("train/accuracy", train_acc, step=global_step)
                tracker.log_metric("test/accuracy", test_acc, step=global_step)

                print(f"Epoch {epoch}, Iter {batch_idx} (step {global_step}) | "
                      f"train_loss={loss_val.item():6.2f}  test_loss={test_loss:6.2f}  "
                      f"train_acc={train_acc*100:5.2f}%  test_acc={test_acc*100:5.2f}%")
                net.train()

            global_step += 1

    # ---- Final eval on full test set ----
    print("\n========== Final Evaluation ==========")
    final_loss, final_acc, all_preds, all_targets = evaluate(
        net, test_loader, device, encode_rate, config.num_steps, config.gain
    )
    print(f"Final test loss: {final_loss:.3f}")
    correct = (all_preds == all_targets).sum().item()
    print(f"Final test accuracy: {final_acc*100:.2f}% ({correct}/{len(all_targets)})")

    tracker.log_metric("test/final_loss", final_loss, step=global_step)
    tracker.log_metric("test/final_accuracy", final_acc, step=global_step)

    # ---- Save checkpoint + history ----
    out_dir = Path(config.viz_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": net.state_dict(),
        "config": config.to_dict(),
        "loss_hist": loss_hist,
        "test_loss_hist": test_loss_hist,
        "test_acc_hist": test_acc_hist,
        "all_preds": all_preds,
        "all_targets": all_targets,
        "final_accuracy": final_acc,
    }, out_dir / "checkpoint.pt")
    print(f"\nCheckpoint saved to {out_dir / 'checkpoint.pt'}")

    if hasattr(tracker, "finish"):
        tracker.finish()
    elif hasattr(tracker, "close"):
        tracker.close()


if __name__ == "__main__":
    main()