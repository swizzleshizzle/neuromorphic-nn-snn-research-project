# experiments/008_week5_snn_mnist_baseline/run.py
# (top of file — imports and encoder live up here together)

import torch
import torch.nn as nn
import snntorch as snn



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
    # Clip gain*pixel to [0, 1] so we have valid probabilities even if gain > 1
    probs = torch.clamp(data * gain, min=0.0, max=1.0)

    # Expand to time dimension: [batch, features] -> [num_steps, batch, features]
    # We use expand (no memory copy) then sample fresh randomness for each step.
    probs_expanded = probs.unsqueeze(0).expand(num_steps, -1, -1)

    # Sample fresh uniform noise of the same shape, compare to probs.
    # If uniform < prob, spike; else no spike. This is a Bernoulli sample.
    spikes = (torch.rand_like(probs_expanded) < probs_expanded).float()

    return spikes

class FeedforwardSNN(nn.Module):
    """Three-layer feedforward spiking neural network for MNIST.

    Architecture: 784 -> 1000 -> 1000 -> 10
    Each Linear layer is followed by a Leaky neuron (snn.Leaky).
    Forward pass iterates over `num_steps` time steps; returns spikes and
    membrane potentials from the output layer across all time steps.

    Returns from forward():
        spk_out_rec: [num_steps, batch_size, num_outputs] — spikes for prediction
        mem_out_rec: [num_steps, batch_size, num_outputs] — membrane for loss
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

        # Layer 1: input -> hidden1
        self.fc1 = nn.Linear(num_inputs, hidden_dims[0])
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)

        # Layer 2: hidden1 -> hidden2
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)

        # Layer 3: hidden2 -> output
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
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Record outputs from the output layer across all time steps
        spk_out_rec = []
        mem_out_rec = []

        for step in range(self.num_steps):
            # =========================================================
            # YOUR JOB: fill in the four lines of the forward pass.
            #
            # Pattern from Tutorial 5, extended to three layers:
            #   synapse -> neuron -> synapse -> neuron -> synapse -> neuron
            #
            # Variables you have to work with at each step:
            #   - spk_in[step]   : input spikes for this time step
            #                       shape [batch_size, num_inputs]
            #   - mem1, mem2, mem3 : membrane potentials (carry from prev step)
            #
            # Variables you need to produce:
            #   - cur1, cur2, cur3 : weighted currents (output of each fc layer)
            #   - spk1, spk2, spk3 : spikes from each neuron layer
            #   - updated mem1, mem2, mem3 (the snn.Leaky call returns new mem)
            #
            # Each spiking-layer call looks like:
            #   spk, mem = self.lifN(cur, mem)
            # =========================================================
            
            
            cur1 = self.fc1(spk_in[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk_out_rec.append(spk3)
            mem_out_rec.append(mem3)

        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)

if __name__ == "__main__":
    # quick smoke test of network forward pass
    net = FeedforwardSNN(num_steps=25)
    fake_spikes = torch.rand(25, 4, 784)  # [time, batch, features]
    fake_spikes = (fake_spikes < 0.2).float()  # 20% firing rate
    spk_out, mem_out = net(fake_spikes)
    print(f"spk_out shape: {spk_out.shape}")   # expect [25, 4, 10]
    print(f"mem_out shape: {mem_out.shape}")   # expect [25, 4, 10]
    print(f"total spikes in output: {spk_out.sum().item():.0f}")
    print(f"mean membrane potential: {mem_out.mean().item():.4f}")
    
     # diagnostic — figure out where spikes are dying
    net.eval()
    with torch.no_grad():
        # Re-run forward, instrumenting intermediate layers
        mem1 = net.lif1.init_leaky()
        mem2 = net.lif2.init_leaky()
        mem3 = net.lif3.init_leaky()
        spk1_total = spk2_total = spk3_total = 0
        mem1_max = mem2_max = mem3_max = -1e9

        for step in range(net.num_steps):
            cur1 = net.fc1(fake_spikes[step])
            spk1, mem1 = net.lif1(cur1, mem1)
            cur2 = net.fc2(spk1)
            spk2, mem2 = net.lif2(cur2, mem2)
            cur3 = net.fc3(spk2)
            spk3, mem3 = net.lif3(cur3, mem3)

            spk1_total += spk1.sum().item()
            spk2_total += spk2.sum().item()
            spk3_total += spk3.sum().item()
            mem1_max = max(mem1_max, mem1.max().item())
            mem2_max = max(mem2_max, mem2.max().item())
            mem3_max = max(mem3_max, mem3.max().item())

        print(f"\nLayer-by-layer diagnostics (over 25 steps, batch=4):")
        print(f"  lif1: total spikes = {spk1_total:.0f}, max membrane = {mem1_max:.3f}")
        print(f"  lif2: total spikes = {spk2_total:.0f}, max membrane = {mem2_max:.3f}")
        print(f"  lif3: total spikes = {spk3_total:.0f}, max membrane = {mem3_max:.3f}")