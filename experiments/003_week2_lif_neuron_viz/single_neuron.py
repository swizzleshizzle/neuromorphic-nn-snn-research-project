# single_neuron.py
# Week 2: LIF Neuron Exploration

import torch
import snntorch as snn
import matplotlib.pyplot as plt

# Reproducibility
torch.manual_seed(42)

# Simulation parameters
num_steps = 200
beta = 0.95
threshold = 1.0

# Instantiate a single LIF neuron
# snn.Leaky is snnTorch's basic LIF implementation
lif_neuron = snn.Leaky(beta=beta, threshold=threshold)

# Initialize the neuron's state (membrane potential starts at zero)
mem = lif_neuron.init_leaky()

# We need an input pattern to feed in. Let's start simple:
# A constant input current of 0.1 for every timestep.
input_current = torch.ones(num_steps) * 0.1

# Storage for recording what happens across the simulation
mem_history = []   # membrane potential at each timestep
spk_history = []   # spikes at each timestep

# Run the simulation loop
for step in range(num_steps):
    # Feed this step's input into the neuron
    # spk is the output spike (0 or 1), mem is the updated membrane potential
    spk, mem = lif_neuron(input_current[step], mem)
    
    # Record for plotting
    mem_history.append(mem.item())
    spk_history.append(spk.item())

# Convert to tensors for easier plotting
mem_history = torch.tensor(mem_history)
spk_history = torch.tensor(spk_history)

print(f"Total spikes fired: {int(spk_history.sum().item())}")
print(f"Final membrane potential: {mem_history[-1].item():.4f}")

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Plot 1: Membrane potential over time
ax1.plot(mem_history, color='steelblue', linewidth=1.5)
ax1.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
ax1.set_ylabel('Membrane Potential')
ax1.set_title('LIF Neuron — Membrane Potential & Spikes (constant input = 0.1)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Spike raster
ax2.eventplot([torch.where(spk_history == 1)[0].numpy()], 
              colors='black', lineoffsets=0, linelengths=0.8)
ax2.set_ylabel('Spikes')
ax2.set_xlabel('Timestep')
ax2.set_yticks([])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('single_neuron_trace.png', dpi=100, bbox_inches='tight')
plt.show()