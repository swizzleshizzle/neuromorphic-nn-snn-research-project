# beta_sweep.py
# Week 2: Explore how beta changes neuron behavior

import torch
import snntorch as snn
import matplotlib.pyplot as plt

torch.manual_seed(42)

num_steps = 200
threshold = 1.0
input_current = torch.ones(num_steps) * 0.1

# Test these beta values
beta_values = [0.5, 0.8, 0.95, 0.99]

fig, axes = plt.subplots(len(beta_values), 1, figsize=(10, 10), sharex=True)

for idx, beta in enumerate(beta_values):
    lif_neuron = snn.Leaky(beta=beta, threshold=threshold)
    mem = lif_neuron.init_leaky()
    
    mem_history = []
    spk_history = []
    
    for step in range(num_steps):
        spk, mem = lif_neuron(input_current[step], mem)
        mem_history.append(mem.item())
        spk_history.append(spk.item())
    
    mem_history = torch.tensor(mem_history)
    spk_history = torch.tensor(spk_history)
    total_spikes = int(spk_history.sum().item())
    
    axes[idx].plot(mem_history, color='steelblue', linewidth=1.2)
    axes[idx].axhline(y=threshold, color='red', linestyle='--', alpha=0.6)
    axes[idx].set_ylabel(f'β = {beta}')
    axes[idx].set_title(f'β = {beta}  |  Spikes: {total_spikes}  |  Steady-state: {0.1/(1-beta):.2f}')
    axes[idx].grid(True, alpha=0.3)

axes[-1].set_xlabel('Timestep')
plt.tight_layout()
plt.savefig('beta_sweep.png', dpi=100, bbox_inches='tight')
plt.show()