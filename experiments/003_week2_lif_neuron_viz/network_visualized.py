# network_visualized.py
# Week 2: 15-neuron LIF network with full visualization
# Shows: input-to-neuron connections, membrane potential traces, and spike raster

import torch
import snntorch as snn
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

# ---- Simulation parameters ----
num_steps = 200
num_neurons = 15
beta = 0.95
threshold = 1.0

# ---- Weights: one per neuron, from a normal distribution ----
weights = torch.randn(num_neurons)
print(f"Weights: {weights}\n")

# ---- Input: constant current ----
input_current = torch.ones(num_steps) * 0.1

# ---- Build the layer ----
lif_layer = snn.Leaky(beta=beta, threshold=threshold)
mem = lif_layer.init_leaky()

# ---- Storage ----
mem_history = torch.zeros(num_steps, num_neurons)
spk_history = torch.zeros(num_steps, num_neurons)

# ---- Simulation loop ----
for step in range(num_steps):
    weighted_input = weights * input_current[step]
    spk, mem = lif_layer(weighted_input, mem)
    mem_history[step] = mem
    spk_history[step] = spk

# ---- Summary ----
total_spikes_per_neuron = spk_history.sum(dim=0)
print("Spikes per neuron:")
for i, count in enumerate(total_spikes_per_neuron):
    print(f"  Neuron {i:2d} (weight={weights[i]:+.3f}): {int(count.item())} spikes")

# ============================================================
# VISUALIZATION
# ============================================================

fig = plt.figure(figsize=(16, 10))

# Use gridspec for a custom layout
gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1],
                       hspace=0.35, wspace=0.25)

# ------------------------------------------------------------
# LEFT PANEL: Network Connection Diagram
# ------------------------------------------------------------
ax_net = fig.add_subplot(gs[:, 0])

# Position the input node on the left, neurons on the right
input_pos = (0.1, 0.5)
neuron_x = 0.75
neuron_y_positions = np.linspace(0.05, 0.95, num_neurons)

# Determine color and thickness of each connection based on weight
for i, (w, y) in enumerate(zip(weights.numpy(), neuron_y_positions)):
    # Color: red for inhibitory (negative), green for excitatory (positive)
    color = 'crimson' if w < 0 else 'forestgreen'
    # Transparency and thickness scale with absolute weight
    alpha = min(0.3 + abs(w) * 0.3, 1.0)
    linewidth = 0.5 + abs(w) * 1.5
    
    ax_net.plot([input_pos[0], neuron_x], [input_pos[1], y],
                color=color, alpha=alpha, linewidth=linewidth, zorder=1)

# Draw the input node
ax_net.scatter(*input_pos, s=800, c='steelblue', edgecolors='black',
               linewidths=2, zorder=3)
ax_net.text(input_pos[0], input_pos[1] - 0.08, 'INPUT\n(0.1)',
            ha='center', va='top', fontsize=10, fontweight='bold')

# Draw each neuron — size scales with how much it fired
max_spikes = max(total_spikes_per_neuron.max().item(), 1)
for i, y in enumerate(neuron_y_positions):
    spikes = total_spikes_per_neuron[i].item()
    w = weights[i].item()
    
    # Size reflects firing activity
    size = 200 + (spikes / max_spikes) * 800
    # Color reflects whether it fired at all
    if spikes > 0:
        color = 'gold'
        edge = 'darkorange'
    else:
        color = 'lightgray'
        edge = 'gray'
    
    ax_net.scatter(neuron_x, y, s=size, c=color, edgecolors=edge,
                   linewidths=1.5, zorder=3)
    ax_net.text(neuron_x + 0.08, y, f'N{i} ({int(spikes)})',
                va='center', fontsize=8)

ax_net.set_xlim(-0.05, 1.05)
ax_net.set_ylim(-0.05, 1.05)
ax_net.set_title('Network Topology\n'
                 '(green = excitatory, red = inhibitory)\n'
                 '(line thickness = |weight|, node size = spike count)',
                 fontsize=10)
ax_net.axis('off')

# ------------------------------------------------------------
# TOP-RIGHT: Membrane potential traces for all 15 neurons
# ------------------------------------------------------------
ax_mem = fig.add_subplot(gs[0, 1])

cmap = plt.cm.viridis
for i in range(num_neurons):
    color = cmap(i / num_neurons)
    ax_mem.plot(mem_history[:, i].numpy(), color=color, linewidth=0.8,
                alpha=0.8, label=f'N{i}' if total_spikes_per_neuron[i] > 0 else None)

ax_mem.axhline(y=threshold, color='red', linestyle='--', alpha=0.6,
               label=f'Threshold = {threshold}')
ax_mem.set_ylabel('Membrane Potential')
ax_mem.set_title('Membrane Potentials — All 15 Neurons')
ax_mem.grid(True, alpha=0.3)
ax_mem.legend(loc='upper right', fontsize=7, ncol=2)

# ------------------------------------------------------------
# BOTTOM-RIGHT: Spike raster plot
# ------------------------------------------------------------
ax_raster = fig.add_subplot(gs[1, 1], sharex=ax_mem)

# For each neuron, get the timesteps where it spiked
spike_events = []
for i in range(num_neurons):
    spike_times = torch.where(spk_history[:, i] == 1)[0].numpy()
    spike_events.append(spike_times)

ax_raster.eventplot(spike_events, colors='black', lineoffsets=range(num_neurons),
                    linelengths=0.8)
ax_raster.set_ylabel('Neuron Index')
ax_raster.set_xlabel('Timestep')
ax_raster.set_title('Spike Raster — All 15 Neurons')
ax_raster.set_yticks(range(num_neurons))
ax_raster.grid(True, alpha=0.3)

plt.suptitle('LIF Network — 15 Neurons, Random Weights, Constant Input',
             fontsize=13, fontweight='bold', y=0.995)
plt.savefig('network_visualization_next.png', dpi=100, bbox_inches='tight')
plt.show()