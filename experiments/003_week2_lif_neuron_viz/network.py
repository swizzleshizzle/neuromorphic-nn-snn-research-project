# network.py
# Week 2: A small network of LIF neurons

import torch
import snntorch as snn
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Simulation parameters
num_steps = 200
num_neurons = 15
beta = 0.95
threshold = 1.0

# Generate random weights — one per neuron
# We use a normal distribution centered at 0 with std 1.0
# Some will be positive (excitatory), some negative (inhibitory)
weights = torch.randn(num_neurons)
print(f"Weights: {weights}")

# Input signal — same constant input we used before
input_current = torch.ones(num_steps) * 0.1

# Create the layer of neurons (all identical structurally)
lif_layer = snn.Leaky(beta=beta, threshold=threshold)

# Initialize membrane potential for all 15 neurons at once
# This is a tensor of shape [15] — one mem value per neuron
mem = lif_layer.init_leaky()

# Storage — we'll record mem and spikes for all 15 neurons at every timestep
mem_history = torch.zeros(num_steps, num_neurons)
spk_history = torch.zeros(num_steps, num_neurons)

# The simulation loop
for step in range(num_steps):
    # Each neuron receives the input scaled by its own weight
    # This is the key line — broadcasting the single input across 15 neurons
    weighted_input = weights * input_current[step]
    
    # Feed the weighted inputs into the layer
    spk, mem = lif_layer(weighted_input, mem)
    
    # Record the state of all 15 neurons
    mem_history[step] = mem
    spk_history[step] = spk

# Summary stats
total_spikes_per_neuron = spk_history.sum(dim=0)
print(f"\nSpikes per neuron:")
for i, count in enumerate(total_spikes_per_neuron):
    print(f"  Neuron {i:2d} (weight={weights[i]:+.3f}): {int(count.item())} spikes")