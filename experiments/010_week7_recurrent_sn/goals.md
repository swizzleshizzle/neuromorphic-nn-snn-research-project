NEUROMORPHIC PROJECT — Phase 1, Step 1.3 / Module L7

Week 7 — Recurrent Spiking Neurons, Sustained Activity, and Working Memory
Session 1 — 2026-05-18 (Mon): RLeaky API + Sustained Activity Experiment

Full session brief lives in the Obsidian note:
  300 Efforts/Active/Coding/Neuromorphic Development/Weekly Notes/
  week-07-recurrent-snn.md

Diagnostic — no training, no MNIST. The point is to *see* the dynamics with
your own eyes before scaling up to Phase 2 attractor-style memory.

Sections (each runnable on its own):

  section2_v_sweep.py
    Single snn.RLeaky neuron, beta=0.9. Sweep recurrent weight
    V in {0.0, 0.3, 0.6, 0.9, 1.2}. Drive with a 10-step above-threshold
    pulse, observe sustained activity after the pulse turns off.
    Maps the decay -> persistent -> runaway regime boundary.

  section2b_10neuron_population.py
    Same V sweep with N=10 RLeaky neurons (all_to_all=False) and small
    Gaussian noise on the input. Companion to section2 — faithful to the
    day's goal #3 ("10 neurons, no learning"). With no lateral coupling
    these are 10 independent copies, so the value-add is the
    population-level view of the regime boundary: at near-critical V some
    neurons sustain and others decay depending on the noise realization.

  section3_ff_vs_recurrent.py
    Same input pulse through snn.Leaky (no memory) and
    snn.RLeaky(V=0.9) (memory regime). Overlay membrane traces and
    spike rasters. Compute the post-pulse "memory amplification factor"
    (recurrent post-pulse spikes / feedforward post-pulse spikes).

  section4_rsynaptic_check.py
    Smoke check for snn.RSynaptic: confirm the (spk, syn, mem) three-state
    API runs end-to-end and qualitatively compare its post-pulse
    behaviour to RLeaky. No full sweep.

  run_all.py
    Runs all three sections and writes plots to outputs/.

All plots use the project viz toolkit (src/neuromorphic/viz/):
  membrane_trace, spike_raster.

Predict-before-execute: write your V-regime predictions in the Obsidian
note's "Section 2 — Mike's Observations" block *before* running
section2_v_sweep.py. The post-pulse spike count is the single best
diagnostic of "memory horizon" — count it for each V and log it in the
note.
