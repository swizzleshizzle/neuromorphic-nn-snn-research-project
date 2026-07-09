"""EXP-027 Component A: cross-region decodability + concept geometry (encoder-only, per seed)."""

from __future__ import annotations

import torch

from neuromorphic.brain import Brain
from neuromorphic.training.pretrain import pretrain_sensory, enumerate_states, split_states
from neuromorphic.analysis.probes import (
    REGION_SIGNALS, region_rate_matrix, task_targets, ridge_probe, peraction_probe,
    shuffle_null, pca_reduce, participation_ratio, unit_importance, keepk_curve,
)


def characterize_seed(seed, *, grid_n=5, pretrain_epochs=200, lam=1e-2, T=32) -> dict:
    gen = torch.Generator().manual_seed(seed)
    brain = Brain(grid_n=grid_n, seed=seed)
    pretrain_sensory(brain.sensory, grid_n=grid_n, epochs=pretrain_epochs, seed=seed,
                     generator=torch.Generator().manual_seed(seed))
    # probe on the encoder's OWN held-out split (states it never trained on)
    tr, te = split_states(enumerate_states(grid_n), frac_heldout=0.2, seed=seed)
    Ttr, Tte = task_targets(tr, grid_n), task_targets(te, grid_n)

    regions = {}
    for region_key, (signal_key, width) in REGION_SIGNALS.items():
        Xtr = region_rate_matrix(brain, tr, region_key=region_key.split("_")[0],
                                 signal_key=signal_key, width=width, recall=False, T=T, generator=gen)
        Xte = region_rate_matrix(brain, te, region_key=region_key.split("_")[0],
                                 signal_key=signal_key, width=width, recall=False, T=T, generator=gen)
        disp = ridge_probe(Xtr, Ttr["displacement"], Xte, Tte["displacement"], lam=lam)["r2"]
        null = shuffle_null(lambda a, b, c, d: ridge_probe(a, b, c, d, lam=lam)["r2"],
                            Xtr, Ttr["displacement"], Xte, Tte["displacement"], n=10, seed=seed)
        pca = {}
        for k in (4, 8):
            if width > k:
                xtr_k, xte_k = pca_reduce(Xtr, Xte, k)
                pca[k] = ridge_probe(xtr_k, Ttr["displacement"], xte_k, Tte["displacement"], lam=lam)["r2"]
        act = peraction_probe(Xtr, Ttr["optimal_action"], Xte, Tte["optimal_action"])["mean_acc"]
        regions[region_key] = {"displacement_r2": disp, "displacement_null_hi": null["hi"],
                               "displacement_pca": pca, "optimal_action_acc": act}

    # concept geometry (aim 2A)
    Cte = region_rate_matrix(brain, te, region_key="sensory", signal_key="concept",
                             width=64, recall=False, T=T, generator=gen)
    Ctr = region_rate_matrix(brain, tr, region_key="sensory", signal_key="concept",
                             width=64, recall=False, T=T, generator=gen)
    order = unit_importance(Ctr, Ttr["displacement"], lam=lam)
    curve = keepk_curve(Ctr, Ttr["displacement"], Cte, Tte["displacement"],
                        order=order, ks=[1, 2, 4, 8, 16, 32, 64], lam=lam)
    return {"seed": seed, "regions": regions,
            "geometry": {"participation_ratio": participation_ratio(Cte), "keepk": curve}}
