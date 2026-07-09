"""EXP-027 Component A: cross-region decodability + concept geometry (encoder-only, per seed)."""

from __future__ import annotations

import torch

from neuromorphic.brain import Brain
from neuromorphic.training.pretrain import pretrain_sensory, enumerate_states, split_states
from neuromorphic.analysis.probes import (
    REGION_SIGNALS, all_region_rates, task_targets, ridge_probe, peraction_probe,
    shuffle_null, pca_reduce, participation_ratio, unit_importance, keepk_curve,
    dropk_curve, fraction_for_r2, single_unit_r2,
)


def characterize_seed(seed, *, grid_n=5, pretrain_epochs=200, lam=1e-2, T=32) -> dict:
    gen = torch.Generator().manual_seed(seed)
    brain = Brain(grid_n=grid_n, seed=seed)
    pretrain_sensory(brain.sensory, grid_n=grid_n, epochs=pretrain_epochs, seed=seed,
                     generator=torch.Generator().manual_seed(seed))
    # probe on the encoder's OWN held-out split (states it never trained on)
    tr, te = split_states(enumerate_states(grid_n), frac_heldout=0.2, seed=seed)
    Ttr, Tte = task_targets(tr, grid_n), task_targets(te, grid_n)

    # ONE forward pass per split -> every region's rate matrix is the SAME Poisson draw,
    # so the cross-region contrast and the concept geometry are on one shared pass.
    Rtr = all_region_rates(brain, tr, recall=False, T=T, generator=gen)
    Rte = all_region_rates(brain, te, recall=False, T=T, generator=gen)

    regions = {}
    for region_key, (signal_key, width) in REGION_SIGNALS.items():
        Xtr, Xte = Rtr[region_key], Rte[region_key]
        disp = ridge_probe(Xtr, Ttr["displacement"], Xte, Tte["displacement"], lam=lam)["r2"]
        null = shuffle_null(lambda a, b, c, d: ridge_probe(a, b, c, d, lam=lam)["r2"],
                            Xtr, Ttr["displacement"], Xte, Tte["displacement"], n=10, seed=seed)
        act = peraction_probe(Xtr, Ttr["optimal_action"], Xte, Tte["optimal_action"])["mean_acc"]
        act_null = shuffle_null(lambda a, b, c, d: peraction_probe(a, b, c, d)["mean_acc"],
                                Xtr, Ttr["optimal_action"], Xte, Tte["optimal_action"], n=10, seed=seed)
        # PCA-matched (k=4,8) neutralizes the region-width confound for BOTH targets.
        pca, act_pca = {}, {}
        for k in (4, 8):
            if width > k:
                xtr_k, xte_k = pca_reduce(Xtr, Xte, k)
                pca[k] = ridge_probe(xtr_k, Ttr["displacement"], xte_k, Tte["displacement"], lam=lam)["r2"]
                act_pca[k] = peraction_probe(xtr_k, Ttr["optimal_action"], xte_k,
                                             Tte["optimal_action"])["mean_acc"]
        regions[region_key] = {"displacement_r2": disp, "displacement_null_hi": null["hi"],
                               "displacement_pca": pca, "optimal_action_acc": act,
                               "optimal_action_null_hi": act_null["hi"], "optimal_action_pca": act_pca}

    # concept geometry (aim 2A) -- reuses the SAME concept matrix from the shared pass
    Ctr, Cte = Rtr["sensory"], Rte["sensory"]
    order = unit_importance(Ctr, Ttr["displacement"], lam=lam)
    ks = [1, 2, 4, 8, 16, 32, 64]
    keep = keepk_curve(Ctr, Ttr["displacement"], Cte, Tte["displacement"], order=order, ks=ks, lam=lam)
    drop = dropk_curve(Ctr, Ttr["displacement"], Cte, Tte["displacement"], order=order, ks=ks, lam=lam)
    full_r2 = ridge_probe(Ctr, Ttr["displacement"], Cte, Tte["displacement"], lam=lam)["r2"]
    frac90 = fraction_for_r2(keep, full_r2, n_units=Ctr.shape[1], frac=0.9)
    single = single_unit_r2(Ctr, Ttr["displacement"], Cte, Tte["displacement"], lam=lam)
    return {"seed": seed, "regions": regions,
            "geometry": {"participation_ratio": participation_ratio(Cte), "keepk": keep,
                         "dropk": drop, "frac_units_90pct_r2": frac90,
                         "single_unit_r2": single.tolist()}}
