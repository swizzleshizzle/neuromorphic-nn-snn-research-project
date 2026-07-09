import torch

from neuromorphic.analysis.probes import (
    optimal_action_targets, ridge_probe, peraction_probe, shuffle_null,
    pca_reduce, participation_ratio, unit_importance, keepk_curve,
)
from neuromorphic.brain import Brain
from neuromorphic.analysis.probes import region_rate_matrix, task_targets


def test_optimal_action_targets_known_cases():
    # actions: 0=up(0,-1) 1=right(1,0) 2=down(0,1) 3=left(-1,0); optimal reduces manhattan
    states = torch.tensor([[0, 0, 4, 4], [4, 4, 0, 0], [2, 2, 2, 0]])
    T = optimal_action_targets(states, grid_n=5)
    assert T.shape == (3, 4)
    # agent(0,0) goal(4,4): right and down reduce distance; up/left clip or increase
    assert T[0].tolist() == [0.0, 1.0, 1.0, 0.0]
    # agent(4,4) goal(0,0): up and left reduce
    assert T[1].tolist() == [1.0, 0.0, 0.0, 1.0]
    # agent(2,2) goal(2,0): only up reduces (goal directly above)
    assert T[2].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_ridge_probe_recovers_linear_signal_and_is_regularized():
    torch.manual_seed(0)
    N, M = 8, 200
    W = torch.randn(N, 2)
    X = torch.randn(M, N)
    Y = X @ W + 0.01 * torch.randn(M, 2)
    out = ridge_probe(X[:150], Y[:150], X[150:], Y[150:], lam=1e-2)
    assert out["r2"] > 0.95            # recovers the linear map
    assert out["weights"].shape == (N, 2)
    # ridge shrinks vs lambda=0 (OLS): larger lambda -> smaller weight norm
    small = ridge_probe(X[:150], Y[:150], X[150:], Y[150:], lam=1e-3)["weights"].norm()
    big = ridge_probe(X[:150], Y[:150], X[150:], Y[150:], lam=1e2)["weights"].norm()
    assert big < small


def test_peraction_probe_shape_and_range():
    torch.manual_seed(0)
    X = torch.randn(120, 16)
    Y = (torch.randn(120, 4) > 0).float()
    out = peraction_probe(X[:90], Y[:90], X[90:], Y[90:])
    assert out["acc"].shape == (4,)
    assert 0.0 <= float(out["mean_acc"]) <= 1.0


def test_peraction_probe_learns_linearly_separable_target():
    # Each action is a linear threshold on X -> a well-trained probe must beat the ~0.5 base rate.
    torch.manual_seed(0)
    N, M = 10, 400
    Wt = torch.randn(N, 4)
    X = torch.randn(M, N)
    Y = (X @ Wt > 0).float()
    out = peraction_probe(X[:300], Y[:300], X[300:], Y[300:], epochs=400)
    base_rate = torch.maximum(Y[300:].mean(dim=0), 1 - Y[300:].mean(dim=0)).mean()
    assert float(out["mean_acc"]) > 0.85            # substantially above chance
    assert float(out["mean_acc"]) > float(base_rate) + 0.2   # and above the majority baseline


def test_shuffle_null_band_is_near_chance():
    # A STRONG real linear signal: the true fit must be high AND the shuffle-null far below it.
    # (Independent noise would give R2~0 with or without permutation -> a vacuous test.)
    torch.manual_seed(0)
    N, M = 8, 240
    W = torch.randn(N, 2)
    X = torch.randn(M, N)
    Y = X @ W + 0.01 * torch.randn(M, 2)   # near-deterministic linear map
    def fit(a, b, c, d): return ridge_probe(a, b, c, d, lam=1e-2)["r2"]
    true_r2 = fit(X[:180], Y[:180], X[180:], Y[180:])
    null = shuffle_null(fit, X[:180], Y[:180], X[180:], Y[180:], n=10, seed=0)
    assert true_r2 > 0.9              # the real signal is recovered
    assert null["hi"] < 0.3           # permuting labels destroys it -> near chance
    assert null["hi"] < true_r2 - 0.6  # the null band sits far below the true fit


def test_pca_reduce_shapes_and_fits_on_train_only():
    torch.manual_seed(0)
    X = torch.randn(100, 6)
    Xtr, Xte = X[:70], X[70:]
    Ztr, Zte = pca_reduce(Xtr, Xte, k=3)
    assert Ztr.shape == (70, 3) and Zte.shape == (30, 3)   # reduced to k columns
    # the test projection must use the TRAIN mean + TRAIN components (PCA is fit on train)
    mu = Xtr.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Xtr - mu, full_matrices=False)
    comp = Vh[:3].T
    assert torch.allclose(Zte, (Xte - mu) @ comp, atol=1e-5)


def test_pca_reduce_preserves_low_rank_signal():
    # A rank-3 signal embedded in 12 dims: reducing to k>=rank must keep the ridge R2.
    torch.manual_seed(0)
    Z = torch.randn(300, 3)
    X = Z @ torch.randn(3, 12)          # intrinsically rank-3 features
    Y = Z @ torch.randn(3, 2) + 0.001 * torch.randn(300, 2)
    Xtr, Xte, Ytr, Yte = X[:220], X[220:], Y[:220], Y[220:]
    full = ridge_probe(Xtr, Ytr, Xte, Yte, lam=1e-3)["r2"]
    xtr3, xte3 = pca_reduce(Xtr, Xte, k=3)
    red = ridge_probe(xtr3, Ytr, xte3, Yte, lam=1e-3)["r2"]
    assert full > 0.9
    assert red > full - 0.05            # k == rank preserves the decodable signal


def test_participation_ratio_bounds():
    torch.manual_seed(0)
    iso = torch.randn(200, 10)                      # ~isotropic -> PR near 10
    rank1 = torch.randn(200, 1) @ torch.randn(1, 10)  # rank-1 -> PR near 1
    assert participation_ratio(iso) > 5.0
    assert participation_ratio(rank1) < 2.0


def test_keepk_curve_monotone_top_units():
    torch.manual_seed(0)
    N = 12
    X = torch.randn(200, N)
    Y = X[:, :3] @ torch.randn(3, 1)   # only first 3 units carry signal
    order = unit_importance(X[:150], Y[:150])
    curve = keepk_curve(X[:150], Y[:150], X[150:], Y[150:], order=order, ks=[1, 3, 12], lam=1e-2)
    r2 = {c["k"]: c["r2"] for c in curve}
    assert r2[3] > 0.8 and r2[3] >= r2[1] - 1e-6   # top-3 recover most signal


def test_region_rate_matrix_sensory_shape_and_state_dependent():
    brain = Brain(grid_n=5, seed=0)
    states = torch.tensor([[0, 0, 4, 4], [4, 4, 0, 0], [1, 2, 3, 0]])
    gen = torch.Generator().manual_seed(0)
    R = region_rate_matrix(brain, states, region_key="sensory", signal_key="concept",
                           width=64, recall=False, T=16, generator=gen)
    assert R.shape == (3, 64)
    assert not torch.allclose(R[0], R[1])   # different states -> different concept rate


def test_region_rate_matrix_zero_fills_bypassed_hippocampus():
    brain = Brain(grid_n=5, seed=0)
    states = torch.tensor([[0, 0, 4, 4], [2, 2, 1, 1]])
    R = region_rate_matrix(brain, states, region_key="hippocampus", signal_key="population",
                           width=150, recall=False, T=16, generator=torch.Generator().manual_seed(0))
    assert R.shape == (2, 150)
    assert torch.count_nonzero(R) == 0   # bypassed -> zero-filled, not a crash


def test_task_targets_shapes():
    states = torch.tensor([[0, 0, 4, 4], [4, 0, 0, 4]])
    t = task_targets(states, grid_n=5)
    assert t["displacement"].shape == (2, 2)
    assert t["optimal_action"].shape == (2, 4)
