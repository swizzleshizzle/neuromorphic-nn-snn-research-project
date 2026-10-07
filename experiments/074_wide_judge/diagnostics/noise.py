"""Per-state Poisson noise of trained pilot judges, and Spearman 7-11 when J is averaged over K draws."""
import importlib.util, sys, statistics as st
from pathlib import Path
import torch
REPO = Path("/root/projects/neuromorphic-nn-snn-research-project")
spec = importlib.util.spec_from_file_location("c", REPO/"experiments/073_learned_judge/cells.py")
cells = importlib.util.module_from_spec(spec); spec.loader.exec_module(cells)
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training import value_iteration as vi
torch.set_num_threads(2)
prov = ExactBFSDistance(max_depth=None)
K = 32
for tag in sys.argv[1:]:
    arm, s = tag.split("_s"); seed = int(s)
    probe = [(st_, d) for st_, d in cells.probe_set(seed, prov) if d >= 5]
    judge = cells.make_judge(seed)
    judge.load_state_dict(torch.load(f"/root/scratch/exp073pilot/judge_{tag}/judge.pt", map_location="cpu"))
    judge.eval()
    g = torch.Generator().manual_seed(999)
    states = [x for x, _ in probe]; dists = [d for _, d in probe]
    with torch.no_grad():
        V = torch.stack([judge(states, g) for _ in range(K)])  # K x n
    sd = V.std(dim=0)
    mean = V.mean(dim=0)
    print(tag)
    for d in range(5, 12):
        idx = [i for i, x in enumerate(dists) if x == d]
        print(f"  d{d}: mean J {mean[idx].mean():.2f}  per-state Poisson sd {sd[idx].mean():.3f}  between-state sd of K-avg J {mean[idx].std():.3f}")
    sel = [i for i, x in enumerate(dists) if 7 <= x <= 11]
    d711 = [dists[i] for i in sel]
    print(f"  spearman_7_11 single draw {vi.spearman(d711, V[0, sel].tolist()):.3f}   K={K} avg {vi.spearman(d711, mean[sel].tolist()):.3f}")
    # expected min-of-6 bias from Poisson noise alone at sd s: E[min of 6 N(0,1)] = -1.267
    print(f"  implied min-of-6 Poisson bias at d9: {1.267*sd[[i for i,x in enumerate(dists) if x==9]].mean():.3f} moves")
