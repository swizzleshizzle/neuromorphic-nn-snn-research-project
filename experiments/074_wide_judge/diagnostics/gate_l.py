"""Calibrate the proposed Gate L on PILOT judges and pilot-seed probe states only: fraction of probe
states (true distance 7..11) whose lowest-J leaf of the 3-move tree is closer than the root, vs chance."""
import importlib.util, sys
from pathlib import Path
import numpy as np, torch
REPO = Path("/root/projects/neuromorphic-nn-snn-research-project")
s = importlib.util.spec_from_file_location("c", REPO/"experiments/073_learned_judge/cells.py"); cells = importlib.util.module_from_spec(s); s.loader.exec_module(cells)
from neuromorphic.envs.cube import N_ACTIONS
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.lookahead import tree_levels
torch.set_num_threads(2)
prov = ExactBFSDistance(max_depth=None)
for tag in sys.argv[1:]:
    arm, ps = tag.split("_s"); seed = int(ps)
    judge = cells.make_judge(seed); judge.load_state_dict(torch.load(f"/root/scratch/exp073pilot/judge_{tag}/judge.pt", map_location="cpu")); judge.eval()
    probe = [(x, d) for x, d in cells.probe_set(seed, prov) if 7 <= d <= 11]
    g = torch.Generator().manual_seed(seed)
    by = {}
    for i, (x, d) in enumerate(probe):
        leaves = tree_levels(x, 3, N_ACTIONS)[3]
        with torch.no_grad(): j = judge(leaves, g)
        ld = [prov.distance(l) for l in leaves]
        o = by.setdefault(d, [[], []]); o[0].append(int(ld[int(j.argmin())] < d)); o[1].append(np.mean([l < d for l in ld]))
    allh = [h for v in by.values() for h in v[0]]; allc = [c for v in by.values() for c in v[1]]
    print(tag, "all7-11 hit %.3f chance %.3f" % (np.mean(allh), np.mean(allc)), " ".join("d%d %.2f/%.2f" % (d, np.mean(v[0]), np.mean(v[1])) for d, v in sorted(by.items())), flush=True)
