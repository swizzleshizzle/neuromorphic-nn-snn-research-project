"""Pilot judges, pilot seeds only: does the lowest-J child step closer? vs chance (fraction of closer children)."""
import importlib.util, sys
from pathlib import Path
import torch
REPO = Path("/root/projects/neuromorphic-nn-snn-research-project")
spec = importlib.util.spec_from_file_location("c", REPO/"experiments/073_learned_judge/cells.py")
cells = importlib.util.module_from_spec(spec); spec.loader.exec_module(cells)
from neuromorphic.envs.cube import apply_move, N_ACTIONS
from neuromorphic.envs.cube_distance import ExactBFSDistance
torch.set_num_threads(2)
prov = ExactBFSDistance(max_depth=None)
for tag in sys.argv[1:]:
    arm, s = tag.split("_s"); seed = int(s)
    probe = [(x, d) for x, d in cells.probe_set(seed, prov) if 7 <= d <= 11]
    judge = cells.make_judge(seed)
    judge.load_state_dict(torch.load(f"/root/scratch/exp073pilot/judge_{tag}/judge.pt", map_location="cpu"))
    g = torch.Generator().manual_seed(7)
    kids = [apply_move(x, a) for x, _ in probe for a in range(N_ACTIONS)]
    with torch.no_grad():
        v = judge(kids, g).view(len(probe), N_ACTIONS)
    out = {}
    for i, (x, d) in enumerate(probe):
        kd = [prov.distance(apply_move(x, a)) for a in range(N_ACTIONS)]
        hit = kd[int(v[i].argmin())] < d
        chance = sum(k < d for k in kd) / N_ACTIONS
        o = out.setdefault(d, [0, 0.0, 0]); o[0] += hit; o[1] += chance; o[2] += 1
    print(tag, "  ".join(f"d{d}: J-pick closer {h/n:.2f} vs chance {c/n:.2f}" for d, (h, c, n) in sorted(out.items())))
