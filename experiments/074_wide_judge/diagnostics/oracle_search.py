"""THROWAWAY spike: does a judge FITTED ON TRUE DISTANCES (raw facelets, 144->128->1 MLP) steer the
EXP-073 J3V search better than P3V at depth 9? Trained on random walks excluding every seed 0-11
evaluation held-out state. Diagnostic only; the real method never sees distances."""
import importlib.util, random, sys, time, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
REPO = Path("/root/projects/neuromorphic-nn-snn-research-project")
def load(name, p):
    s = importlib.util.spec_from_file_location(name, REPO/p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cells = load("c73", "experiments/073_learned_judge/cells.py")
from neuromorphic.envs.cube import SOLVED, apply_move, N_ACTIONS
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training import lookahead as la
torch.set_num_threads(2); torch.manual_seed(0)
prov = ExactBFSDistance(max_depth=None)
NTR, EP, BAL = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "bal"
seeds_eval = [int(x) for x in sys.argv[4].split(",")]
depths = [int(x) for x in sys.argv[5].split(",")]
excl = set()
for s in range(12): excl |= cells.exclusion_set(s, prov)
print("excluded", len(excl))
rng = random.Random(1); train = []
if BAL:
    # distance-balanced: random walks of length 1..14, then accept with prob inversely to shell frequency
    pool = []
    while len(pool) < NTR * 3:
        x = SOLVED
        for _ in range(rng.randint(1, 14)): x = apply_move(x, rng.randrange(N_ACTIONS))
        if x not in excl: pool.append(x)
    d = np.array([prov.distance(x) for x in pool]); cnt = np.bincount(d, minlength=15)
    w = 1.0 / cnt[d]; w /= w.sum(); idx = np.random.default_rng(0).choice(len(pool), NTR, p=w)
    train = [pool[i] for i in idx]
else:
    while len(train) < NTR:
        x = SOLVED
        for _ in range(rng.randint(1, 14)): x = apply_move(x, rng.randrange(N_ACTIONS))
        if x not in excl: train.append(x)
y = torch.tensor([prov.distance(x) for x in train], dtype=torch.float32)
print("train hist", np.bincount(y.int().numpy()).tolist())
oh = lambda S: F.one_hot(torch.as_tensor(np.array(S), dtype=torch.long), 6).flatten(1).float()
X = oh(train)
net = nn.Sequential(nn.Linear(144, 128), nn.ReLU(), nn.Linear(128, 1)); opt = torch.optim.Adam(net.parameters(), 1e-3)
for ep in range(EP):
    perm = torch.randperm(len(X))
    for i in range(0, len(X), 512):
        b = perm[i:i+512]; loss = F.mse_loss(net(X[b]).squeeze(-1), y[b]); opt.zero_grad(); loss.backward(); opt.step()
net.eval()
# probe ordering on seed-12 probe (pilot seed, disjoint from eval sets by construction of cells)
probe = cells.probe_set(12, prov); pd = [d for _, d in probe]
with torch.no_grad(): pj = net(oh([s for s, _ in probe])).squeeze(-1).tolist()
sel = [i for i, d in enumerate(pd) if 7 <= d <= 11]
from neuromorphic.training import value_iteration as vi
print("oracle probe sp7-11 %.3f" % vi.spearman([pd[i] for i in sel], [pj[i] for i in sel]),
      "meanJ7..11", [round(np.mean([pj[i] for i in sel if pd[i] == d]), 2) for d in range(7, 12)])
def oracle_values(agent, critic, states, *, generator):
    with torch.no_grad(): return -net(oh(states)).squeeze(-1)
la.imagined_values = oracle_values
for depth in depths:
    for seed in seeds_eval:
        t0 = time.time()
        agent, head, states, ts = cells.c70.load_cell(depth, seed)
        res = la.evaluate_lookahead(agent, head, states, depth=depth, mode="C", k=3,
                                    generator=torch.Generator().manual_seed(ts), rng_seed=ts, imag_seed=ts,
                                    critic=lambda c: c, no_revisit=True)
        p3v = json.load(open(REPO/f"experiments/072_p3v_frontier/outputs/exp072_P3V_d{depth}_s{seed}.json"))
        print(f"d{depth} s{seed}: oracle-J3V solved {res['solved']}/{res['n']}   published P3V {p3v['solved']}/{p3v['n']}   ({time.time()-t0:.0f}s)", flush=True)
