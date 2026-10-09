"""THROWAWAY spike (EXP-075 motivation, 2026-10-09): is the 192-unit readout what caps the judge,
or the value-iteration training signal? Fits a head on TRUE distances (an instrument, never the
method) over seed 0's frozen E1 wide readout (concept 64 + hidden 128 mean rates, one Poisson draw)
and drives the J3V search at depth 9 seed 0, the cell already disclosed in EXP-074 spec section 2.
Same training set, epochs and head as `oracle_search.py 300000 8 walk 0 9` (raw facelets, 88/200),
which is re-run here as `raw` to confirm it reproduces.

Variants: raw; wide (E1, 1 draw); wide8 (E1, 8 draws averaged, train and search); rand128 and
rand512 (a frozen RANDOM-init region at hidden 128 or 512, concept 64, readout concept+hidden).

Usage: PYTHONPATH=src .venv/bin/python -u experiments/074_wide_judge/diagnostics/oracle_wide.py raw,wide
"""
import importlib.util, random, sys, time, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
REPO = Path("/root/projects/neuromorphic-nn-snn-research-project")
def load(name, p):
    s = importlib.util.spec_from_file_location(name, REPO/p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cells = load("c73", "experiments/073_learned_judge/cells.py")
gate_l = load("gl74", "experiments/074_wide_judge/gate_l.py")
from neuromorphic.envs.cube import SOLVED, apply_move, N_ACTIONS
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training import lookahead as la
from neuromorphic.training import value_iteration as vi
from neuromorphic.training.cube_baseline import make_agent
from neuromorphic.regions.sensory_cortex import SensoryCortex
torch.set_num_threads(2); torch.manual_seed(0)
prov = ExactBFSDistance(max_depth=None)
WHICH = sys.argv[1].split(",")
NTR, EP, SEED, DEPTH = 300000, 8, 0, 9
excl = set()
for s in range(12): excl |= cells.exclusion_set(s, prov)
rng = random.Random(1); train = []
while len(train) < NTR:
    x = SOLVED
    for _ in range(rng.randint(1, 14)): x = apply_move(x, rng.randrange(N_ACTIONS))
    if x not in excl: train.append(x)
y = torch.tensor([prov.distance(x) for x in train], dtype=torch.float32)
oh = lambda S: F.one_hot(torch.as_tensor(np.array(S), dtype=torch.long), 6).flatten(1).float()

brain = make_agent(cells.c70.published_config(7, SEED))
wide = vi.Judge(brain._encoder, brain.sensory, brain.T, brain.content, readout="wide")
for p in wide.parameters(): p.requires_grad_(False)
def make_feats(judge, draws=1):
    for p in judge.parameters(): p.requires_grad_(False)
    def feats(states, generator):
        out = []
        with torch.no_grad():
            for i in range(0, len(states), 2000):
                out.append(sum(judge._wide(states[i:i+2000], generator) for _ in range(draws)) / draws)
        return torch.cat(out)
    return feats
def rand_judge(h):
    return vi.Judge(brain._encoder, SensoryCortex(n_obs=144, hidden=h, concept=64, num_steps=brain.T, seed=SEED),
                    brain.T, 64, readout="wide")
FEATS = {"wide": lambda: make_feats(wide), "wide8": lambda: make_feats(wide, 8),
         "rand128": lambda: make_feats(rand_judge(128)), "rand512": lambda: make_feats(rand_judge(512))}

probe = cells.probe_set(SEED, prov)
for which in WHICH:
    t0 = time.time()
    if which == "raw":
        feat = lambda S, g: oh(S); X = oh(train)
    else:
        feat = FEATS[which](); X = feat(train, torch.Generator().manual_seed(5))
    print(f"[{which}] features {tuple(X.shape)} in {time.time()-t0:.0f}s", flush=True)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(EP):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 512):
            b = perm[i:i+512]; loss = F.mse_loss(net(X[b]).squeeze(-1), y[b]); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    class J(torch.nn.Module):
        def forward(self, states, generator):
            with torch.no_grad(): return net(feat(states, generator)).squeeze(-1)
    j = J()
    gl = gate_l.leaf_rank_margin(j, probe, prov.distance, SEED, N_ACTIONS)
    print(f"[{which}] Gate L margin (seed {SEED} probe) {gl['margin']:.4f}", flush=True)

    def values(agent, critic, states, *, generator):
        with torch.no_grad(): return -j(states, generator)
    la.imagined_values = values
    agent, head, states, ts = cells.c70.load_cell(DEPTH, SEED)
    res = la.evaluate_lookahead(agent, head, states, depth=DEPTH, mode="C", k=3,
                                generator=torch.Generator().manual_seed(ts), rng_seed=ts, imag_seed=ts,
                                critic=lambda c: c, no_revisit=True)
    print(f"[{which}] d{DEPTH} s{SEED}: oracle-J3V solved {res['solved']}/{res['n']}  ({time.time()-t0:.0f}s)", flush=True)
