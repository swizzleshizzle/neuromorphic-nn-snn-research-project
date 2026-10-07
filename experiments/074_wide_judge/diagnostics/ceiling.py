"""THROWAWAY spike: supervised ceilings for ordering distances 7-11 (true distance as the label,
which the real method never sees). Seed 12 (pilot seed), E1 encoder, probe set held out."""
import importlib.util, random, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
REPO = Path("/root/projects/neuromorphic-nn-snn-research-project")
spec = importlib.util.spec_from_file_location("c", REPO/"experiments/073_learned_judge/cells.py")
cells = importlib.util.module_from_spec(spec); spec.loader.exec_module(cells)
from neuromorphic.envs.cube import SOLVED, apply_move, N_ACTIONS
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training import value_iteration as vi
torch.set_num_threads(2); torch.manual_seed(0)
prov = ExactBFSDistance(max_depth=None)
SEED = 12
probe = cells.probe_set(SEED, prov); pset = {s for s, _ in probe}
rng = random.Random(1)
train = []
import sys
NTR = int(sys.argv[1]); WHICH = sys.argv[2].split(",")
while len(train) < NTR:
    s = SOLVED
    for _ in range(rng.randint(1, 14)): s = apply_move(s, rng.randrange(N_ACTIONS))
    if s not in pset: train.append(s)
ytr = torch.tensor([prov.distance(s) for s in train], dtype=torch.float32)
yte = [d for _, d in probe]; te = [s for s, _ in probe]
print("train distance histogram", np.bincount(ytr.int().numpy()).tolist())
judge = cells.make_judge(SEED); sens = judge.sensory
g = torch.Generator().manual_seed(5)
def spiking_feats(states, draws=1):
    out = []
    with torch.no_grad():
        for i in range(0, len(states), 2000):
            obs = torch.as_tensor(np.array(states[i:i+2000]), dtype=torch.long)
            acc = 0
            for _ in range(draws):
                spk = judge.encoder_fn(obs, T=judge.T, generator=g)
                m1 = m2 = None; h = []; c = []
                for t in range(spk.shape[0]):
                    s1, m1 = sens.lif1(sens.fc1(spk[t]), m1); s2, m2 = sens.lif2(sens.fc2(s1), m2)
                    h.append(s1); c.append(s2)
                acc = acc + torch.cat([torch.stack(c).mean(0), torch.stack(h).mean(0)], 1)
            out.append(acc / draws)
    return torch.cat(out)
def onehot(states):
    return F.one_hot(torch.as_tensor(np.array(states), dtype=torch.long), 6).flatten(1).float()
if any(w[0] in "ab" for w in WHICH):
    t0 = time.time(); Ftr = spiking_feats(train); Fte = spiking_feats(te); print("encode s", round(time.time()-t0))
Rtr, Rte = onehot(train), onehot(te)
def fit(Xtr, Xte, widths, epochs=40, name=""):
    torch.manual_seed(0)
    layers = []; d = Xtr.shape[1]
    for w in widths: layers += [nn.Linear(d, w), nn.ReLU()]; d = w
    net = nn.Sequential(*layers, nn.Linear(d, 1)); opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 512):
            b = perm[i:i+512]; loss = F.mse_loss(net(Xtr[b]).squeeze(-1), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): p = net(Xte).squeeze(-1).tolist(); ptr = net(Xtr[:5000]).squeeze(-1)
    sel = [i for i, d in enumerate(yte) if 7 <= d <= 11]
    means = [np.mean([p[i] for i in range(len(p)) if yte[i] == d]) for d in range(7, 12)]
    print(f"{name:42s} sp7-11 {vi.spearman([yte[i] for i in sel],[p[i] for i in sel]):.3f}  sp_all {vi.spearman(yte,p):.3f}  "
          f"trainRMSE {float(((ptr-ytr[:5000])**2).mean().sqrt()):.2f}  meanJ7..11 {' '.join('%.2f'%m for m in means)}")
EP = 8
if "a" in WHICH: fit(Ftr[:, :64], Fte[:, :64], [128], EP, name="(a) concept 64, head 128")
if "a2" in WHICH: fit(Ftr[:, :64], Fte[:, :64], [512, 512], EP, name="(a') concept 64, head 512x512")
if "b" in WHICH: fit(Ftr, Fte, [128], EP, name="(b) concept+hidden 192, head 128")
if "b2" in WHICH: fit(Ftr, Fte, [512, 512], EP, name="(b') concept+hidden 192, head 512x512")
if "c" in WHICH: fit(Rtr, Rte, [128], EP, name="(c) raw 144, head 128")
if "c2" in WHICH: fit(Rtr, Rte, [512, 512], EP, name="(c') raw 144, head 512x512")
if "c3" in WHICH: fit(Rtr, Rte, [1024, 1024, 512], EP, name="(c'') raw 144, head 1024x1024x512")
