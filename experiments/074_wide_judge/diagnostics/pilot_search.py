"""THROWAWAY spike: EXP-073 PILOT judges (self-trained, seeds 12/13, no answers) driving J3V on
EVALUATION cells. Caveat: pilot training excluded only its own seed's held-out sets, so some
evaluation states may have been seen in random walks (no labels were ever used)."""
import importlib.util, sys, time, json
from pathlib import Path
import torch
REPO = Path("/root/projects/neuromorphic-nn-snn-research-project")
s = importlib.util.spec_from_file_location("c73", REPO/"experiments/073_learned_judge/cells.py"); cells = importlib.util.module_from_spec(s); s.loader.exec_module(cells)
from neuromorphic.training import lookahead as la
torch.set_num_threads(2)
tag, folder = sys.argv[1], sys.argv[2]
depth = int(sys.argv[3]); seeds = [int(x) for x in sys.argv[4].split(",")]
arm, ps = tag.split("_s")
judge = cells.make_judge(int(ps))
judge.load_state_dict(torch.load(REPO/f"experiments/073_learned_judge/{folder}/judge_{tag}/judge.pt", map_location="cpu") if (REPO/f"experiments/073_learned_judge/{folder}/judge_{tag}/judge.pt").exists() else torch.load(f"/root/scratch/exp073pilot/judge_{tag}/judge.pt", map_location="cpu"))
judge.eval()
def jvals(agent, critic, states, *, generator):
    with torch.no_grad(): return -judge(states, generator)
la.imagined_values = jvals
for seed in seeds:
    t0 = time.time()
    agent, head, states, ts = cells.c70.load_cell(depth, seed)
    res = la.evaluate_lookahead(agent, head, states, depth=depth, mode="C", k=3,
                                generator=torch.Generator().manual_seed(ts), rng_seed=ts, imag_seed=ts,
                                critic=lambda c: c, no_revisit=True)
    p3v = json.load(open(REPO/f"experiments/072_p3v_frontier/outputs/exp072_P3V_d{depth}_s{seed}.json"))
    print(f"pilot judge {tag}: d{depth} s{seed} J3V solved {res['solved']}/{res['n']}  published P3V {p3v['solved']}/{p3v['n']}  ({time.time()-t0:.0f}s)", flush=True)
