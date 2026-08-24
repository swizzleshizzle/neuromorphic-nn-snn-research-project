"""Claim 5: does the probe drift DOWN again in round 2 (E2 below E1)?"""
import sys, importlib.util, statistics as st, torch, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
sp = importlib.util.spec_from_file_location("pe","experiments/047_encoder_finetuning/probe_encoders.py")
pe = importlib.util.module_from_spec(sp); sp.loader.exec_module(pe)
from neuromorphic.envs.cube_distance import ExactBFSDistance
import itertools
torch.set_num_threads(2)
prov = ExactBFSDistance(max_depth=7)
states, masks, depths = pe.build_dataset(prov)
E1 = "experiments/047_encoder_finetuning/outputs/exp047_ft_d6_lr0.0001_regionalized_d6_s{}_sig0.0_encoder.pt"
E2 = "experiments/049_second_round/outputs/exp049_ft2_d6_regionalized_d6_s{}_sig0.0_encoder.pt"
a=[];b=[]
for seed in range(12):
    tr, he = pe.standard_split(depths, seed)
    for path, acc in ((E1, a), (E2, b)):
        x = pe.encoder_features(pe.load_sensory(path.format(seed)), states, seed)
        m = pe.probe.fit_linear_probe(x[tr], [masks[i] for i in tr], epochs=300, lr=0.1, seed=seed)
        with torch.no_grad(): lg = m(x[he])
        sel=[j for j,i in enumerate(he) if depths[i]==6]
        acc.append(pe.probe.top1_accuracy(lg[sel], [masks[he[j]] for j in sel]))
    print(f"  seed {seed}: E1 {a[-1]:.4f} -> E2 {b[-1]:.4f}", flush=True)
d=[y-x for x,y in zip(a,b)]
obs=abs(sum(d)); n=len(d)
p=sum(1 for s in itertools.product((1,-1),repeat=n) if abs(sum(x*y for x,y in zip(s,d)))>=obs-1e-12)/2**n
print(f"\ndepth-6 probe: E1 {st.mean(a):.4f} -> E2 {st.mean(b):.4f}  {st.mean(d):+.4f}  "
      f"W-L {sum(1 for v in d if v>0)}-{sum(1 for v in d if v<0)}  p {p:.4f}")
