"""Diagnostic: is the fine-tuned encoder MORE DECODABLE, or just EASIER TO LEARN FROM?

The probe trains with full oracle supervision to convergence (300 epochs), so it measures
representational CAPACITY. REINFORCE trains on sparse reward, so the policy measures what is
actually EXPLOITABLE. EXP-028 already found this cube/grid family is optimization-limited rather
than encoder-limited, which would let the two disagree exactly as EXP-047/048 saw.

Test: fit the SAME probe at increasing epoch budgets. If B only matches A at convergence but
gets there much faster, capacity is flat and learnability improved.
"""
import sys, importlib.util, statistics as st, torch
sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location("pe", "experiments/047_encoder_finetuning/probe_encoders.py")
pe = importlib.util.module_from_spec(spec); spec.loader.exec_module(pe)
from neuromorphic.envs.cube_distance import ExactBFSDistance
torch.set_num_threads(2)

BUDGETS = [5, 15, 50, 300]
DEPTH = 6
prov = ExactBFSDistance(max_depth=7)
states, masks, depths = pe.build_dataset(prov)
print(f"{len(states)} states; depth-{DEPTH} probe accuracy by epoch budget\n")

rows = {b: {"A": [], "B": []} for b in BUDGETS}
for seed in range(12):
    A = pe.load_sensory(f"experiments/040_pretrained_encoder_policy/outputs/exp040_encoder_s{seed}.pt")
    B = pe.load_sensory(f"experiments/047_encoder_finetuning/outputs/"
                        f"exp047_ft_d6_lr0.0001_regionalized_d6_s{seed}_sig0.0_encoder.pt")
    tr, he = pe.standard_split(depths, seed)
    for name, enc in (("A", A), ("B", B)):
        x = pe.encoder_features(enc, states, seed)
        for b in BUDGETS:
            m = pe.probe.fit_linear_probe(x[tr], [masks[i] for i in tr], epochs=b, lr=0.1, seed=seed)
            with torch.no_grad():
                lg = m(x[he])
            sel = [j for j, i in enumerate(he) if depths[i] == DEPTH]
            rows[b][name].append(pe.probe.top1_accuracy(lg[sel], [masks[he[j]] for j in sel]))
    print(f"  seed {seed} done", flush=True)

print(f"\n{'epochs':>7}  {'A (pretrained)':>15}  {'B (fine-tuned)':>15}  {'B - A':>8}  {'W-L':>6}")
for b in BUDGETS:
    a, bb = rows[b]["A"], rows[b]["B"]
    d = [y - x for x, y in zip(a, bb)]
    w = sum(1 for v in d if v > 0); l = sum(1 for v in d if v < 0)
    print(f"{b:7d}  {st.mean(a):15.4f}  {st.mean(bb):15.4f}  {st.mean(d):+8.4f}  {w:3d}-{l:<3d}")


# ---------------------------------------------------------------------------
# PART 2 - the behavioural comparison, which is what actually resolved it.
#
# Part 1 above refutes both "more decodable" and "faster to learn from": B is flat-to-slightly
# WORSE than A at every epoch budget. So the gain is not in single-step move prediction at all.
#
# The probe scores moves from ISOLATED STATES drawn from a shell. The policy is judged on
# TRAJECTORIES. These come apart exactly when a policy undoes its own work - and on a cube a face
# move has ORDER 4, so oscillation is the dominant failure mode (the same arithmetic that made
# EXP-042's depth-1 trap).
# ---------------------------------------------------------------------------
import itertools, json
from pathlib import Path


def _load(d, tag, depth=6):
    out = {}
    for p in Path(d).glob("*.json"):
        r = json.loads(p.read_text())
        if isinstance(r, dict) and r.get("tag") == tag and r.get("depth") == depth:
            out[r["seed"]] = r
    return out


def _perm_p(diffs):
    n = len(diffs)
    obs = abs(sum(diffs))
    hits = sum(1 for s in itertools.product((1, -1), repeat=n)
               if abs(sum(x * y for x, y in zip(s, diffs))) >= obs - 1e-12)
    return hits / 2 ** n


def behavioural():
    A = _load("experiments/043_cap_at_depth_5_6/outputs", "exp043_capped_d6")
    B = _load("experiments/048_fresh_head/outputs", "exp048_freshhead_d6")
    seeds = sorted(set(A) & set(B))
    print(f"\n\nPART 2 - behaviour, n={len(seeds)} paired seeds")
    print(f"{'metric':26s} {'A':>9} {'B':>9} {'B-A':>9} {'p':>7}")
    for k in ("success_rate", "eval_revisit_rate", "optimality",
              "revisit_rate", "mean_train_entropy", "greedy_modal_action_frac"):
        a = [A[s][k] for s in seeds]
        b = [B[s][k] for s in seeds]
        d = [y - x for x, y in zip(a, b)]
        print(f"{k:26s} {st.mean(a):9.4f} {st.mean(b):9.4f} {st.mean(d):+9.4f} {_perm_p(d):7.4f}")


behavioural()
