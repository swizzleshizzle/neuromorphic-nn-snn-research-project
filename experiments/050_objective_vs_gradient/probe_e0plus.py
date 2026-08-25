"""EXP-050 Claim 4: does the probe's anti-correlation belong to the RL objective specifically?

  EXP-039: inverse-model pretraining RAISES the probe (+0.3396 at depth 4, 12-0).
  EXP-049: RL fine-tuning LOWERS it (0-12, p 0.0005) while policy success nearly doubles.

PRE-REGISTERED PREDICTION: E0+ probes HIGHER than E0 (more of the objective that raised it)
while arm F gains LESS policy than arm B. If both hold, the anti-correlation is a property of
the RL objective rather than of encoder training in general.

Arm F is now known: 0.0887 against arm A's 0.1800 - more pretraining HALVED the policy. So if
the probe also rises here, the probe is anti-correlated with policy in BOTH directions and the
finding is far more general than "RL is special".

Run (repo root):
    .venv/bin/python -u experiments/050_objective_vs_gradient/probe_e0plus.py
"""

from __future__ import annotations

import importlib.util
import itertools
import statistics as st
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

_sp = importlib.util.spec_from_file_location(
    "pe", "experiments/047_encoder_finetuning/probe_encoders.py")
pe = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(pe)

from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402

torch.set_num_threads(2)

E0 = "experiments/040_pretrained_encoder_policy/outputs/exp040_encoder_s{}.pt"
E0P = "experiments/050_objective_vs_gradient/outputs/exp050_encoder_plus_s{}.pt"
DEPTHS_REPORTED = ("4", "6")


def perm_p(d):
    n, obs = len(d), abs(sum(d))
    return sum(1 for s in itertools.product((1, -1), repeat=n)
               if abs(sum(x * y for x, y in zip(s, d))) >= obs - 1e-12) / 2 ** n


def main() -> None:
    prov = ExactBFSDistance(max_depth=7)
    states, masks, depths = pe.build_dataset(prov)
    acc = {d: {"E0": [], "E0+": []} for d in DEPTHS_REPORTED}

    for seed in range(12):
        tr, he = pe.standard_split(depths, seed)
        for name, tmpl in (("E0", E0), ("E0+", E0P)):
            x = pe.encoder_features(pe.load_sensory(tmpl.format(seed)), states, seed)
            m = pe.probe.fit_linear_probe(x[tr], [masks[i] for i in tr],
                                          epochs=300, lr=0.1, seed=seed)
            with torch.no_grad():
                lg = m(x[he])
            for d in DEPTHS_REPORTED:
                sel = [j for j, i in enumerate(he) if depths[i] == int(d)]
                acc[d][name].append(
                    pe.probe.top1_accuracy(lg[sel], [masks[he[j]] for j in sel]))
        print(f"  seed {seed}: d4 {acc['4']['E0'][-1]:.4f} -> {acc['4']['E0+'][-1]:.4f}   "
              f"d6 {acc['6']['E0'][-1]:.4f} -> {acc['6']['E0+'][-1]:.4f}", flush=True)

    print(f"\n{'depth':>6}  {'E0':>8}  {'E0+':>8}  {'delta':>8}  {'W-L':>7}  {'p':>7}")
    for d in DEPTHS_REPORTED:
        a, b = acc[d]["E0"], acc[d]["E0+"]
        diff = [y - x for x, y in zip(a, b)]
        print(f"{d:>6}  {st.mean(a):8.4f}  {st.mean(b):8.4f}  {st.mean(diff):+8.4f}  "
              f"{sum(1 for v in diff if v > 0):3d}-{sum(1 for v in diff if v < 0):<3d}  "
              f"{perm_p(diff):7.4f}")

    print(f"\nPOLICY, for the same encoders: E0 -> 0.1800, E0+ -> 0.0887 (HALVED, p 0.0078)")
    print("If the probe ROSE while the policy halved, then together with EXP-049 the probe is")
    print("anti-correlated with policy in BOTH directions, and the finding is not about RL.")


if __name__ == "__main__":
    main()
