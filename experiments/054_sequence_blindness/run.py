"""EXP-054: is the concept sequence-blind, and does that explain the pretraining collapse?

Five arms of 12 seeds. Every encoder ALREADY EXISTS on disk (E0 is rebuilt from its seed).
Nothing is trained, no policy is run, and this must not touch the laptop - EXP-053's arms are
using it.

    epochs   move-accuracy   depth-6 policy
      0            -            0.0000     random frozen encoder (EXP-036)
     10          0.383          0.2012
     20          0.414          0.1850
     40          0.437          0.1800     the inherited value (EXP-043)
     80          0.452          0.0887     (EXP-050)

THE PARADOX: the pretext metric climbs monotonically while the policy halves. EXP-052
established that and could not say why.

PRE-REGISTERED CONTRACT: docs/superpowers/specs/2026-08-29-exp054-sequence-blindness-design.md

  1. PRIMARY: does S fall with epochs? Confirmed if it decreases in at least 2 of the 3
     adjacent contrasts (10-20, 20-40, 40-80) at p <= 0.05.
  2. THE TRADEOFF: report S beside move-accuracy and policy.
  3. THE FLOOR: E0. A random encoder may score HIGHEST - random projections preserve geometry -
     and that is the result, not a failure of it. See spec section 3.
  4. THE DISQUALIFIER: if S correlates with policy in OPPOSITE directions within and between
     arms, S is retired on the spot. That is exactly how the entropy trace behaved.

Run (repo root):
    .venv/bin/python -u experiments/054_sequence_blindness/run.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neuromorphic.analysis.sequence_sensitivity import DEPTHS, N_PER_SHELL, sequence_sensitivity
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.encoder_pretrain import load_encoder, make_sensory

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
E052 = Path("experiments/052_pretraining_optimum/outputs")
E040 = Path("experiments/040_pretrained_encoder_policy/outputs")
E050 = Path("experiments/050_objective_vs_gradient/outputs")

SEEDS = tuple(range(12))

ARMS = {
    # E0 is REBUILT, not loaded: a random init is exactly reproducible from its seed and no
    # file was ever saved for it.
    "E0":  {"epochs": 0,  "path_fn": None,                                        "policy": 0.0000},
    "E10": {"epochs": 10, "path_fn": lambda s: E052 / f"exp052_encoder_e10_s{s}.pt", "policy": 0.2012},
    "E20": {"epochs": 20, "path_fn": lambda s: E052 / f"exp052_encoder_e20_s{s}.pt", "policy": 0.1850},
    "E40": {"epochs": 40, "path_fn": lambda s: E040 / f"exp040_encoder_s{s}.pt",     "policy": 0.1800},
    "E80": {"epochs": 80, "path_fn": lambda s: E050 / f"exp050_encoder_plus_s{s}.pt", "policy": 0.0887},
}

# PER-SEED policy, which Claim 4 requires. The arm means above are for the printed table only.
# Claim 4 correlates S against policy WITHIN an arm, and an arm-constant policy has no variance
# to correlate against - the disqualifier could then never fire and the hard rule would be
# decorative. Verified 2026-08-30 that all four tags hold 12 depth-6 records each.
E043 = Path("experiments/043_cap_at_depth_5_6/outputs")
POLICY_SOURCES = {
    "E10": (E052, "exp052_e10_d6"),
    "E20": (E052, "exp052_e20_d6"),
    "E40": (E043, "exp043_capped_d6"),
    "E80": (E050, "exp050_pre2_d6"),
    # E0 has no records: EXP-036 measured every seed at exactly 0.0000. It is excluded from
    # Claim 4 for exactly that reason and carries the constant here.
}


def policy_by_seed(arm: str) -> dict:
    """That arm's per-seed held-out success, read from the experiment that measured it."""
    if arm == "E0":
        return {s: 0.0 for s in SEEDS}
    directory, tag = POLICY_SOURCES[arm]
    out = {}
    for p in Path(directory).glob("*.json"):
        r = json.loads(p.read_text())
        if isinstance(r, dict) and r.get("tag") == tag and r.get("depth") == 6:
            out[int(r["seed"])] = float(r["success_rate"])
    return out


def record_filename(arm: str, seed: int) -> str:
    return f"exp054_{arm}_s{seed}.json"


def encoder_for(arm: str, seed: int):
    """The arm's encoder for one seed. E0 is reconstructed; every other arm is loaded."""
    path_fn = ARMS[arm]["path_fn"]
    if path_fn is None:
        return make_sensory(seed)
    return load_encoder(path_fn(seed), seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for arm in args.arms:
        fn = ARMS[arm]["path_fn"]
        if fn is None:
            continue
        missing += [str(fn(s)) for s in args.seeds if not fn(s).exists()]
    if missing:
        raise SystemExit(f"missing encoder files: {missing[:5]} ({len(missing)} total)")

    cells = [(a, s) for a in args.arms for s in args.seeds]
    print(f"EXP-054: {len(cells)} encoders, shells {DEPTHS}, up to {N_PER_SHELL} states each")
    print("  the statistic trains NOTHING. No policy runs. Does not touch the laptop.")
    print("  E0 may score HIGHEST (random projections preserve geometry) - see spec section 3.\n",
          flush=True)
    if args.dry_run:
        print(f"  --dry-run: {len(cells)} cell(s) NOT measured.")
        return

    # One bounded build, reused for every encoder. Depth 6 is 11,913 states, about 0.04s.
    provider = ExactBFSDistance(max_depth=max(DEPTHS))

    policies = {}
    for arm in args.arms:
        policies[arm] = policy_by_seed(arm)
        absent = [s for s in args.seeds if s not in policies[arm]]
        if absent:
            raise SystemExit(
                f"arm {arm} is missing per-seed policy for seeds {absent}. Claim 4 correlates S "
                "against WITHIN-arm policy variance and cannot be computed without it."
            )

    for i, (arm, seed) in enumerate(cells, 1):
        out_path = args.out_dir / record_filename(arm, seed)
        if args.skip_existing and out_path.exists():
            continue
        result = sequence_sensitivity(encoder_for(arm, seed), provider, seed=seed)
        record = {
            "arm": arm,
            "epochs": ARMS[arm]["epochs"],
            "seed": seed,
            "policy_success": policies[arm][seed],
            "arm_mean_policy": ARMS[arm]["policy"],
            **result,
        }
        out_path.write_text(json.dumps(record), encoding="utf-8")
        print(f"  {i}/{len(cells)}  {arm} s{seed}  S={result['S']:+.4f}", flush=True)

    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
