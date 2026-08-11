"""Single source of truth for the visual story's numbers.

**Scenes must not hardcode results.** A number transcribed into an animation is a number that
can drift from the experiment that produced it, and this project's whole method is
measured-not-assumed. Everything here either reads a committed record or is marked
`PUBLISHED` with the file it came from.

Two provenance classes, deliberately distinguished:

    measured_*   read from experiments/*/outputs/*.json on this machine
    PUBLISHED_*  transcribed from a committed RESULTS.md, because those raw records are
                 gitignored and were never fetched here (EXP-027, EXP-029, EXP-030)

If a PUBLISHED value is ever needed at per-seed resolution, the records have to be recovered
from whichever machine ran them - the means alone cannot produce a scatter plot.

Usage:
    from data import depth_curve, curriculum_climb, probe_table
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments"


# --------------------------------------------------------------------------- records

def _load(exp_dir: str) -> list[dict]:
    d = OUT / exp_dir / "outputs"
    if not d.is_dir():
        return []
    recs = []
    for p in sorted(d.glob("*.json")):
        try:
            recs.append(json.loads(p.read_text()))
        except (ValueError, OSError):
            continue
    return recs


def _cell(recs, *, depth=None, arm="regionalized", tag_has=None, **cfg_eq):
    """Records matching a cell. `cfg_eq` filters on `config` keys."""
    out = []
    for r in recs:
        if depth is not None and r.get("depth") != depth:
            continue
        if arm is not None and r.get("arm") != arm:
            continue
        if tag_has and tag_has not in r.get("tag", ""):
            continue
        if any(r.get("config", {}).get(k) != v for k, v in cfg_eq.items()):
            continue
        out.append(r)
    return out


def _mean(recs, field="success_rate"):
    vals = [r[field] for r in recs if field in r]
    return st.mean(vals) if vals else None


def _series(recs, field="success_rate"):
    return sorted((r["seed"], r[field]) for r in recs if field in r)


# --------------------------------------------------------------------------- published

# EXP-029 `RESULTS.md`. THE ORIGIN POINT of the cube line: five regions, a Linear(64 -> 6) head,
# 390 trainable parameters, frozen random encoder. Raw records gitignored, never fetched.
PUBLISHED_EXP029 = {
    1: {"success": 0.875, "floor": 0.208, "eval": "train-dist", "n": 6},
    2: {"success": 0.380, "floor": 0.043, "eval": "train-dist", "n": 27},
    3: {"success": 0.022, "floor": 0.014, "eval": "held-out", "n": 30},
    4: {"success": 0.000, "floor": 0.003, "eval": "held-out", "n": 133},
}

# EXP-033 `RESULTS.md`, the linear-probe ceiling on the RAW OBSERVATION. Chance is measured,
# not 1/6, because states differ in how many moves are optimal.
PUBLISHED_FACELET_PROBE = {1: 1.000, 2: 1.000, 3: 0.956, 4: 0.766, 5: 0.598}
PUBLISHED_PROBE_CHANCE = {1: 0.167, 2: 0.183, 3: 0.181, 4: 0.182, 5: 0.194}
PUBLISHED_CONCEPT64_PROBE = {1: 0.583, 2: 0.833, 3: 0.631, 4: 0.459, 5: 0.377}
# Width refuted as the route: 8x the width closes about half the gap and saturates.
PUBLISHED_CONCEPT512_PROBE = {1: 0.792, 2: 0.952, 3: 0.825, 4: 0.638, 5: 0.479}

# EXP-034 / EXP-035 `RESULTS.md`. The curriculum climb at depth 3, and the CONTROL that makes it
# a real result: direct training at 5x the episodes scores WORSE than the 600-episode baseline.
PUBLISHED_CURRICULUM = [
    ("EXP-029 baseline", 0.022),
    ("curriculum 600", 0.097),
    ("curriculum 3,000", 0.256),
    ("curriculum 10,000", 0.397),
    ("curriculum 30,000", 0.500),
]
PUBLISHED_DIRECT_CONTROL = [("direct 600", 0.022), ("direct 3,000", 0.019)]

# 2x2 cube state-space census, God's number 14. From the vault roadmap.
STATE_CENSUS = [
    ("1-3", 153, 0.00004),
    ("4-6", 11_759, 0.0032),
    ("7-9", 507_715, 0.141),
    ("10-12", 3_063_976, 0.975),
    ("13-14", 90_556, 1.000),
]
TOTAL_STATES = 3_674_160
RANDOM_SCRAMBLE_DEPTH = 11
UNIFORM_MODAL_9STEP = 0.354      # budget-dependent; 0.309 at depth 6's 15 steps


# --------------------------------------------------------------------------- the story

def depth_curve() -> dict:
    """Act 3's payoff: EXP-036 (frozen) vs EXP-040 (pretrained), measured here."""
    before, after = _load("036_generalisation_gap"), _load("040_pretrained_encoder_policy")
    out = {}
    for d in (4, 5, 6):
        b, a = _cell(before, depth=d), _cell(after, depth=d)
        out[d] = {
            "before": _mean(b),
            "after": _mean(a),
            "before_seeds": _series(b),
            "after_seeds": _series(a),
            "floor": _mean(_cell(before, depth=d, arm="random")),
        }
    return out


def encoder_probe() -> dict:
    """EXP-039: frozen vs trained concept vs the facelet ceiling, measured here."""
    recs = _load("039_encoder_pretraining")
    out = {}
    for d in ("3", "4", "5", "6"):
        row = {}
        for arm in ("frozen", "trained", "facelets"):
            vals = [r[arm]["by_depth"][d]["top1"] for r in recs
                    if arm in r and d in r[arm].get("by_depth", {})]
            row[arm] = st.mean(vals) if vals else None
        out[int(d)] = row
    return out


def collapse_evidence() -> dict:
    """Act 2's twist: de-collapsing the policy did NOT help.

    EXP-038's depth-6 dose axis, measured here. The entropy bonus drove modal fraction from
    0.975 down to 0.631 and success stayed at the floor.
    """
    recs = _load("038_depth6_collapse")
    base = _cell(_load("036_generalisation_gap"), depth=6)
    out = {"baseline": {"modal": _mean(base, "greedy_modal_action_frac"),
                        "success": _mean(base, "success_rate")}}
    for beta in sorted({r["config"]["entropy_beta"] for r in recs if r["depth"] == 6}):
        c = _cell(recs, depth=6, entropy_beta=beta)
        out[f"beta {beta}"] = {"modal": _mean(c, "greedy_modal_action_frac"),
                               "success": _mean(c, "success_rate")}
    return out


def origin_vs_now() -> dict:
    """Scene 7: EXP-029's opening table against today's."""
    now = depth_curve()
    return {
        "origin": PUBLISHED_EXP029,
        "now": {d: now[d]["after"] for d in now},
        "depth3_then": PUBLISHED_EXP029[3]["success"],
        "depth3_now": 0.500,          # EXP-035 at 30,000 episodes
        "trainable_params": 390,
    }


def availability() -> dict:
    """What is actually on this machine. Print before rendering anything."""
    wanted = ["029_cube_baseline", "030_memory_engagement", "031_policy_collapse",
              "033_concept_decodability", "034_learning_signal", "035_budget_scaling",
              "036_generalisation_gap", "038_depth6_collapse", "039_encoder_pretraining",
              "040_pretrained_encoder_policy"]
    return {e: len(_load(e)) for e in wanted}


if __name__ == "__main__":
    print("record availability (0 = published numbers only):")
    for exp, n in availability().items():
        print(f"  {exp:<34} {n:>4}")
    print("\ndepth curve, EXP-036 -> EXP-040:")
    for d, row in depth_curve().items():
        print(f"  depth {d}: {row['before']:.4f} -> {row['after']:.4f}  (floor {row['floor']:.4f})")
    print("\nencoder probe, EXP-039:")
    for d, row in encoder_probe().items():
        print(f"  depth {d}: frozen {row['frozen']:.3f}  trained {row['trained']:.3f}  "
              f"facelets {row['facelets']:.3f}")
    print("\ncollapse evidence, EXP-038 depth 6:")
    for k, v in collapse_evidence().items():
        print(f"  {k:<12} modal {v['modal']:.3f}  success {v['success']:.4f}")
