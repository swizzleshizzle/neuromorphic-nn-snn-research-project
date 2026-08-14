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

import itertools
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


def _require(recs: list[dict], exp: str, what: str) -> list[dict]:
    """Fail with the machine's name for the problem, not a TypeError three frames deep.

    Records are gitignored, so each machine has only the experiments it ran. The VPS has no
    EXP-029/030; the laptop had no EXP-039 until 2026-08-14. A missing set used to surface as
    `unsupported operand type(s) for -: 'NoneType' and 'float'` inside a scene's coordinate
    function, which says nothing about which experiment to go and fetch.
    """
    if not recs:
        raise RuntimeError(
            f"{what} needs {exp}'s records and this machine has none at "
            f"{OUT / exp / 'outputs'}. They are gitignored, so they live on whichever machine "
            f"ran them - copy them across, do NOT transcribe. `python data.py` prints what is "
            f"present here."
        )
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

# EXP-034 / EXP-035's curriculum climb used to be transcribed here. It is now read from the
# records by `curriculum_climb()` - they are all present locally, so there was never a reason to
# keep a second copy that could drift. The transcription was already off in the last digit
# (0.256 against a measured 0.2556).

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

def _exact_paired_p(diffs: list[float]) -> float:
    """Two-sided exact paired permutation over all 2**n sign flips.

    Same method as `experiments/043_cap_at_depth_5_6/aggregate.py`, which is the authority; it
    is repeated here so a caption is computed from the records rather than transcribed. No
    scipy in this repo, by design.
    """
    n = len(diffs)
    if not 1 <= n <= 20:
        raise ValueError(f"exact permutation needs 1 <= n <= 20, got {n}")
    observed = abs(sum(diffs))
    hits = sum(1 for signs in itertools.product((1, -1), repeat=n)
               if abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12)
    return hits / 2 ** n


# EXP-043's pre-registered "working" rule, restated from EXP-036. Quoted so a caption can say
# what the bar WAS rather than describing a result against a bar invented afterwards.
WORKING_BAR = 0.10
WORKING_MIN_SEEDS = 8        # of 12, each individually above the bar
WORKING_MIN_SE = 1.0         # margin, because EXP-040 cleared the bare rule by 0.11 SE

# Which record set carries the current "after" arm at each depth. The depth-1 training cap
# (`max_steps_by_depth=((1,2),)`) is the only change from the pretrained arm.
CAPPED_SOURCE = {
    4: ("042_depth1_trap", "exp042_capped"),
    5: ("043_cap_at_depth_5_6", "exp043_capped"),
    6: ("043_cap_at_depth_5_6", "exp043_capped"),
}


def depth_curve() -> dict:
    """Act 3's payoff, THREE arms, all measured here. The two levers compound; neither alone
    reaches depth 6.

        before      EXP-036 - frozen randomly-initialised encoder
        pretrained  EXP-040 - encoder trained self-supervised (EXP-039), depth-1 trap still in
        after       EXP-042 (d4) / EXP-043 (d5,d6) - same encoder, depth-1 training budget capped

    `after_p` is the paired permutation p of after-vs-pretrained. It is NOT decoration: depth 5's
    +0.1108 misses its pre-registered 0.05 at p 0.0815 and is REFUTED, and any caption that shows
    that bar has to say so.
    """
    before = _require(_load("036_generalisation_gap"), "036_generalisation_gap",
                      "TheBreakPointMoves")
    pretrained = _require(_load("040_pretrained_encoder_policy"),
                          "040_pretrained_encoder_policy", "TheBreakPointMoves")
    out = {}
    for d in (4, 5, 6):
        exp, tag = CAPPED_SOURCE[d]
        b = _cell(before, depth=d)
        m = _cell(pretrained, depth=d)
        a = _cell(_load(exp), depth=d, tag_has=tag)
        paired = dict(_series(m))
        diffs = [v - paired[s] for s, v in _series(a) if s in paired]
        out[d] = {
            "before": _mean(b),
            "pretrained": _mean(m),
            "after": _mean(a),
            "before_seeds": _series(b),
            "pretrained_seeds": _series(m),
            "after_seeds": _series(a),
            "after_p": _exact_paired_p(diffs) if diffs else None,
            "floor": _mean(_cell(before, depth=d, arm="random")),
        }
    return out


def working_bar(depth: int = 6) -> dict:
    """Is a depth WORKING by the pre-registered rule, measured from the records?

    EXP-040's depth 6 satisfied the bare `mean >= 0.10` on noise - 0.1037, 0.11 SE of margin,
    5 of 12 seeds above it. The margin and seed-count conditions exist because of that, so a
    caption that claims "working" should carry them.
    """
    seeds = depth_curve()[depth]["after_seeds"]
    vals = [v for _, v in seeds]
    mean = st.mean(vals)
    se = st.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
    above = sum(1 for v in vals if v > WORKING_BAR)
    return {
        "mean": mean,
        "se_margin": (mean - WORKING_BAR) / se if se else None,
        "seeds_above": above,
        "n": len(vals),
        "working": (mean >= WORKING_BAR and se and (mean - WORKING_BAR) / se >= WORKING_MIN_SE
                    and above >= WORKING_MIN_SEEDS),
    }


def curriculum_climb() -> dict:
    """Act 2's one thing that worked, at depth 3, measured here.

    The climb alone proves nothing - more episodes is more compute. The result is the CONTROL:
    direct training at 3,000 episodes, five times the budget and no curriculum, scores WORSE than
    the 600-episode run. Both series must be on screen or the scene is just a rising line.

    EXP-029's baseline and chance floor at depth 3 stay PUBLISHED; those records were never
    fetched here.
    """
    recs = _require(_load("034_learning_signal") + _load("035_budget_scaling"),
                    "EXP-034/035", "TheCurriculumUnlock")

    def at(tag):
        return _mean(_cell(recs, depth=3, tag_has=tag))

    return {
        "curriculum": [(600, at("exp034_curriculum_e600")),
                       (3_000, at("exp034_curriculum_e3000")),
                       (10_000, at("exp035_curriculum_e10000")),
                       (30_000, at("exp035_curriculum_e30000"))],
        "direct": [(600, at("exp034_direct_e600")),
                   (3_000, at("exp034_direct_e3000"))],
        "floor": PUBLISHED_EXP029[3]["floor"],
        "origin": PUBLISHED_EXP029[3]["success"],
    }


def encoder_probe() -> dict:
    """EXP-039: frozen vs trained concept vs the facelet ceiling, measured here."""
    recs = _require(_load("039_encoder_pretraining"), "039_encoder_pretraining",
                    "TheEncoderLearns")
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
    recs = _require(_load("038_depth6_collapse"), "038_depth6_collapse",
                    "CollapseIsASymptom")
    base = _cell(_load("036_generalisation_gap"), depth=6)
    out = {"baseline": {"modal": _mean(base, "greedy_modal_action_frac"),
                        "success": _mean(base, "success_rate")}}
    for beta in sorted({r["config"]["entropy_beta"] for r in recs if r["depth"] == 6}):
        c = _cell(recs, depth=6, entropy_beta=beta)
        out[f"beta {beta}"] = {"modal": _mean(c, "greedy_modal_action_frac"),
                               "success": _mean(c, "success_rate")}
    return out


TRAINABLE_PARAMS = 390       # Linear(64 -> 6) plus bias. Unchanged since EXP-029.


def origin_vs_now() -> list[dict]:
    """Scene 7's table: EXP-029's opening numbers against today's, row per depth.

    `then` is None where EXP-029 never went. It stopped at depth 4, having scored 0.000 there.

    > THE DEPTH-3 "NOW" CELL IS NOT ON TODAY'S RECIPE, and carries a note saying so. It is
    > EXP-035's 30,000-episode run on a FROZEN encoder. Depth 3 has never been re-measured with
    > the pretrained encoder and the depth-1 cap, so dropping it into the column unmarked would
    > silently mix two encoders and a 3x budget difference into one "now". The other three cells
    > are all 10,000 episodes on today's recipe.
    """
    now = depth_curve()
    rows = [{"depth": 3,
             "then": PUBLISHED_EXP029[3]["success"],
             "now": curriculum_climb()["curriculum"][-1][1],
             "note": "EXP-035: frozen encoder, 30,000 episodes"}]
    rows += [{"depth": d,
              "then": PUBLISHED_EXP029.get(d, {}).get("success"),
              "now": now[d]["after"],
              "note": None} for d in (4, 5, 6)]
    return rows


def availability() -> dict:
    """What is actually on this machine. Print before rendering anything."""
    wanted = ["029_cube_baseline", "030_memory_engagement", "031_policy_collapse",
              "033_concept_decodability", "034_learning_signal", "035_budget_scaling",
              "036_generalisation_gap", "038_depth6_collapse", "039_encoder_pretraining",
              "040_pretrained_encoder_policy", "042_depth1_trap", "043_cap_at_depth_5_6"]
    return {e: len(_load(e)) for e in wanted}


if __name__ == "__main__":
    print("record availability (0 = published numbers only):")
    for exp, n in availability().items():
        print(f"  {exp:<34} {n:>4}")
    print("\ndepth curve, EXP-036 -> EXP-040 -> EXP-042/043 (cap):")
    for d, row in depth_curve().items():
        print(f"  depth {d}: {row['before']:.4f} -> {row['pretrained']:.4f} -> {row['after']:.4f}"
              f"  (floor {row['floor']:.4f}, cap-vs-pretrained p {row['after_p']:.4f})")
    w = working_bar(6)
    print(f"\ndepth 6 against the pre-registered bar: mean {w['mean']:.4f}, "
          f"{w['se_margin']:.2f} SE of margin, {w['seeds_above']}/{w['n']} seeds above "
          f"{WORKING_BAR} -> {'WORKING' if w['working'] else 'NOT working'}")
    climb = curriculum_climb()
    print("\ncurriculum climb at depth 3, EXP-034/035 (chance floor "
          f"{climb['floor']:.3f}):")
    for label in ("curriculum", "direct"):
        pts = "  ".join(f"{e:,}: {v:.4f}" for e, v in climb[label])
        print(f"  {label:<11} {pts}")

    print("\nencoder probe, EXP-039:")
    for d, row in encoder_probe().items():
        print(f"  depth {d}: frozen {row['frozen']:.3f}  trained {row['trained']:.3f}  "
              f"facelets {row['facelets']:.3f}")
    print("\ncollapse evidence, EXP-038 depth 6:")
    for k, v in collapse_evidence().items():
        print(f"  {k:<12} modal {v['modal']:.3f}  success {v['success']:.4f}")
