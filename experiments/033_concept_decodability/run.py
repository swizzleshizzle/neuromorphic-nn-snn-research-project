# experiments/033_concept_decodability/run.py
"""EXP-033 Part 1 driver: is the optimal move linearly readable from the frozen concept?

No reinforcement learning happens here. This is an offline probe, so it is cheap and it
isolates representation from optimisation, which every previous cube experiment has
confounded.

Feature sets, all fitted at matched states and matched splits:

    facelets            raw 144-d one-hot observation
    concept @ W         frozen concept at content width W, for W in WIDTHS
    chance              measured from the labels, never assumed

PRE-REGISTERED CONTRACT (committed before any number exists). Three outcomes, three
different next experiments:

  A. facelets HIGH, concept low, and concept does not close the gap as W grows.
     The encoder is destroying information that is present in the observation. Width is
     REFUTED as the lever and Part 2 (training the encoder on cube dynamics) is justified.

  B. facelets LOW and concept low.
     The optimal move is not linearly decodable from the observation at all. The linear
     head is then the binding constraint as much as the encoder, and Part 2 must pair a
     learned representation with a nonlinear readout rather than assume the head is fine.

  C. concept ~ facelets and BOTH high.
     The information is present and reachable by a linear map, so the failure is in the
     reinforcement learning, not the representation. Part 2 would be misdirected; the next
     move would be credit assignment.

  Width is judged SUFFICIENT only if concept@512 comes within 5 points of the facelets
  probe on held-out top-1 accuracy. That bar is set here, before the numbers, and must not
  be moved afterwards.

Reporting is per depth as well as pooled, because the policy only ever trains on depths 1-3
and a number pooled over depths 1-5 could hide a collapse in exactly the regime that matters.

Run (repo root):
    .venv/bin/python -u experiments/033_concept_decodability/run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.envs.cube import CubeEnv
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.cube_baseline import CubeConfig, make_agent
from neuromorphic.training.reinforce import concept_rate

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp033_probe", HERE / "probe.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

WIDTHS = [64, 128, 256, 512]   # 64 is the shipped default
DEPTHS = [1, 2, 3, 4, 5]       # policy trains on 1-3; 4-5 test whether it generalises
FRAC_HELDOUT = 0.25
SUFFICIENCY_MARGIN = 0.05      # pre-registered: within 5 points of facelets


def build_dataset(provider):
    """All states at DEPTHS, with their optimal-move masks. Depth kept for per-depth reporting."""
    states, masks, depths = [], [], []
    for d in DEPTHS:
        for s in provider.states_at_distance(d):
            states.append(s)
            masks.append(probe.optimal_move_mask(provider, s))
            depths.append(d)
    return states, masks, depths


def concept_features(states, width: int, seed: int) -> torch.Tensor:
    """Frozen concept for each state. The brain is never trained, here or anywhere in v1."""
    brain = make_agent(CubeConfig(seed=seed, content=width))
    gen = torch.Generator().manual_seed(seed)
    env = CubeEnv(scramble_depth=max(DEPTHS), max_steps=1, scramble_seed=seed)
    rows = []
    for s in states:
        obs, _ = env.reset(options={"state": s})
        with torch.no_grad():
            out = brain.step(obs, store=False, recall=False, record=False, generator=gen)
        rows.append(concept_rate(out))
    return torch.stack(rows)


def _evaluate(x, masks, depths, idx_train, idx_held, seed) -> dict:
    model = probe.fit_linear_probe(
        x[idx_train], [masks[i] for i in idx_train], epochs=300, lr=0.1, seed=seed
    )
    with torch.no_grad():
        logits = model(x[idx_held])
    held_masks = [masks[i] for i in idx_held]
    out = {
        "top1": probe.top1_accuracy(logits, held_masks),
        "chance": probe.chance_floor(held_masks),
        "by_depth": {},
    }
    for d in DEPTHS:
        sel = [j for j, i in enumerate(idx_held) if depths[i] == d]
        if sel:
            out["by_depth"][str(d)] = {
                "top1": probe.top1_accuracy(logits[sel], [held_masks[j] for j in sel]),
                "chance": probe.chance_floor([held_masks[j] for j in sel]),
                "n": len(sel),
            }
    return out


def _run_one(width: int, seed: int, out_dir: Path) -> dict:
    torch.set_num_threads(1)
    # One depth BEYOND the deepest labelled state: optimal_move_mask inspects neighbours,
    # and a depth-5 state has depth-6 neighbours that must be in the table.
    provider = ExactBFSDistance(max_depth=max(DEPTHS) + 1)
    states, masks, depths = build_dataset(provider)

    # Stratify the split by depth so a deep shell cannot dominate the held-out set.
    idx_train, idx_held = [], []
    for d in DEPTHS:
        pool = [i for i, dd in enumerate(depths) if dd == d]
        tr, he = probe.split_states(pool, FRAC_HELDOUT, seed)
        idx_train += tr
        idx_held += he

    record = {"width": width, "seed": seed, "n_states": len(states),
              "n_train": len(idx_train), "n_heldout": len(idx_held)}
    record["facelets"] = _evaluate(
        probe.facelet_features(states), masks, depths, idx_train, idx_held, seed
    )
    record["concept"] = _evaluate(
        concept_features(states, width, seed), masks, depths, idx_train, idx_held, seed
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"exp033_w{width}_s{seed}.json").write_text(json.dumps(record), encoding="utf-8")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--widths", type=int, nargs="+", default=WIDTHS)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    jobs = [(w, s) for w in args.widths for s in args.seeds]
    print(f"EXP-033 Part 1: {len(jobs)} probe fits "
          f"({len(args.widths)} widths x {len(args.seeds)} seeds), depths {DEPTHS}")
    print(f"sufficiency bar (pre-registered): concept@{max(args.widths)} within "
          f"{SUFFICIENCY_MARGIN:.2f} of facelets on held-out top-1\n")

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, w, s, args.out_dir): (w, s) for w, s in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            if i % 5 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    print(f"\ndone. one record per (width, seed) in {args.out_dir}\n")
    face = st.mean(r["facelets"]["top1"] for r in records)
    chance = st.mean(r["concept"]["chance"] for r in records)
    print(f"{'features':>16}{'heldout top1':>15}{'vs chance':>12}")
    print(f"{'chance (measured)':>16}{chance:>15.3f}{'':>12}")
    print(f"{'facelets (144d)':>16}{face:>15.3f}{face - chance:>+12.3f}")
    for w in sorted(args.widths):
        sub = [r for r in records if r["width"] == w]
        if sub:
            m = st.mean(r["concept"]["top1"] for r in sub)
            print(f"{f'concept @ {w}':>16}{m:>15.3f}{m - chance:>+12.3f}")


if __name__ == "__main__":
    main()
