# experiments/039_encoder_pretraining/run.py
"""EXP-039 driver: does inverse-model pretraining raise the EXP-033 probe ceiling?

Vault Stage 2, first increment - the moment the SNN stops being a fixed random projection.

Everything the cube line has achieved runs on a frozen randomly-initialised brain and a
`Linear(64 -> 6)` head: 390 trainable parameters. EXP-033 measured what that costs. Top-1
accuracy of a linear probe for "which moves reduce distance-to-solved":

    depth              3       4       5
    chance           0.181   0.182   0.194
    facelets (144d)  0.956   0.766   0.598      <- the linear ceiling ON THE OBSERVATION
    concept @ 64     0.631   0.459   0.377      <- what the policy head actually sees
    concept @ 512    0.825   0.638   0.479      <- width refuted as the route (EXP-033 F1)

So the encoder must be TRAINED rather than widened. A linear probe on a trained encoder is a
NONLINEAR function of the observation, so unlike width it is not bounded by 0.766 at all.

NO REINFORCEMENT LEARNING HAPPENS HERE. Supervised pretraining plus an offline probe, which is
why it runs on the VPS. Whether a raised ceiling converts into a better policy is the Stage 2
follow-on and needs the laptop.

PRE-REGISTERED CONTRACT, committed before any number exists. Full version and the reasoning
behind each threshold: docs/superpowers/specs/2026-08-08-exp039-encoder-pretraining-design.md

  All comparisons PAIRED per seed, exact permutation over all 2**12 = 4096 sign flips.

  1. PRIMARY. Probe top-1 at DEPTH 4, trained vs frozen. CONFIRMED needs >= +0.05 at
     p <= 0.05. +0.05 is what one width doubling buys (0.459 -> 0.517), so the bar is "at
     least as much as doubling the width", from an intervention that adds no width.

  2. THE THESIS BAR. Trained concept@64 at depth 4 exceeds 0.766, the raw-facelet linear
     ceiling. Clearing it means the encoder supplies genuine nonlinear structure - the first
     moment the SNN earns its place. Deliberately hard; stated in advance so that clearing
     Bar 1 alone is never later described as if it cleared Bar 2.

  3. DEPTH PROFILE. Report depths 3-6. Gains at depth 3 are the LEAST interesting (frozen
     already reaches 0.631 and the policy already works there). Depths 5 and 6 are where
     Wall 1 bites hardest. A gain that GROWS with depth is materially stronger than a uniform
     shift and must be reported as such.

  4. THE NULL IS A RESULT. An inverse model learns WHAT A MOVE DID, not WHICH MOVE IS GOOD.
     That transfer IS the hypothesis. If Bar 1 refutes, report it as "inverse dynamics is not
     sufficient to make optimality linearly readable" and redirect Stage 2 to a value/heuristic
     objective - do not leave the stage open-ended.

TWO CONTROLS DECIDE WHETHER ANY OF IT IS VALID:

  A. The probe's held-out states are EXCLUDED FROM PRETRAINING, by BOTH pair endpoints. `s'`
     goes through the same encoder as `s`, so a pair whose successor is held out shows the
     encoder that state's facelets just as directly. The inverse model never sees distance
     labels so it cannot memorise optimality, but it can memorise state-specific structure.

  B. The FROZEN ARM IS RE-MEASURED HERE, not taken from EXP-033. This batches SensoryCortex
     instead of looping `brain.step`, which consumes the Poisson generator differently, so the
     spike draws differ. Verified 2026-08-08 to agree in DISTRIBUTION (mean per-unit |diff|
     falls 0.0202 -> 0.0054 as draws go 12 -> 240, i.e. 1/sqrt(N)). EXP-033's 0.459 is an
     EXTERNAL SANITY CHECK ONLY: if the frozen arm lands outside 0.459 +- 0.10 at depth 4,
     stop and explain that before reading anything else.

Run (repo root):
    .venv/bin/python -u experiments/039_encoder_pretraining/run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11
    .venv/bin/python -u experiments/039_encoder_pretraining/run.py --seeds 0 --epochs 8   # pilot
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.encoder_pretrain import (
    PretrainConfig,
    build_pairs,
    concept_rates,
    make_sensory,
    states_to_obs,
    train_inverse_model,
)

HERE = Path(__file__).resolve().parent

# EXP-033's probe, imported UNMODIFIED and by file path. `experiments` is not a package and its
# directory names start with digits, so a normal import is impossible. Reimplementing the probe
# would make every comparison against EXP-033 meaningless, which is the whole point of reusing
# it: same labels, same fit, same top-1 definition.
_PROBE_PATH = HERE.parent / "033_concept_decodability" / "probe.py"
_spec = importlib.util.spec_from_file_location("exp033_probe", _PROBE_PATH)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

# EXP-033 used depths 1-5. Depth 6 is added because the vault's Stage 2 criterion asks about
# 4-6 and depth 6 is where the RL policy scores exactly 0.0000. The joint fit therefore spans a
# different mix than EXP-033's, which is a further reason the frozen arm is re-measured rather
# than compared across pipelines.
DEPTHS = [1, 2, 3, 4, 5, 6]
FRAC_HELDOUT = 0.25              # EXP-033's value, unchanged
REPORT_DEPTHS = [3, 4, 5, 6]     # what the pre-registered bars are read from

BAR1_DELTA = 0.05                # trained - frozen at depth 4
BAR2_FACELET_CEILING = 0.766     # EXP-033's raw-facelet probe at depth 4
EXP033_FROZEN_D4 = 0.459         # external sanity check only, +- 0.10
SANITY_TOLERANCE = 0.10


def build_dataset(provider):
    """All states at DEPTHS with their optimal-move masks. Mirrors EXP-033's build_dataset."""
    states, masks, depths = [], [], []
    for d in DEPTHS:
        for s in provider.states_at_distance(d):
            states.append(s)
            masks.append(probe.optimal_move_mask(provider, s))
            depths.append(d)
    return states, masks, depths


def _evaluate(x, masks, depths, idx_train, idx_held, seed) -> dict:
    """EXP-033's `_evaluate`, unchanged: one probe fit jointly, reported per depth."""
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


def encoder_features(sensory, states, seed: int, batch: int = 512) -> torch.Tensor:
    """Concept rates for every state, batched. Frozen or trained - same code path both ways."""
    obs = states_to_obs(states)
    gen = torch.Generator().manual_seed(seed)
    rows = []
    with torch.no_grad():
        for start in range(0, len(states), batch):
            rows.append(concept_rates(sensory, obs[start:start + batch], generator=gen))
    return torch.cat(rows, dim=0)


def run_one(seed: int, cfg: PretrainConfig, out_dir: Path, *, quiet: bool = False) -> dict:
    torch.set_num_threads(1)
    t_start = time.time()

    # One depth BEYOND the deepest labelled state: optimal_move_mask inspects neighbours, and a
    # depth-6 state has depth-7 neighbours that must be in the table. The provider REFUSES
    # rather than guessing if they are not, which is why this is max(DEPTHS) + 1.
    provider = ExactBFSDistance(max_depth=max(DEPTHS) + 1)
    states, masks, depths = build_dataset(provider)

    # Stratify by depth so a deep shell cannot dominate the held-out set (EXP-033's approach).
    idx_train, idx_held = [], []
    for d in DEPTHS:
        pool = [i for i, dd in enumerate(depths) if dd == d]
        tr, he = probe.split_states(pool, FRAC_HELDOUT, seed)
        idx_train += tr
        idx_held += he

    # CONTROL A. Everything the probe scores is withheld from pretraining, at BOTH endpoints.
    # Exclusion rather than inclusion: a successor is free to be a depth-0 or depth-7 state,
    # which needs no label because pretraining is self-supervised. Requiring successors to sit
    # inside the PROBED depths instead would delete every outward move from the deepest shell,
    # and the deepest shell is most of the data (measured: 22% of pairs surviving).
    held_states = {states[i] for i in idx_held}
    pairs = build_pairs(states, forbidden=held_states)
    assert not (held_states & {p[0] for p in pairs}), "held-out state leaked in as a source"
    assert not (held_states & {p[2] for p in pairs}), "held-out state leaked in as a successor"

    if not quiet:
        print(f"  seed {seed}: {len(states)} states, {len(idx_held)} held out, "
              f"{len(pairs)} pretraining pairs", flush=True)

    result = train_inverse_model(
        pairs, cfg,
        progress=None if quiet else (
            lambda r: print(f"    epoch {r['epoch']:>3} loss {r['loss']:.4f} "
                            f"move-acc {r['accuracy']:.3f}", flush=True)),
    )

    record = {
        "seed": seed, "n_states": len(states), "n_train": len(idx_train),
        "n_heldout": len(idx_held), "n_pairs": len(pairs),
        "epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr,
        "pretrain_history": result.history,
        "pretrain_final_move_accuracy": result.final_accuracy,
    }

    # CONTROL B. The frozen arm is measured HERE, through this pipeline, on this seed.
    record["facelets"] = _evaluate(
        probe.facelet_features(states), masks, depths, idx_train, idx_held, seed)
    record["frozen"] = _evaluate(
        encoder_features(make_sensory(seed, content=cfg.content), states, seed),
        masks, depths, idx_train, idx_held, seed)
    record["trained"] = _evaluate(
        encoder_features(result.sensory, states, seed),
        masks, depths, idx_train, idx_held, seed)

    record["wall_seconds"] = round(time.time() - t_start, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"exp039_s{seed}.json").write_text(json.dumps(record), encoding="utf-8")

    if not quiet:
        f4 = record["frozen"]["by_depth"].get("4", {}).get("top1")
        t4 = record["trained"]["by_depth"].get("4", {}).get("top1")
        print(f"  seed {seed} done in {record['wall_seconds']}s: "
              f"depth-4 frozen {f4:.3f} -> trained {t4:.3f}", flush=True)
        if f4 is not None and abs(f4 - EXP033_FROZEN_D4) > SANITY_TOLERANCE:
            print(f"  WARNING: frozen depth-4 {f4:.3f} is outside "
                  f"{EXP033_FROZEN_D4} +- {SANITY_TOLERANCE} (EXP-033). Control B says stop "
                  f"and explain this before reading any other number.", flush=True)
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    # Calibrated 2026-08-08 on seed 0, not guessed. lr chosen by the PRETRAINING OBJECTIVE
    # (move-naming accuracy 0.455 at 3e-3 against 0.425 at 1e-3), never by the probe - which
    # would have picked the other one (0.784 vs 0.791) and tuned the outcome metric. Spec 6a.
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    print(f"EXP-039: {len(args.seeds)} seeds, depths {DEPTHS}, "
          f"{args.epochs} epochs, batch {args.batch_size}, lr {args.lr}")
    print(f"  Bar 1 (primary): depth-4 trained - frozen >= {BAR1_DELTA} at p <= 0.05")
    print(f"  Bar 2 (thesis):  depth-4 trained > the facelets arm MEASURED HERE, paired "
          f"(EXP-033 published {BAR2_FACELET_CEILING} as an external check)")
    print(f"  sanity: frozen depth-4 within {EXP033_FROZEN_D4} +- {SANITY_TOLERANCE}\n",
          flush=True)

    cfg_for = lambda s: PretrainConfig(  # noqa: E731
        seed=s, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, s, cfg_for(s), args.out_dir, quiet=True): s
                       for s in args.seeds}
            for i, fut in enumerate(as_completed(futures), 1):
                fut.result()
                print(f"  {i}/{len(args.seeds)}", flush=True)
    else:
        for s in args.seeds:
            run_one(s, cfg_for(s), args.out_dir)

    print(f"\ndone. one record per seed in {args.out_dir}")
    print("Run aggregate.py for the pre-registered verdicts.")


if __name__ == "__main__":
    main()
