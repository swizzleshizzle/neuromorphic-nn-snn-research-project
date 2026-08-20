# experiments/047_encoder_finetuning/probe_encoders.py
"""EXP-047 Claim 2: did RL improve the REPRESENTATION, or only fit the head to it?

A better success rate is a performance measurement, and `CLAUDE.md` prefers a mechanism one.
EXP-039 measured what a frozen encoder can linearly express about "which moves reduce distance
to solved". Re-running that probe on a FINE-TUNED encoder separates two very different worlds:

    score up,   probe up    ->  the representation genuinely improved
    score up,   probe down  ->  the head fitted itself to a degrading code
    score down, probe down  ->  catastrophic forgetting; the rate is too high

For each seed this probes TWO encoders and pairs them:
    "before"  `exp040_encoder_s{seed}.pt`, the pretrained encoder RL started from
    "after"   the fine-tuned encoder EXP-047 serialised at the end of the run

Both go through the SAME code path, so the comparison is internal to this file and does not
depend on EXP-039's published numbers. EXP-039's Control B logic applies: batching
`SensoryCortex` consumes the Poisson generator differently from looping `brain.step`, so
cross-pipeline comparisons are sanity checks only, never claims.

> [!warning] THE PRE-REGISTERED ASYMMETRY. Read section 6, Claim 2 of the spec before using the
> standard numbers. RL fine-tunes on the RL split; the probe holds out a DIFFERENT split. Most
> states the standard probe scores were seen by the encoder during fine-tuning. So:
>
>   DEGRADATION IS CLEAN     - no leak can make the probe fall.
>   IMPROVEMENT IS CONFOUNDED - it may be memorisation of the very states being scored.
>
> Hence the second probe below.

THE LEAK-FREE SLICE. Depth 6's RL held-out states (200 per seed) are the one set that neither
stage ever saw: EXP-040's `rl_heldout_union` excluded them from PRETRAINING, and `split_shell`
excludes them from RL TRAINING. A second probe is fit with exactly those states held out and
everything else available, so an improvement measured there cannot be memorisation. If the two
probes disagree, the spec pre-commits to reporting the WEAKER of the two.

Run (repo root):
    .venv/bin/python -u experiments/047_encoder_finetuning/probe_encoders.py --mode pilot
    .venv/bin/python -u experiments/047_encoder_finetuning/probe_encoders.py --mode confirm
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.regions.sensory_cortex import SensoryCortex
from neuromorphic.training.cube_baseline import (
    CUBE_N_OBS,
    shell_states,
    split_shell,
)
from neuromorphic.training.encoder_pretrain import concept_rates, states_to_obs

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
ENCODERS = Path("experiments/040_pretrained_encoder_policy/outputs")

# EXP-033's probe, imported UNMODIFIED and by file path, exactly as EXP-039 imports it. Same
# labels, same fit, same top-1 definition - which is the entire reason these numbers can be set
# beside EXP-039's at all.
_PROBE_PATH = HERE.parent / "033_concept_decodability" / "probe.py"
_spec = importlib.util.spec_from_file_location("exp033_probe", _PROBE_PATH)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

DEPTHS = [1, 2, 3, 4, 5, 6]          # EXP-039's set, unchanged
REPORT_DEPTHS = [3, 4, 5, 6]
HEADLINE_DEPTH = 4                    # EXP-039's primary depth, and the gate's depth
FRAC_HELDOUT = 0.25                   # EXP-033's value, unchanged
LEAKFREE_DEPTH = 6
HELDOUT_CAP = 200                     # matches CubeConfig's default, so the RL split reproduces
PROBE_EPOCHS = 300                    # EXP-039's `_evaluate`, unchanged
PROBE_LR = 0.1


def build_dataset(provider):
    """All states at DEPTHS with their optimal-move masks. Mirrors EXP-039's build_dataset."""
    states, masks, depths = [], [], []
    for d in DEPTHS:
        for s in provider.states_at_distance(d):
            states.append(s)
            masks.append(probe.optimal_move_mask(provider, s))
            depths.append(d)
    return states, masks, depths


def encoder_features(sensory, states, seed: int, batch: int = 512) -> torch.Tensor:
    """Concept rates for every state, batched. EXP-039's function, unchanged.

    Same generator seeding for both encoders of a pair, so "before" and "after" see the same
    Poisson draws and the difference is the weights alone.
    """
    obs = states_to_obs(states)
    gen = torch.Generator().manual_seed(seed)
    rows = []
    with torch.no_grad():
        for start in range(0, len(states), batch):
            rows.append(concept_rates(sensory, obs[start:start + batch], generator=gen))
    return torch.cat(rows, dim=0)


def _evaluate(x, masks, depths, idx_train, idx_held, seed) -> dict:
    """EXP-033's `_evaluate` as EXP-039 reused it: one joint fit, reported per depth."""
    model = probe.fit_linear_probe(
        x[idx_train], [masks[i] for i in idx_train], epochs=PROBE_EPOCHS, lr=PROBE_LR, seed=seed
    )
    with torch.no_grad():
        logits = model(x[idx_held])
    held_masks = [masks[i] for i in idx_held]
    out = {"top1": probe.top1_accuracy(logits, held_masks),
           "chance": probe.chance_floor(held_masks), "by_depth": {}}
    for d in DEPTHS:
        sel = [j for j, i in enumerate(idx_held) if depths[i] == d]
        if sel:
            out["by_depth"][str(d)] = {
                "top1": probe.top1_accuracy(logits[sel], [held_masks[j] for j in sel]),
                "chance": probe.chance_floor([held_masks[j] for j in sel]),
                "n": len(sel),
            }
    return out


def standard_split(depths, seed):
    """EXP-039's stratified split, so no depth's shell can dominate the held-out set."""
    idx_train, idx_held = [], []
    for d in DEPTHS:
        pool = [i for i, dd in enumerate(depths) if dd == d]
        tr, he = probe.split_states(pool, FRAC_HELDOUT, seed)
        idx_train += tr
        idx_held += he
    return idx_train, idx_held


def leakfree_split(states, provider, seed):
    """Held out = depth-6 states RL never trained on AND pretraining never saw.

    `split_shell` is keyed on `split_seed`, which falls back to `seed` in `CubeConfig`, so this
    reproduces exactly the evaluation set the RL run was scored on. EXP-040's `rl_heldout_union`
    excluded the same states from inverse-model pretraining. They are therefore the only states
    in the dataset that are clean at BOTH stages, which is what makes an improvement measured
    here impossible to explain as memorisation.
    """
    _, eval_states, _ = split_shell(
        shell_states(provider, LEAKFREE_DEPTH), LEAKFREE_DEPTH, seed=seed,
        heldout_cap=HELDOUT_CAP, heldout_frac=FRAC_HELDOUT,
    )
    held = set(eval_states)
    idx_held = [i for i, s in enumerate(states) if s in held]
    held_idx = set(idx_held)          # hoisted: rebuilding it per element is O(n^2) over ~11,900
    idx_train = [i for i in range(len(states)) if i not in held_idx]
    return idx_train, idx_held


def load_sensory(path: Path) -> SensoryCortex:
    """A sensory region matching the one `Brain` builds for the cube, with `path`'s weights.

    `strict=True` deliberately, for EXP-040's reason: a silently partial load would leave half a
    random encoder in place and every probe number would describe an architecture nobody chose.
    """
    sensory = SensoryCortex(n_obs=CUBE_N_OBS, concept=64, num_steps=32, seed=0)
    sensory.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    return sensory


def probe_one(seed: int, after_path: Path, encoder_lr: float) -> dict:
    """Probe the before/after encoder pair for one seed. Returns one provenance row."""
    torch.set_num_threads(1)
    t0 = time.time()

    provider = ExactBFSDistance(max_depth=max(DEPTHS) + 1)
    states, masks, depths = build_dataset(provider)
    std_train, std_held = standard_split(depths, seed)
    lf_train, lf_held = leakfree_split(states, provider, seed)

    before = load_sensory(ENCODERS / f"exp040_encoder_s{seed}.pt")
    after = load_sensory(after_path)

    row = {"seed": seed, "encoder_lr": encoder_lr, "after_path": str(after_path),
           "n_states": len(states), "n_heldout_standard": len(std_held),
           "n_heldout_leakfree": len(lf_held)}

    for name, sensory in (("before", before), ("after", after)):
        x = encoder_features(sensory, states, seed)
        row[name] = _evaluate(x, masks, depths, std_train, std_held, seed)
        row[f"{name}_leakfree"] = _evaluate(x, masks, depths, lf_train, lf_held, seed)

    b = row["before"]["by_depth"][str(HEADLINE_DEPTH)]["top1"]
    a = row["after"]["by_depth"][str(HEADLINE_DEPTH)]["top1"]
    row["delta_headline"] = a - b
    lb = row["before_leakfree"]["by_depth"][str(LEAKFREE_DEPTH)]["top1"]
    la = row["after_leakfree"]["by_depth"][str(LEAKFREE_DEPTH)]["top1"]
    row["delta_leakfree"] = la - lb
    row["wall_seconds"] = round(time.time() - t0, 1)
    return row


def discover(out_dir: Path, mode: str):
    """Find every fine-tuned encoder EXP-047 wrote, as (seed, path, encoder_lr).

    Reads the seed and rate out of the RECORD rather than parsing the filename, so a renamed or
    re-tagged cell cannot be silently mis-attributed to the wrong rate.
    """
    found = []
    for rec_path in sorted(out_dir.glob("exp047_ft_d*.json")):
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        enc = rec_path.with_name(rec_path.name.replace(".json", "_encoder.pt"))
        if not enc.exists():
            raise SystemExit(
                f"{rec_path.name} has no serialised encoder beside it ({enc.name}). Claim 2 "
                "cannot be asked of a run whose weights were never written."
            )
        found.append((rec["seed"], enc, rec["config"]["encoder_lr"]))
    if mode == "pilot":
        found = [f for f in found if f[0] >= 12]
    else:
        found = [f for f in found if f[0] < 12]
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "confirm"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = discover(args.out_dir, args.mode)
    if not targets:
        raise SystemExit(f"no EXP-047 {args.mode} records with encoders in {args.out_dir}")

    dest = args.out_dir / f"probe_{args.mode}.json"
    print(f"EXP-047 Claim 2 probe: {len(targets)} encoder pair(s), {args.workers} workers")
    print(f"  standard split: EXP-039's stratified {FRAC_HELDOUT:.0%}, headline depth "
          f"{HEADLINE_DEPTH}")
    print(f"  leak-free slice: depth-{LEAKFREE_DEPTH} RL held-out states, clean at BOTH stages")
    print(f"  IMPROVEMENT ON THE STANDARD SPLIT IS CONFOUNDED. Degradation is not. Spec Claim 2.")
    print(f"  -> {dest}\n", flush=True)
    if args.dry_run:
        print(f"  --dry-run: {len(targets)} pair(s) NOT started.")
        return

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(probe_one, s, p, lr): s for s, p, lr in targets}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            print(f"  {i}/{len(targets)} seed {row['seed']} lr {row['encoder_lr']:g}: "
                  f"d{HEADLINE_DEPTH} {row['before']['by_depth'][str(HEADLINE_DEPTH)]['top1']:.3f}"
                  f" -> {row['after']['by_depth'][str(HEADLINE_DEPTH)]['top1']:.3f} "
                  f"({row['delta_headline']:+.4f}), leak-free d{LEAKFREE_DEPTH} "
                  f"{row['delta_leakfree']:+.4f}, {row['wall_seconds']}s", flush=True)

    dest.write_text(json.dumps(sorted(rows, key=lambda r: (r["encoder_lr"], r["seed"])), indent=1),
                    encoding="utf-8")
    print(f"\ndone. {len(rows)} row(s) in {dest}")


if __name__ == "__main__":
    main()
