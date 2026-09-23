"""Does RE-EVALUATING a tracked checkpoint reproduce across machines?

Training does not: EXP-036 depth 3 seed 0 regenerated here shares **0 of 390** parameters with
its published head (cosine 0.524), and it uses no pretrained encoder. So retraining from a
fresh clone on another machine reproduces nothing.

But `.gitignore` tracks `*_head.pt` for a stated reason - it is "what makes 'never retrain to
re-evaluate' true from a fresh checkout". **That is the claim this tests.** If evaluation is
portable, the project's reproducibility story is "re-evaluate the tracked checkpoints", which is
a real and defensible guarantee. If it is not, nothing reproduces off this laptop and Phase 4
must say so plainly.

Evaluation is cheap - no training - so this runs in about a minute per cell on either machine.
Prints the held-out success rate and a digest of the full action sequence, which is far more
sensitive: two runs can agree on a success rate while taking different paths.

Usage:
    .venv/bin/python -u scripts/verify_eval_portability.py --depth 3 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import random
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments" / "036_generalisation_gap"
_spec = importlib.util.spec_from_file_location("exp036_run", EXP / "run.py")
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training.cube_baseline import (  # noqa: E402
    AblatedConcept, evaluate_states, feature_width, make_agent, record_filename,
    resolve_seed, shell_states, split_shell,
)
import torch.nn as nn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    torch.set_num_threads(1)

    published = EXP / "outputs"
    configs = [c for c in _run.sweep_configs(args.seeds, published)
               if c.arm == "regionalized" and c.depth == args.depth and c.seed in args.seeds]

    print(f"EVALUATION PORTABILITY - EXP-036 depth {args.depth}, seeds {args.seeds}")
    print( "  Loads the TRACKED head and re-evaluates it. No training.")
    print( "  Run this on BOTH machines and compare: identical rows mean 'never retrain to")
    print( "  re-evaluate' is a real guarantee from a fresh checkout.\n", flush=True)
    print(f"{'seed':>5s} {'success':>9s} {'optimality':>11s} {'action_digest':>16s}")

    provider = ExactBFSDistance(max_depth=max(6, args.depth))
    for cfg in sorted(configs, key=lambda c: c.seed):
        head_file = published / record_filename(cfg).replace(".json", "_head.pt")
        if not head_file.exists():
            print(f"{cfg.seed:>5d}   MISSING {head_file.name}")
            continue
        agent = make_agent(cfg)
        head = AblatedConcept(nn.Linear(feature_width(cfg), cfg.n_actions), None,
                              width=feature_width(cfg))
        head.load_state_dict(torch.load(head_file, map_location="cpu"))
        split_seed = resolve_seed(cfg, "split")
        _train, eval_states, _ = split_shell(
            shell_states(provider, cfg.depth), cfg.depth, seed=split_seed,
            heldout_cap=cfg.heldout_cap, heldout_frac=cfg.heldout_frac,
        )
        gen = torch.Generator().manual_seed(resolve_seed(cfg, "train"))
        res = evaluate_states(agent, head, eval_states, depth=cfg.depth, generator=gen,
                              rng_seed=resolve_seed(cfg, "train"))
        # A digest over the evaluation outcome, finer than the headline rate.
        payload = f"{res['success_rate']!r}|{res['mean_steps']!r}|{res['optimality']!r}|" \
                  f"{res['eval_revisit_rate']!r}|{res['greedy_modal_action_frac']!r}"
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        print(f"{cfg.seed:>5d} {res['success_rate']:>9.4f} {res['optimality']:>11.4f} "
              f"{digest:>16s}", flush=True)


if __name__ == "__main__":
    main()
