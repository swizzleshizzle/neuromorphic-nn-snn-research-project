"""Record two REAL depth-6 rollouts from the SAME held-out scramble, for `PolicyCollapse`.

**Runs in the PROJECT venv, not the manim one.** It needs torch and the brain; manim does not
need either. Its output, `traces.json`, is committed, so the scene replays a recorded rollout
instead of importing the training stack.

WHY RECORD RATHER THAN DRAW
The scene is a claim about what these policies do. Hand-animating "a collapsed policy repeats a
move" would be an illustration of a belief; this replays what the checkpoint actually did. The
repo has already had a cube frame labelled `solved: yes` on a scrambled cube pass every unit
test, so a picture nobody checked against the model is not evidence.

THE TWO ARMS - both at depth 6, both greedy, same scramble, and NOT the same experiment:

  left   EXP-036 seed 3. Frozen random encoder. `greedy_modal_action_frac` **1.000** and success
         **0.0000** - it plays one move forever. Four of its twelve seeds do exactly this.
  right  EXP-043 seed 11, today's recipe (pretrained encoder + the depth-1 cap). Success 0.315
         at depth 6, the best of its twelve seeds.

That crossing is deliberate and the scene must caption it: EXP-036 has NO working seed at depth 6
to pair against - all twelve score 0.0000 - so a same-experiment pairing does not exist at this
depth. The alternative, depth 3, has no fully collapsed seed (modal tops out at 0.793).

Each config is rebuilt from its OWN record's `config` block rather than re-specified here, so a
rollout cannot silently differ from the run that produced the checkpoint.

Usage (repo root):
    .venv/bin/python viz/manim/record_traces.py
"""

from __future__ import annotations

import json
import random
from dataclasses import fields
from pathlib import Path

import torch
from torch import nn

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from neuromorphic.envs.cube import MOVE_LABELS, SOLVED, CubeEnv  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.analysis.ablate import AblatedConcept, AblationSpec  # noqa: E402
from neuromorphic.training.cube_baseline import (  # noqa: E402
    CubeConfig,
    feature_width,
    make_agent,
    max_steps_for,
    shell_states,
    split_shell,
)
from neuromorphic.training.reinforce import greedy_action  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "viz" / "manim" / "traces.json"
DEPTH = 6

LEFT = {"exp": "036_generalisation_gap", "tag": "exp036_d6_e10000", "seed": 3,
        "name": "EXP-036, frozen encoder"}
RIGHT = {"exp": "043_cap_at_depth_5_6", "tag": "exp043_capped_d6", "seed": 11,
         "name": "EXP-043, trained encoder + cap"}

_FIELDS = {f.name for f in fields(CubeConfig)}
_TUPLES = {"curriculum", "curriculum_weights", "max_steps_by_depth"}


def load_arm(spec: dict) -> tuple[CubeConfig, dict, object, nn.Module]:
    """Rebuild the exact config the checkpoint was trained under, then load the head."""
    out_dir = REPO / "experiments" / spec["exp"] / "outputs"
    # `_regionalized_` matters: EXP-036 wrote a `random` floor record under the same tag and
    # seed, and matching both would pick a checkpoint-less floor run.
    hits = [p for p in sorted(out_dir.glob("*.json"))
            if spec["tag"] in p.name and f"_s{spec['seed']}_" in p.name
            and "_regionalized_" in p.name]
    if len(hits) != 1:
        raise SystemExit(f"expected 1 record for {spec}, found {[p.name for p in hits]}")
    rec = json.loads(hits[0].read_text())

    raw = {k: v for k, v in rec["config"].items() if k in _FIELDS}
    for k in _TUPLES:
        if isinstance(raw.get(k), list):
            # `max_steps_by_depth` is a tuple OF tuples and JSON flattens both levels.
            raw[k] = tuple(tuple(x) if isinstance(x, list) else x for x in raw[k])
    raw["out_dir"] = out_dir
    if raw.get("encoder_state_path"):
        # Recorded on Windows, so it is a BACKSLASH path and `Path(...).name` returns the whole
        # string here - on Linux a backslash is an ordinary character, not a separator.
        base = raw["encoder_state_path"].replace("\\", "/").rsplit("/", 1)[-1]
        raw["encoder_state_path"] = str(REPO / "experiments" /
                                        "040_pretrained_encoder_policy" / "outputs" / base)
    cfg = CubeConfig(**raw)

    head_path = hits[0].with_name(hits[0].name.replace(".json", "_head.pt"))
    if not head_path.exists():
        raise SystemExit(f"missing head checkpoint {head_path}")

    agent = make_agent(cfg)
    spec_ab = (AblationSpec(kind="gaussian", dose=cfg.sigma, seed=cfg.train_seed or cfg.seed)
               if cfg.sigma else None)
    head = AblatedConcept(nn.Linear(feature_width(cfg), cfg.n_actions), spec_ab,
                          width=feature_width(cfg))
    head.load_state_dict(torch.load(head_path, map_location="cpu"), strict=True)
    head.eval()
    return cfg, rec, agent, head


def rollout(agent, head, state, *, depth: int, seed: int) -> dict:
    """One greedy rollout, recording every facelet state. Mirrors `evaluate_states`' loop."""
    limit = max_steps_for(depth)
    env = CubeEnv(scramble_depth=depth, max_steps=limit, scramble_seed=seed)
    obs, _ = env.reset(options={"state": state})
    frames = [list(env._state)]
    actions: list[int] = []
    solved = False
    for _ in range(limit):
        with torch.no_grad():
            action = int(greedy_action(agent, head, obs))
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(list(env._state))
        if terminated:
            solved = True
            break
        if truncated:
            break
    return {"frames": frames, "actions": actions,
            "labels": [MOVE_LABELS[a] for a in actions], "solved": solved,
            "modal": max(actions.count(a) for a in set(actions)) / len(actions) if actions else 0.0}


def main() -> None:
    lcfg, lrec, lagent, lhead = load_arm(LEFT)
    rcfg, rrec, ragent, rhead = load_arm(RIGHT)
    print(f"left  {LEFT['name']}: modal {lrec['greedy_modal_action_frac']:.3f}  "
          f"success {lrec['success_rate']:.4f}")
    print(f"right {RIGHT['name']}: modal {rrec['greedy_modal_action_frac']:.3f}  "
          f"success {rrec['success_rate']:.4f}")

    # Search the RIGHT arm's own held-out split, so the state is one it never trained on. Its
    # split is reproduced from its own config, not re-drawn here.
    states = shell_states(ExactBFSDistance(max_depth=DEPTH), DEPTH)
    _, heldout, _ = split_shell(states, DEPTH, seed=rcfg.split_seed or rcfg.seed,
                                heldout_cap=rcfg.heldout_cap, heldout_frac=rcfg.heldout_frac)
    rng = random.Random(0)
    candidates = list(heldout)
    rng.shuffle(candidates)

    chosen = None
    for i, state in enumerate(candidates[:40]):
        right = rollout(ragent, rhead, state, depth=DEPTH, seed=rcfg.seed)
        if not right["solved"]:
            continue
        left = rollout(lagent, lhead, state, depth=DEPTH, seed=lcfg.seed)
        if left["solved"]:
            continue                      # it never does, but assert it rather than assume it
        chosen = (state, left, right)
        print(f"  state {i}: right solves in {len(right['actions'])}, "
              f"left plays {left['labels'][0]} x{len(left['actions'])}")
        break
    if chosen is None:
        raise SystemExit("no held-out state in the first 40 where right solves and left does not")

    state, left, right = chosen
    payload = {
        "depth": DEPTH,
        "scramble": list(state),
        "solved_state": list(SOLVED),
        "step_budget": max_steps_for(DEPTH),
        "left": {**left, "name": LEFT["name"], "seed": LEFT["seed"],
                 "arm_modal": lrec["greedy_modal_action_frac"],
                 "arm_success": lrec["success_rate"]},
        "right": {**right, "name": RIGHT["name"], "seed": RIGHT["seed"],
                  "arm_modal": rrec["greedy_modal_action_frac"],
                  "arm_success": rrec["success_rate"]},
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT} ({len(left['frames'])} left frames, {len(right['frames'])} right frames)")
    print(f"  left solved={left['solved']}  right solved={right['solved']}")


if __name__ == "__main__":
    main()
