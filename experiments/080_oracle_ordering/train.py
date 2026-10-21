"""EXP-080 training driver: one (arm, seed) per job, MSE against the minted true distance.

O-Xf and O-Xt train on block u of the seed's mint at update u (paired on identical states),
4000 updates of 1000, checkpointing every 250 so a run killed by the 03:31 restart resumes where
it stopped. O-raw trains 12 epochs over the mint's first 600,000 states. No BFS table is built
here: labels, the probe set and every probe leaf's distance all come from the mint, and Gate L is
measured at the end from those minted distances.

Spec: docs/superpowers/specs/2026-10-20-exp080-oracle-ordering-ceiling-design.md, section 3.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/080_oracle_ordering/train.py --pilot --seeds 12 13 \
        --n-updates 4000 --workers 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp080_cells", HERE / "cells.py")
mint = _load("exp080_mint", HERE / "mint.py")
gate_l = _load("exp074_gate_l", REPO / "experiments" / "074_wide_judge" / "gate_l.py")

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402

torch.set_num_threads(1)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _block(m, u, batch):
    """Update u's states and labels: rows u*batch to (u+1)*batch of the mint."""
    lo, hi = u * batch, (u + 1) * batch
    return m.states[lo:hi], torch.as_tensor(m.labels[lo:hi], dtype=torch.float32)


def _probe(judge, m, seed):
    return vi.probe_judge(judge, m.probe, torch.Generator().manual_seed(seed))


def train_x_arm(arm, judge, m, *, n_updates, batch, probe_every, seed, ckpt_dir, log=print):
    need = n_updates * batch
    if len(m.labels) < need:
        raise SystemExit(f"mint for seed {seed} holds {len(m.labels)} states; {arm} at "
                         f"{n_updates} x {batch} needs {need}")
    ck = Path(ckpt_dir)
    ck.mkdir(parents=True, exist_ok=True)
    state_path = ck / "state.json"
    opt = cells.optimizer_for(arm, judge)
    gen = torch.Generator().manual_seed(seed)
    start, probes, losses = 0, [], []
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("mint_sha") != m.sha:
            raise SystemExit(f"{state_path}: checkpoint was trained on a different mint "
                             f"({state.get('mint_sha')}) than seed {seed}'s now ({m.sha}); delete "
                             f"{ck} and retrain")
        judge.load_state_dict(torch.load(ck / "judge.pt"))
        opt.load_state_dict(torch.load(ck / "opt.pt"))
        gen.set_state(torch.tensor(state["generator_state"], dtype=torch.uint8))
        start, probes, losses = state["update"], state["probes"], state["losses"]
    t0 = time.time()
    loss_last = None
    for u in range(start, n_updates):
        xs, y = _block(m, u, batch)
        loss = F.mse_loss(judge(xs, gen), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_last = loss.item()
        if u % probe_every == 0 or u == n_updates - 1:
            pr = _probe(judge, m, seed)
            probes.append(pr)
            losses.append([u, loss_last])
            torch.save(judge.state_dict(), ck / "judge.pt")
            torch.save(opt.state_dict(), ck / "opt.pt")
            _write_json(state_path, {"arm": arm, "update": u + 1, "mint_sha": m.sha,
                                     "generator_state": gen.get_state().tolist(),
                                     "probes": probes, "losses": losses})
            log(f"{arm} s{seed} update {u}: loss {loss_last:.6f} "
                f"spearman_7_11 {pr['spearman_7_11']}")
    return {"updates": n_updates, "resumed_from": start, "wall_s": time.time() - t0,
            "loss_last": loss_last, "probes": probes, "losses": losses,
            "final_probe": probes[-1] if probes else _probe(judge, m, seed)}


def train_oraw(judge, m, *, n_states, epochs, batch, seed, ckpt_dir, log=print):
    """Minutes per seed, so no mid-run checkpoint: an interrupted run retrains from scratch."""
    if len(m.labels) < n_states:
        raise SystemExit(f"mint for seed {seed} holds {len(m.labels)} states; O-raw needs "
                         f"{n_states}")
    ck = Path(ckpt_dir)
    ck.mkdir(parents=True, exist_ok=True)
    x = cells.RawJudge.one_hot(m.states[:n_states])
    y = torch.as_tensor(m.labels[:n_states], dtype=torch.float32)
    opt = cells.optimizer_for("O-raw", judge)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    losses, steps, loss_last = [], 0, None
    for ep in range(epochs):
        perm = torch.randperm(n_states, generator=g)
        for i in range(0, n_states, batch):
            b = perm[i:i + batch]
            loss = F.mse_loss(judge.net(x[b]).squeeze(-1), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
            loss_last = loss.item()
        losses.append([ep, loss_last])
        log(f"O-raw s{seed} epoch {ep}: loss {loss_last:.6f}")
    pr = _probe(judge, m, seed)
    torch.save(judge.state_dict(), ck / "judge.pt")
    _write_json(ck / "state.json", {"arm": "O-raw", "epochs": epochs, "mint_sha": m.sha})
    return {"updates": steps, "epochs": epochs, "resumed_from": 0, "wall_s": time.time() - t0,
            "loss_last": loss_last, "probes": [pr], "losses": losses, "final_probe": pr}


def run(arm, seed, out_dir, *, n_updates=cells.N_UPDATES, batch=cells.BATCH,
        probe_every=cells.PROBE_EVERY, raw_states=cells.RAW_STATES, raw_epochs=cells.RAW_EPOCHS,
        raw_batch=cells.RAW_BATCH, mint_root=None, x_root=None, enc_root=None) -> dict:
    torch.set_num_threads(1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mint_root = out_dir if mint_root is None else mint_root
    judge = cells.make_judge(arm, seed, x_root, enc_root)  # a missing input fails first
    m = mint.read_mint(mint_root, seed)
    ck = cells.ckpt_dir(arm, seed, out_dir)
    if arm == "O-raw":
        result = train_oraw(judge, m, n_states=raw_states, epochs=raw_epochs, batch=raw_batch,
                            seed=seed, ckpt_dir=ck)
        settings = {"raw_states": raw_states, "raw_epochs": raw_epochs, "raw_batch": raw_batch,
                    "raw_hidden": cells.RAW_HIDDEN}
    else:
        result = train_x_arm(arm, judge, m, n_updates=n_updates, batch=batch,
                             probe_every=probe_every, seed=seed, ckpt_dir=ck)
        settings = {"n_updates": n_updates, "batch": batch, "probe_every": probe_every,
                    "encoder_drift": cells.drift(judge.sensory.state_dict(),
                                                 cells.reference_sensory(arm, seed, x_root,
                                                                         enc_root))}
    judge.eval()
    rec = {**result, **settings, "arm": arm, "seed": seed, "mint_sha": m.sha,
           "mint_meta": m.meta, "n_actions": N_ACTIONS,
           "gate_l": gate_l.leaf_rank_margin(judge, m.probe, m.leaf_dist.__getitem__, seed,
                                             N_ACTIONS),
           "git_commit": _git_commit()}
    _write_json(out_dir / cells.record_name(arm, seed), rec)
    return rec


def pending_jobs(jobs, out_dir) -> list:
    return [(a, s) for a, s in jobs if not (Path(out_dir) / cells.record_name(a, s)).exists()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", choices=list(cells.ARMS), default=list(cells.TRAIN_ORDER))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    # Required, with no default: a run must state its budget.
    ap.add_argument("--n-updates", type=int, required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="outputs/ (outputs_pilot/ with --pilot) when omitted; also the mint root")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cells.check_seeds(args.seeds, args.pilot)
    out = args.out_dir if args.out_dir is not None else (
        cells.PILOT_OUT if args.pilot else cells.OUT)
    x_root, enc_root = cells.x_judge_root(args.pilot), cells.encoder_root(args.pilot)
    jobs = pending_jobs([(a, s) for a in args.arms for s in args.seeds], out)
    print(f"EXP-080 train: {len(jobs)} runs, {args.workers} workers, pilot={args.pilot}, "
          f"n_updates={args.n_updates}, out {out}, X judges {x_root}, encoders {enc_root}",
          flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run, a, s, out, n_updates=args.n_updates, x_root=x_root,
                            enc_root=enc_root): (a, s) for a, s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            drift = r.get("encoder_drift")
            # None when the session resumed a finished checkpoint and trained nothing.
            loss = "None" if r["loss_last"] is None else format(r["loss_last"], ".6f")
            print(f"  {i}/{len(jobs)}  {r['arm']} seed {r['seed']}  updates {r['updates']}  "
                  f"loss_last {loss}  wall_s {r['wall_s']:.1f}  "
                  f"drift {drift}  gate_l margin {r['gate_l']['margin']:.4f}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
