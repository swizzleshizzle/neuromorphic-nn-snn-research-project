"""EXP-084 aggregator: Gates C, P, E, M and T, the reported block, and the one-line verdict.

`--stage p` reads only Gate P and prints CONTINUE or HALT (the chain stops on HALT, before Gate E).
`--stage all` reads everything. Writes nothing; the chain tees the output.

Spec: docs/superpowers/specs/2026-10-31-exp084-gpu-port-design.md, section 5.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python experiments/084_gpu_port/aggregate.py --stage all
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp084_cells", HERE / "cells.py")
a80 = _load("exp080_aggregate", cells.REPO / "experiments" / "080_oracle_ordering" / "aggregate.py")
one_sided_p = a80.one_sided_p
N_P = len(cells.ARMS) * len(cells.SEEDS)


def within(got: int, want: int) -> bool:
    return abs(got - want) <= cells.TOL


def verdict(c_ok: bool, e_ok: bool, m_ok: bool, t_ok: bool) -> str:
    if not c_ok:
        return "VOID"
    if not e_ok:
        return "DESKTOP DIFFERS FROM LAPTOP ON CPU TOO" if not m_ok else "GPU PORT DOES NOT REPRODUCE"
    if not t_ok:
        return "GPU SEARCH REPRODUCES, TRAINING DOES NOT LEARN"
    return "GPU PORT REPRODUCES"


def _read(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing record {path.name}; run that stage first")
    return json.loads(path.read_text())


def _full(rec: dict, name: str) -> dict:
    if rec.get("limit_states") is not None or rec.get("n") != 200:
        raise SystemExit(f"{name}: not a full cell (n {rec.get('n')}, limit_states "
                         f"{rec.get('limit_states')})")
    return rec


def _pairs():
    return [(a, s) for s in cells.SEEDS for a in cells.ARMS]


def gate_p(out: Path) -> dict:
    recs = [_full(_read(out / cells.p_name(a, s)), cells.p_name(a, s)) for a, s in _pairs()]
    dis = sum(r["disagree"] for r in recs)
    halt = dis > cells.P_LIMIT
    lines = [f"GATE P (first moves, CPU vs device): {sum(r['agree'] for r in recs)} of "
             f"{sum(r['n'] for r in recs)} agree, {dis} disagree "
             f"({sum(r['near_ties'] for r in recs)} near-ties), max |dJ| "
             f"{max(r['max_abs_dj'] for r in recs):.3g}, goal check fired on "
             f"{sum(r['goal_fired'] for r in recs)}",
             f"GATE P: HALT, ask Michael before Gate E ({dis} > {cells.P_LIMIT})" if halt
             else f"GATE P: CONTINUE ({dis} <= {cells.P_LIMIT})"]
    return {"halt": halt, "disagree": dis, "lines": lines}


def _e_line(label, rec) -> str:
    ok = within(rec["solved"], rec["published"])
    return (f"  {label}: {rec['solved']} vs published {rec['published']} "
            f"({rec['solved'] - rec['published']:+d}) {'ok' if ok else 'OUTSIDE 2'}")


def run(stage: str, out=cells.OUT) -> dict:
    out = Path(out)
    p = gate_p(out)
    if stage == "p":
        return {"halt": p["halt"], "lines": p["lines"]}
    lines = []
    c = _read(out / cells.GATEC_NAME)
    c_ok = c.get("pass") is True
    lines.append(f"GATE C (CPU path unchanged): {'PASS' if c_ok else 'FAIL'}, differences "
                 f"{c.get('differences')}")
    lines += p["lines"]
    if p["halt"]:
        return {"halt": True, "verdict": None, "lines": lines}

    e = {(a, s): _full(_read(out / cells.e_name(a, s)), cells.e_name(a, s)) for a, s in _pairs()}
    gr = _full(_read(out / cells.E_GATER_NAME), cells.E_GATER_NAME)
    e_all = list(e.values()) + [gr]
    e_ok = all(within(r["solved"], r["published"]) for r in e_all)
    lines.append(f"GATE E (re-evaluation on the device, within {cells.TOL}): "
                 f"{'PASS' if e_ok else 'FAIL'}, {sum(r['solved'] == r['published'] for r in e_all)}"
                 f" of {len(e_all)} exact, summed |diff| "
                 f"{sum(abs(r['solved'] - r['published']) for r in e_all)}")
    lines += [_e_line(f"{a} s{s}", e[a, s]) for a, s in _pairs()]
    lines.append(_e_line("EXP-082 gate R cell", gr))

    m = _full(_read(out / cells.M_NAME), cells.M_NAME)
    m_ok = within(m["solved"], m["published"])
    lines.append(f"GATE M (desktop CPU control): {'PASS' if m_ok else 'FAIL'}, "
                 f"{m['solved']} vs {m['published']}")

    t = [_read(out / cells.train_name(s)) for s in cells.SEEDS]
    for s, r in zip(cells.SEEDS, t):
        if r.get("n_updates") != cells.N_UPDATES:
            raise SystemExit(f"{cells.train_name(s)}: n_updates {r.get('n_updates')}, "
                             f"Gate T needs {cells.N_UPDATES}")
    cpu = [_read(cells.E75_OUT / f"exp075_train_X_s{s}.json") for s in cells.SEEDS]
    gpu_o, cpu_o = [a80.overall(r) for r in t], [a80.overall(r) for r in cpu]
    t_ok = a80.gate_t(gpu_o)
    lines.append(f"GATE T (device-trained X learns): {'PASS' if t_ok else 'FAIL'}, mean overall "
                 f"margin {st.mean(gpu_o):.4f} vs {a80.GATE_T_THRESHOLD:.5f}")

    v = verdict(c_ok, e_ok, m_ok, t_ok)
    lines.append(f"VERDICT: {v}")
    lines.append("REPORTED (never gating):")
    for label, fn in (("overall margin", a80.overall), ("far band", a80.far)):
        g, c_ = [fn(r) for r in t], [fn(r) for r in cpu]
        diffs = [a - b for a, b in zip(g, c_)]
        lines.append(f"  device X minus EXP-075 CPU X, {label}: {st.mean(g):.4f} vs "
                     f"{st.mean(c_):.4f}, diff {st.mean(diffs):+.4f}, ahead "
                     f"{sum(d > 0 for d in diffs)} of 12, p {one_sided_p(diffs):.4f}")
    lines.append(f"  training s/update on the device: mean {st.mean(r['s_per_update'] for r in t):.3f}"
                 f" (min {min(r['s_per_update'] for r in t):.3f}, max "
                 f"{max(r['s_per_update'] for r in t):.3f})")
    lines.append(f"  Gate E search wall per cell: mean {st.mean(r['wall_s'] for r in e_all):.0f} s "
                 f"(min {min(r['wall_s'] for r in e_all):.0f}, max "
                 f"{max(r['wall_s'] for r in e_all):.0f}); Gate M on the desktop CPU {m['wall_s']:.0f} s")
    return {"halt": False, "verdict": v, "lines": lines}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("p", "all"), default="all")
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    for line in run(args.stage, args.out_dir)["lines"]:
        print(line)


if __name__ == "__main__":
    main()
