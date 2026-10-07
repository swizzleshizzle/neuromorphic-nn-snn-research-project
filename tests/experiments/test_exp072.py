"""EXP-072: P3V at depths 8 and 9, paired against EXP-070's committed records."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "072_p3v_frontier"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, EXP / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = _load("exp072_run", "run.py")
agg = _load("exp072_aggregate", "aggregate.py")


def test_arms_are_exactly_the_spec_list():
    """Catches a missing or extra arm; continuity arms are separate and limited to seed 0."""
    assert [run.arm_name(*a) for a in run.ARMS] == ["G0V", "P3V", "R3V"]
    assert [run.arm_name(*a) for a in run.CONTINUITY_ARMS] == ["G0", "P3"]
    assert run.DEPTHS == (8, 9)


@pytest.mark.parametrize("bad", ["P1", "C3", "E3V", "G0V8"])
def test_parse_arm_rejects_arms_outside_the_spec(bad):
    """Catches an arm the spec never registered sneaking in through the CLI."""
    with pytest.raises(SystemExit):
        run.parse_arm(bad)


def test_record_names_are_unique():
    """Catches records overwriting each other across arms, depths and seeds."""
    names = [run.cell_record_name(*a, d, s) for a in run.ARMS + run.CONTINUITY_ARMS
             for d in run.DEPTHS for s in range(12)]
    assert len(names) == len(set(names))


def test_run_cell_writes_a_complete_record(tmp_path):
    """Catches a record missing a field the aggregator reads, or V not reaching the rollout."""
    rec = run.run_cell("G", 0, True, 8, 0, tmp_path, limit_states=2)
    on_disk = json.loads((tmp_path / run.cell_record_name("G", 0, True, 8, 0)).read_text())
    assert on_disk == rec
    for key in ("solved", "n", "success_rate", "eval_revisit_rate", "fallback_frac",
                "no_revisit", "mode", "k", "arm", "depth", "seed", "limit_states"):
        assert key in rec, key
    assert rec["arm"] == "G0V" and rec["no_revisit"] is True and rec["depth"] == 8
    assert rec["eval_revisit_rate"] == 0.0


def test_plain_continuity_arm_runs_without_the_rule(tmp_path):
    """Catches no_revisit leaking into a plain arm, which would void Gate 0(b)."""
    rec = run.run_cell("G", 0, False, 9, 0, tmp_path, limit_states=2)
    assert rec["no_revisit"] is False and rec["arm"] == "G0"


def _rates(value_by_seed):
    return {s: v for s, v in enumerate(value_by_seed)}


def test_pair_diffs_refuses_mismatched_seed_sets():
    """Catches pairing arms over different seed sets (the seed confound, CLAUDE.md)."""
    with pytest.raises(ValueError):
        agg.pair_diffs({0: 0.1, 1: 0.2}, {0: 0.1, 2: 0.2})
    assert agg.pair_diffs({0: 0.3, 1: 0.2}, {0: 0.1, 1: 0.2}) == pytest.approx([0.2, 0.0])


def _fake_rates(p3v8, p38, p3v9, g09):
    return {("P3V", 8): _rates(p3v8), ("P3", 8): _rates(p38),
            ("P3V", 9): _rates(p3v9), ("G0", 9): _rates(g09)}


def test_primaries_use_the_registered_contrasts_and_alpha():
    """Catches Claim 1 compared against G0 instead of P3, Claim 2 against P3 instead of G0,
    or the 0.05 alpha. Ten of twelve positive is p 0.0193: CONFIRMED at 0.025."""
    ten = [0.10] * 10 + [0.06] * 2
    rates = _fake_rates(p3v8=ten, p38=[0.08] * 12, p3v9=[0.05] * 12, g09=[0.01] * 12)
    rates[("G0", 8)] = _rates([0.0123] * 12)
    rates[("P3", 9)] = _rates([0.0456] * 12)
    out = agg.primary_verdicts(rates, gate_v={8: True, 9: True}, gate0_ok=True)
    assert out["claim1"][0] == "CONFIRMED"
    assert out["claim2"][0] == "CONFIRMED"
    # Which arms each claim compared, by their distinctive means: the comparator of Claim 1 is
    # depth-8 P3 (0.08), of Claim 2 depth-9 G0 (0.01). A swapped arm or depth changes these.
    assert out["claim1"][3] == pytest.approx(sum(ten) / 12)
    assert out["claim1"][4] == pytest.approx(0.08)
    assert out["claim2"][3] == pytest.approx(0.05)
    assert out["claim2"][4] == pytest.approx(0.01)


def test_claim2_is_unresolved_when_both_sit_on_the_floor():
    """Catches a depth-9 'win' between two arms both under 0.02 (the EXP-064 trap)."""
    rates = _fake_rates(p3v8=[0.1] * 12, p38=[0.08] * 12, p3v9=[0.019] * 12, g09=[0.017] * 12)
    out = agg.primary_verdicts(rates, gate_v={8: True, 9: True}, gate0_ok=True)
    assert out["claim2"][0] == "UNRESOLVED"


def test_a_failed_gate_v_voids_only_its_own_depth():
    """Catches Gate V at depth 9 voiding depth 8's claim, or being ignored."""
    rates = _fake_rates(p3v8=[0.1] * 12, p38=[0.08] * 12, p3v9=[0.05] * 12, g09=[0.01] * 12)
    out = agg.primary_verdicts(rates, gate_v={8: True, 9: False}, gate0_ok=True)
    assert out["claim1"][0] == "CONFIRMED" and out["claim2"][0] == "VOID"


def test_a_failed_gate_0_voids_everything():
    rates = _fake_rates(p3v8=[0.1] * 12, p38=[0.08] * 12, p3v9=[0.05] * 12, g09=[0.01] * 12)
    out = agg.primary_verdicts(rates, gate_v={8: True, 9: True}, gate0_ok=False)
    assert out["claim1"][0] == "VOID" and out["claim2"][0] == "VOID"
