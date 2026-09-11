"""EXP-059's decision rules, measured on cases whose answer is already known.

Written alongside the aggregator, while the run had produced ZERO records, so nothing here was
tuned to a result. Every assertion below is chosen to FAIL against a specific plausible wrong
implementation, named in each test. The four defects this repo has already hidden behind
can-never-fail assertions are the reason that constraint is spelled out rather than assumed.

The two failures being defended against are opposites and both are real:
  * EXP-058 died of a gate that could not PASS (`mean_n_stored > 10`, bounded by a 7.76-step
    mean episode).
  * The mirror is a gate that cannot FAIL, and there is a live path to it here, because
    `recall_content_cos` is `None` when a readout had no real recall to probe and `None`
    coerced to 0.0 sails under a "below 0.95" ceiling while measuring nothing.
Both directions are tested.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AGG = REPO / "experiments" / "059_memory_depth5" / "aggregate.py"

# The real calibration measurements from the spec, used as the fixture values so the tests
# exercise the regime the gate will actually run in rather than a convenient invention.
COS_REAL_RUN = 0.8128     # measured on a real depth-5 run
COS_RANDOM_LOADED = 0.9437
COS_EMPTY = 1.000000      # nothing ever stored: the gate MUST fail here
FRAC_REAL = 0.1652        # EXP-058's measured unshuffled_frac


def fresh():
    """A fresh module instance per test, so mutating `HERE` cannot leak between them."""
    spec = importlib.util.spec_from_file_location("exp059_agg", AGG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rec(tag, readout, seed, success, revisit, cos, frac):
    return {
        "tag": tag, "readout": readout, "seed": seed, "arm": "regionalized", "depth": 5,
        "success_rate": success, "revisit_rate": revisit,
        "recall_content_cos": cos, "unshuffled_frac": frac,
    }


def write_world(mod, tmp_path, *, m_cos=COS_REAL_RUN, s_frac=FRAC_REAL, a_cos=1.0,
                m_success=0.33, a_success=0.33, m_revisit=0.20, a_revisit=0.20, n=24):
    """Lay down one full three-arm world and point the module at it."""
    out = tmp_path / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    for seed in range(n):
        for key, (readout, tag) in mod.ARMS.items():
            if key == "M":
                success, revisit, cos, frac = m_success, m_revisit, m_cos, 0.0
            elif key == "A":
                success, revisit, cos, frac = a_success, a_revisit, a_cos, 0.0
            else:
                success, revisit, cos, frac = 0.30, 0.22, COS_RANDOM_LOADED, s_frac
            r = rec(tag, readout, seed, success, revisit, cos, frac)
            (out / f"{tag}_regionalized_d5_s{seed}_sig0.0.json").write_text(json.dumps(r))
    mod.HERE = tmp_path
    return out


# --------------------------------------------------------------------------- the gate


def test_gate_can_fail_on_an_empty_attractor(tmp_path):
    """Breaks an always-passing gate. An empty attractor measures 1.000000 by construction."""
    mod = fresh()
    write_world(mod, tmp_path, m_cos=COS_EMPTY)
    ok, lines = mod.check_gate(*(mod.load(tmp_path / "outputs", mod.ARMS[k][1]) for k in ("M", "S")))
    assert ok is False
    assert any("FAIL" in ln for ln in lines)


def test_gate_can_pass_on_a_real_run(tmp_path):
    """Breaks an unsatisfiable gate: the EXP-058 shape that voided a 20-hour experiment."""
    mod = fresh()
    write_world(mod, tmp_path, m_cos=COS_REAL_RUN, s_frac=FRAC_REAL)
    ok, _ = mod.check_gate(*(mod.load(tmp_path / "outputs", mod.ARMS[k][1]) for k in ("M", "S")))
    assert ok is True


def test_missing_recall_probe_is_a_failure_not_a_pass(tmp_path):
    """THE load-bearing test. Breaks `None`-coerced-to-0.0, which would pass a 0.95 ceiling
    while the gate measured nothing at all."""
    mod = fresh()
    write_world(mod, tmp_path, m_cos=None)
    ok, lines = mod.check_gate(*(mod.load(tmp_path / "outputs", mod.ARMS[k][1]) for k in ("M", "S")))
    assert ok is False, "a missing probe must never satisfy a ceiling gate"
    assert any("MISSING" in ln for ln in lines)


def test_gate_reading_reports_missing_seeds_rather_than_dropping_them(tmp_path):
    """Breaks a mean that silently skips `None`, which hides partial probe failure."""
    mod = fresh()
    out = write_world(mod, tmp_path, m_cos=COS_REAL_RUN)
    readout, tag = mod.ARMS["M"]
    r = json.loads((out / f"{tag}_regionalized_d5_s7_sig0.0.json").read_text())
    r["recall_content_cos"] = None
    (out / f"{tag}_regionalized_d5_s7_sig0.0.json").write_text(json.dumps(r))
    mean, missing = mod.gate_reading(mod.load(out, tag), "recall_content_cos")
    assert missing == [7]
    assert mean == pytest.approx(COS_REAL_RUN)   # the present values still average correctly


def test_gate_fails_on_a_PARTIAL_probe_failure(tmp_path):
    """The dangerous case, and the one mutation testing found missing. If a single seed's probe
    is absent, the mean of the seeds that DID report can sit comfortably under 0.95 and the gate
    would pass on 23 of 24 arms while claiming to have checked all of them. Breaks a `check_gate`
    that consults the mean but not the missing list."""
    mod = fresh()
    out = write_world(mod, tmp_path, m_cos=COS_REAL_RUN)
    readout, tag = mod.ARMS["M"]
    f = out / f"{tag}_regionalized_d5_s11_sig0.0.json"
    r = json.loads(f.read_text())
    r["recall_content_cos"] = None
    f.write_text(json.dumps(r))
    M, S = mod.load(out, tag), mod.load(out, mod.ARMS["S"][1])
    mean, missing = mod.gate_reading(M, "recall_content_cos")
    assert missing == [11]
    assert mean < mod.GATE_MAX_RECALL_COS, "the present seeds alone would pass; that is the trap"
    ok, lines = mod.check_gate(M, S)
    assert ok is False, "one missing probe must void the gate, not be averaged away"
    assert any("MISSING" in ln for ln in lines)


def test_unshuffled_frac_gate_discriminates(tmp_path):
    """Breaks a gate that only ever looks at the cosine and ignores arm S entirely."""
    mod = fresh()
    write_world(mod, tmp_path, s_frac=0.25)      # above the 0.20 ceiling
    ok, _ = mod.check_gate(*(mod.load(tmp_path / "outputs", mod.ARMS[k][1]) for k in ("M", "S")))
    assert ok is False


def test_void_refuses_to_print_any_claim(tmp_path, capsys):
    """Breaks 'print the claims with a warning attached'. A failed condition means the numbers
    are not readable, which is the whole point of pre-registering it as a condition."""
    mod = fresh()
    write_world(mod, tmp_path, m_cos=COS_EMPTY, m_success=0.45, a_success=0.20)
    mod.main()
    text = capsys.readouterr().out
    assert "EXPERIMENT VOID" in text
    assert "CLAIM 1, PRIMARY" not in text
    assert "paired delta" not in text, "a void run must not report a contrast"


def test_arm_a_cosine_near_one_is_checked_descriptively(tmp_path, capsys):
    """Arm A zeroes W_rec at the read site, so its cosine is ~1.0 by construction. A low value
    means the probe is not measuring the read site, which would undermine a PASSING gate.
    Breaks an aggregator that never looks."""
    mod = fresh()
    write_world(mod, tmp_path, a_cos=0.42)
    mod.main()
    text = capsys.readouterr().out
    assert "WARNING" in text and "not ~1.0" in text


# ------------------------------------------------------------------- the claim wordings


def _sig(delta, n=24):
    """Differences with a tiny spread, so the permutation p is as small as it can be."""
    return [delta + (0.0005 if i % 2 else -0.0005) for i in range(n)]


def test_claim1_negative_is_a_real_finding_not_a_confirmation():
    """Breaks `abs(delta) >= BAR`, which would report memory HURTING as CONFIRMED. The spec
    pre-registers this reading explicitly because EXP-058's ordering made it the likely one."""
    mod = fresh()
    status, words = mod.claim1_verdict(_sig(-0.08), p=0.001)
    assert status == "HURTS"
    assert "REAL FINDING" in words
    assert "CONFIRMED" not in words


def test_claim1_positive_confirms():
    mod = fresh()
    status, words = mod.claim1_verdict(_sig(+0.08), p=0.001)
    assert status == "HELPS"
    assert "CONFIRMED" in words


def test_claim1_significant_sub_bar_names_its_direction():
    """The likeliest outcome given EXP-058's ~-0.03 ordering, and the easiest to under-report.
    A significant -0.045 is neither a confirmation nor a null, and EXP-057's Claim 2 at -0.0446
    needed that said in prose because its wording did not say it. Breaks wording that reports
    only the magnitude."""
    mod = fresh()
    status, words = mod.claim1_verdict(_sig(-0.045), p=0.0002)
    assert status == "SUB_BAR"
    assert "AGAINST memory" in words
    assert "NOT a confirmation" in words and "NOT a null" in words
    up_status, up_words = mod.claim1_verdict(_sig(+0.045), p=0.0002)
    assert up_status == "SUB_BAR"
    assert "in memory's favour" in up_words


def test_claim1_null_is_a_bound_never_an_equivalence():
    """Breaks 'the arms are equivalent' wording, and EXP-050's Claim 4 error."""
    mod = fresh()
    status, words = mod.claim1_verdict([0.001, -0.002] * 12, p=0.9)
    assert status == "NULL"
    assert "BOUND" in words
    for banned in ("equivalent", "no difference", "identical"):
        assert banned not in words.lower()


def test_claim2_sign_is_not_collapsed():
    """Memory is supposed to REDUCE cycling, so -0.05 confirms and +0.05 does not. Breaks a
    flipped comparison, which is the single easiest bug to write here and would invert the
    mechanism finding."""
    mod = fresh()
    good, words_good = mod.claim2_verdict(_sig(-0.05), p=0.001)
    bad, words_bad = mod.claim2_verdict(_sig(+0.05), p=0.001)
    assert good == "CONFIRMED"
    assert bad == "OPPOSITE"
    assert "INCREASED cycling" in words_bad
    assert "CONFIRMED" not in words_bad


def test_joint_reading_names_the_interesting_case():
    """The spec singles out mechanism-confirms-performance-does-not as the informative outcome."""
    mod = fresh()
    text = mod.joint_reading("NULL", "CONFIRMED")
    assert "localises the failure to the READOUT" in text
    other = mod.joint_reading("HELPS", "NULL")
    assert "NOT the anti-cycling story" in other


# ------------------------------------------------------------------------ the statistics


def test_t_multiplier_matches_the_degrees_of_freedom():
    """Breaks the hardcoded T95_DF11 = 2.201 that EXP-055/056/057 all carry. At n=24 the
    multiplier is 2.069, and using the n=12 value would report every interval ~6% too wide."""
    mod = fresh()
    assert mod.t95_for(24) == pytest.approx(2.069)
    assert mod.t95_for(12) == pytest.approx(2.201)
    assert mod.t95_for(24) != mod.t95_for(12)


def test_exact_permutation_p_on_a_case_with_a_known_answer():
    """All differences equal and positive: only the all-plus and all-minus flips reach the
    observed magnitude, so p is exactly 2/2**n. Breaks an off-by-one or a one-sided test."""
    mod = fresh()
    p, method = mod.permutation_p([0.1] * 10)
    assert p == pytest.approx(2 / 2 ** 10)
    assert "exact" in method


def test_permutation_switches_method_on_n_and_says_which():
    """The spec requires the method be PRINTED, because an exact proportion and a Monte Carlo
    estimate are not interchangeable. Breaks a silent switch."""
    mod = fresh()
    _, exact = mod.permutation_p([0.01, -0.02] * 6)          # n=12
    _, sampled = mod.permutation_p([0.01, -0.02] * 12)       # n=24
    assert "exact" in exact and "2**12" in exact
    assert "sampled" in sampled and "200,000" in sampled


def test_sampled_permutation_never_reports_exactly_zero_and_is_reproducible():
    """A sampled p of 0.0000 claims an exactness the sampling cannot support. Breaks a plain
    hits/draws estimator, and breaks an unseeded rng."""
    mod = fresh()
    diffs = [0.5] * 24
    p1, _ = mod.permutation_p(diffs)
    p2, _ = mod.permutation_p(diffs)
    assert p1 > 0.0
    assert p1 == p2, "the p-value must be reproducible at a fixed seed"


def test_bonferroni_is_across_three_contrasts_not_four():
    """Claim 3 is a CONDITION with no p-value and belongs to no family. Breaks 0.05/4."""
    mod = fresh()
    assert mod.N_CONTRASTS == 3
    assert mod.BONFERRONI == pytest.approx(0.05 / 3)


def test_significance_note_distinguishes_alpha_from_bonferroni():
    """Breaks silently substituting one threshold for the other in either direction."""
    mod = fresh()
    assert "clears Bonferroni" in mod._sig_note(0.01)
    assert "NOT Bonferroni" in mod._sig_note(0.03)
    assert "above alpha" in mod._sig_note(0.20)


# ------------------------------------------------------------------------- arm integrity


def test_a_tag_carrying_the_wrong_readout_is_refused(tmp_path):
    """All three arms share arm='regionalized' and differ ONLY in `readout`, so a driver edit
    pointing two tags at one readout would contrast an arm against itself with nothing in the
    numbers looking wrong. Breaks an unchecked load."""
    mod = fresh()
    out = write_world(mod, tmp_path)
    readout, tag = mod.ARMS["M"]
    p = out / f"{tag}_regionalized_d5_s3_sig0.0.json"
    r = json.loads(p.read_text())
    r["readout"] = "concept"
    p.write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="refusing"):
        mod.load(out, tag, expected_readout=readout)


def test_pairing_intersects_seeds_and_reports_n(tmp_path):
    """A partial run must be scored on the seeds present, with n visible, not silently padded."""
    mod = fresh()
    out = write_world(mod, tmp_path, n=24)
    readout, tag = mod.ARMS["M"]
    (out / f"{tag}_regionalized_d5_s23_sig0.0.json").unlink()
    M = mod.load(out, tag)
    A = mod.load(out, mod.ARMS["A"][1])
    seeds, diffs = mod.paired(M, A, "success_rate")
    assert len(seeds) == 23 and 23 not in seeds
    assert len(diffs) == 23
