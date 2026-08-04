"""Tests for the record-comparison script.

The script's whole job is to distinguish states, so every test here checks that it reports a
DIFFERENT verdict for a different input. A test that only asserted "returns 0 on identical
records" would pass against a script that always returns 0, which is the failure mode the
2026-08-02 handoff logged five separate instances of.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "verify_instrument_neutrality.py"

NEW_FIELDS = "train_success_rate,n_train_eval,generalisation_gap"


def write(directory: Path, name: str, **over) -> None:
    record = {
        "arm": "regionalized", "depth": 3, "seed": 0, "sigma": 0.0,
        "tag": "exp035_curriculum_e10000", "success_rate": 0.397,
        "revisit_rate": 0.166, "n": 30,
        "config": {"seed": 0, "depth": 3, "episodes": 10000, "out_dir": "outputs"},
    }
    record.update(over)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(record))


def run(old: Path, new: Path, *extra) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(old), str(new), *extra],
        capture_output=True, text=True,
    )


def test_identical_records_pass(tmp_path):
    write(tmp_path / "a", "r.json")
    write(tmp_path / "b", "r.json")
    # --new-fields "" means "this change adds nothing", which is what makes the two
    # records comparable in full. The default is EXP-030's instrument pair.
    result = run(tmp_path / "a", tmp_path / "b", "--new-fields", "")
    assert result.returncode == 0
    assert "PASS" in result.stdout


def test_a_changed_field_fails_and_names_it(tmp_path):
    """Must FAIL, and must say WHICH field moved.

    A script that failed without naming the field would be nearly useless for debugging, so
    the field name is part of the contract, not incidental output.
    """
    write(tmp_path / "a", "r.json")
    write(tmp_path / "b", "r.json", success_rate=0.500)
    result = run(tmp_path / "a", tmp_path / "b", "--new-fields", "")
    assert result.returncode == 1
    assert "success_rate" in result.stdout


def test_disjoint_filenames_fail_rather_than_vacuously_pass(tmp_path):
    """The bug this script exists to avoid.

    Two sweeps under different tags produce disjoint filenames. Keyed by filename the
    intersection is empty, and an empty intersection must never be reported as success.
    """
    write(tmp_path / "a", "exp035_regionalized_d3_s0_sig0.0.json")
    write(tmp_path / "b", "exp036_regionalized_d3_s0_sig0.0.json", tag="exp036_d3_e10000")
    result = run(tmp_path / "a", tmp_path / "b", "--new-fields", "")
    assert result.returncode == 1
    assert "nothing was actually compared" in result.stdout
    assert "--key-by cell" in result.stdout          # points at the fix


def test_key_by_cell_matches_across_tags(tmp_path):
    """The EXP-036 case: same run, different tag, different filename, must compare and pass."""
    write(tmp_path / "a", "exp035_regionalized_d3_s0_sig0.0.json")
    write(
        tmp_path / "b", "exp036_regionalized_d3_s0_sig0.0.json",
        tag="exp036_d3_e10000", train_success_rate=0.42, n_train_eval=90,
        generalisation_gap=0.023,
    )
    result = run(tmp_path / "a", tmp_path / "b",
                 "--key-by", "cell", "--new-fields", NEW_FIELDS, "--exempt", "tag")
    assert result.returncode == 0
    assert "compared:  1" in result.stdout


def test_key_by_cell_still_catches_a_real_change(tmp_path):
    """Cross-tag matching must not become a way to pass by ignoring too much.

    Same coordinates, same exemptions, but success_rate moved. If this passed, --key-by cell
    would be laundering differences rather than matching records.
    """
    write(tmp_path / "a", "exp035_regionalized_d3_s0_sig0.0.json")
    write(
        tmp_path / "b", "exp036_regionalized_d3_s0_sig0.0.json",
        tag="exp036_d3_e10000", success_rate=0.500,
        train_success_rate=0.42, n_train_eval=90, generalisation_gap=0.08,
    )
    result = run(tmp_path / "a", tmp_path / "b",
                 "--key-by", "cell", "--new-fields", NEW_FIELDS, "--exempt", "tag")
    assert result.returncode == 1
    assert "success_rate" in result.stdout


def test_missing_added_fields_fail(tmp_path):
    """If the new run lacks the fields the change was supposed to add, it did not take."""
    write(tmp_path / "a", "r.json")
    write(tmp_path / "b", "r.json")
    result = run(tmp_path / "a", tmp_path / "b", "--new-fields", NEW_FIELDS)
    assert result.returncode == 1
    assert "lack the added fields" in result.stdout


def test_cell_key_collision_is_reported_not_shadowed(tmp_path):
    """Two records sharing a cell key means the key does not identify a run here.

    Silently keeping the last one would compare the wrong pair and could report PASS on a
    sweep whose cells were never actually checked.
    """
    write(tmp_path / "a", "e600.json", config={"episodes": 600, "out_dir": "o"})
    write(tmp_path / "a", "e3000.json", config={"episodes": 3000, "out_dir": "o"})
    write(tmp_path / "b", "e600.json", config={"episodes": 600, "out_dir": "o"})
    result = run(tmp_path / "a", tmp_path / "b", "--key-by", "cell", "--new-fields", "")
    assert result.returncode == 1
    assert "collision" in result.stdout.lower()


def test_out_dir_alone_never_counts_as_a_difference(tmp_path):
    """Provenance, not measurement. A re-run writes elsewhere by design."""
    write(tmp_path / "a", "r.json", config={"seed": 0, "out_dir": "outputs"})
    write(tmp_path / "b", "r.json", config={"seed": 0, "out_dir": "outputs_rerun"})
    assert run(tmp_path / "a", tmp_path / "b", "--new-fields", "").returncode == 0
