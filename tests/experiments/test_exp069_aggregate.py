"""Tests for EXP-069's aggregator: the verdict BANDS, the gate, and the grid loader.

The decomposition itself is EXP-068's and is covered by test_exp068_aggregate.py. What is new
here is exactly what EXP-068 got wrong: an unresolved band that is a VERDICT, not a warning.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "exp069_aggregate",
    Path(__file__).resolve().parents[2] / "experiments" / "069_encoder_decomposition" / "aggregate.py",
)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)


class TestEncoderVerdict:
    @pytest.mark.parametrize("share,expected", [
        (0.00, "MINOR"), (0.0999, "MINOR"),
        (0.10, "UNRESOLVED"), (0.267, "UNRESOLVED"), (0.3499, "UNRESOLVED"),
        (0.35, "MAJOR"), (0.90, "MAJOR"),
    ])
    def test_bands_including_both_edges(self, share, expected):
        """0.267 is EXP-068's own number: a mid-band share must come back UNRESOLVED, not a
        verdict. Edge values pin which side of each boundary is inclusive."""
        assert agg.encoder_verdict(share) == expected


class TestInteractionVerdict:
    @pytest.mark.parametrize("share,expected", [
        (0.65, "REPLICATED"), (0.50, "REPLICATED"),
        (0.4999, "UNRESOLVED"), (0.30, "UNRESOLVED"),
        (0.2999, "NOT REPLICATED"),
    ])
    def test_bands(self, share, expected):
        assert agg.interaction_verdict(share) == expected


class TestGate:
    def _s(self, grand, sd):
        return {"grand": grand, "total_sd": sd}

    def test_a_dead_grid_is_void_even_with_variance_and_a_resolving_effect(self):
        """EXP-064's lesson as a gate: a grid of floor policies has nothing to decompose."""
        ok, working, sd_ok, resolves = agg.check_gate(self._s(0.02, 0.20), 0.001, 0.001)
        assert (ok, working, sd_ok, resolves) == (False, False, True, True)

    def test_no_variance_is_void(self):
        assert agg.check_gate(self._s(0.30, 0.01), 0.001, 0.001)[0] is False

    def test_all_interaction_is_void(self):
        assert agg.check_gate(self._s(0.30, 0.20), 0.40, 0.60)[0] is False

    def test_passes_when_all_three_hold(self):
        assert agg.check_gate(self._s(0.30, 0.20), 0.40, 0.001)[0] is True

    def test_competence_bar_sits_below_the_known_cell(self):
        """EXP-043 measured this cell at 0.2900 to 0.3412 with shipped encoders. The bar must be
        attainable, or the gate could never pass."""
        assert agg.MIN_GRAND_MEAN < 0.29


class TestLoadGrid:
    def test_rows_are_the_ENCODER_and_columns_the_REST(self, tmp_path):
        """Transposing here would swap the primary claim's subject silently."""
        for e, r, v in [(0, 0, 0.11), (2, 6, 0.22), (7, 1, 0.33)]:
            (tmp_path / f"exp069_en{e}rs{r}_regionalized_d5_s{r}_sig0.0.json").write_text(
                json.dumps({"success_rate": v})
            )
        g = agg.load_grid(tmp_path)
        assert g[2][6] == pytest.approx(0.22)
        assert g[7][1] == pytest.approx(0.33)
        assert g[6][2] is None


class TestReusesExp068:
    def test_the_decomposition_is_exp068s_not_a_copy(self):
        """One audited implementation read by both experiments."""
        assert agg.anova.__module__ == "exp068_aggregate"
