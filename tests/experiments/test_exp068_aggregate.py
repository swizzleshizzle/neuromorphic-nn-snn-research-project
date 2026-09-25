"""Tests for EXP-068's aggregator.

The decomposition is the entire experiment: if `anova` mislabels which factor carries the
variance, the primary claim reads backwards and nothing downstream would notice. So these plant
KNOWN structure and assert the exact share, rather than asserting a share is between 0 and 1.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "exp068_aggregate",
    Path(__file__).resolve().parents[2] / "experiments" / "068_seed_decomposition" / "aggregate.py",
)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)


def grid_from(fn, n=10):
    return [[float(fn(i, j)) for j in range(n)] for i in range(n)]


class TestAnovaAttribution:
    def test_a_pure_ROW_effect_is_attributed_entirely_to_the_task_draw(self):
        """Value depends only on the split seed. share_a must be 1.0, not merely 'largest'.

        Transposing the grid is the mutation this catches: it would report share_b 1.0 and the
        primary claim would read exactly backwards.
        """
        stats = agg.anova(grid_from(lambda i, j: i))
        assert stats["share_a"] == pytest.approx(1.0)
        assert stats["share_b"] == pytest.approx(0.0)
        assert stats["share_i"] == pytest.approx(0.0)

    def test_a_pure_COLUMN_effect_is_attributed_entirely_to_the_trajectory(self):
        stats = agg.anova(grid_from(lambda i, j: j))
        assert stats["share_b"] == pytest.approx(1.0)
        assert stats["share_a"] == pytest.approx(0.0)

    def test_pure_interaction_has_no_main_effects(self):
        """Every row mean and every column mean is zero, so all variance is interaction.

        This is the third answer the spec names: Claim 1 confirmed with Claim 2 null.
        """
        stats = agg.anova(grid_from(lambda i, j: 1 if (i + j) % 2 == 0 else -1))
        assert stats["share_i"] == pytest.approx(1.0)
        assert stats["share_a"] == pytest.approx(0.0)
        assert stats["share_b"] == pytest.approx(0.0)

    def test_non_square_grid_scales_each_effect_by_the_RIGHT_dimension(self):
        """3 rows x 5 columns, value depends only on the row.

        A square grid cannot see a rows/cols swap in the sum-of-squares, and the real design is
        10x10, so this is the only test that pins the scaling. `ms_a` is 5 * sum((i-1)^2) / 2 = 5
        and sigma2_a is ms_a / cols = 1.0; scaling by `rows` instead gives 0.6.
        """
        stats = agg.anova([[float(i) for _ in range(5)] for i in range(3)])
        assert stats["ms_a"] == pytest.approx(5.0)
        assert stats["sigma2_a"] == pytest.approx(1.0)

    def test_the_estimator_SUBTRACTS_the_error_term(self):
        """x[i][j] = i + 5*(-1)^(i+j): a pure row effect plus a large zero-mean interaction.

        By hand: ms_a = 825/9 = 91.6667, ms_i = 2500/81 = 30.8642, so the unbiased component is
        (91.6667 - 30.8642)/10 = 6.0802. Forgetting the subtraction gives 9.1667, a 51% error.
        A pure-interaction grid cannot catch that mutation because its ms_a is exactly zero.
        """
        stats = agg.anova([[i + 5 * (-1) ** (i + j) for j in range(10)] for i in range(10)])
        assert stats["ms_a"] == pytest.approx(91.6667, rel=1e-5)
        assert stats["ms_i"] == pytest.approx(30.8642, rel=1e-5)
        assert stats["sigma2_a"] == pytest.approx(6.0802, rel=1e-4)

    def test_known_mixture_recovers_the_planted_ratio(self):
        """Row effect exactly twice the column effect in variance, no interaction.

        Values are i + 2j scaled so the row variance is a quarter of the column variance; the
        assertion is on the RATIO, which is what Claim 1 reads.
        """
        stats = agg.anova(grid_from(lambda i, j: i + 2 * j))
        assert stats["sigma2_b"] / stats["sigma2_a"] == pytest.approx(4.0, rel=1e-6)


class TestCollapsedGrid:
    def test_a_constant_grid_reaches_the_gate_instead_of_dividing_by_zero(self):
        """A dead policy is exactly what the gate exists to catch, so it must be REPORTABLE.

        Before the guard this raised ZeroDivisionError inside `anova`, so the gate could never
        run and the run would look like a crash rather than a VOID.
        """
        stats = agg.anova(grid_from(lambda i, j: 0.0))
        assert stats["total_sd"] == pytest.approx(0.0)
        assert stats["f_a"] == pytest.approx(1.0)
        assert agg.check_gate(stats, 1.0, 1.0)[0] is False

    def test_refuses_an_incomplete_grid(self):
        g = grid_from(lambda i, j: i)
        g[4][7] = None
        with pytest.raises(ValueError):
            agg.anova(g)


class TestGate:
    def _stats(self, sd):
        return {"total_sd": sd}

    def test_fails_when_the_grid_has_no_variance_to_decompose(self):
        ok, sd_ok, resolves = agg.check_gate(self._stats(0.01), 0.0001, 0.0001)
        assert (ok, sd_ok, resolves) == (False, False, True)

    def test_fails_when_NEITHER_main_effect_resolves(self):
        """The resolution half. A grid that is all interaction passes the sd bar and must
        still VOID, because Claim 1's share would be a ratio of two noise estimates."""
        ok, sd_ok, resolves = agg.check_gate(self._stats(0.20), 0.40, 0.60)
        assert (ok, sd_ok, resolves) == (False, True, False)

    def test_passes_only_when_both_conditions_hold(self):
        assert agg.check_gate(self._stats(0.20), 0.40, 0.0001)[0] is True

    def test_the_sd_bar_can_be_met_by_the_known_configuration(self):
        """Gate-calibration: EXP-036 measured this cell's per-seed sd at 0.1158, so the bar
        must sit below that or it could never pass."""
        assert agg.MIN_TOTAL_SD < agg.EXP036_D3_SD


class TestRandomizationP:
    def test_is_deterministic_across_calls(self):
        """A fixed shuffle seed is what makes a published p-value reproducible.

        The grid is pure noise at a fixed seed, so both p-values land MID-RANGE (about 0.26 and
        0.46). That matters: a saturated p of 1/(n+1) is identical whether or not the shuffle is
        seeded, so a strongly-structured grid cannot detect an unseeded rng at all.
        """
        rng = random.Random(7)
        g = [[rng.gauss(0, 0.1) for _ in range(10)] for _ in range(10)]
        first = agg.randomization_p(g, n_shuffles=500)
        assert 0.05 < first[0] < 0.95 and 0.05 < first[1] < 0.95, "grid must not saturate p"
        assert agg.randomization_p(g, n_shuffles=500) == first

    def test_a_strong_planted_column_effect_gets_a_small_p_and_a_weak_row_effect_does_not(self):
        g = grid_from(lambda i, j: j)
        p_a, p_b = agg.randomization_p(g, n_shuffles=500)
        assert p_b < 0.01
        assert p_a > 0.05

    def test_p_can_never_be_zero(self):
        """(hits + 1) / (n + 1). A reported p of exactly 0 is a claim the data cannot make."""
        g = grid_from(lambda i, j: j)
        assert min(agg.randomization_p(g, n_shuffles=100)) > 0.0


class TestLoadGrid:
    def test_parses_both_seeds_out_of_the_filename(self, tmp_path):
        """The filename is the ONLY place the two seeds survive: `record_filename` encodes
        neither, which is why the tag carries them. Reading them back wrong would transpose
        or scramble the grid silently."""
        for split, train, val in [(0, 0, 0.11), (3, 7, 0.22), (9, 4, 0.33)]:
            (tmp_path / f"exp068_sp{split}tr{train}_regionalized_d3_s{split}_sig0.0.json").write_text(
                json.dumps({"tag": f"exp068_sp{split}tr{train}", "success_rate": val})
            )
        grid = agg.load_grid(tmp_path)
        assert grid[0][0] == pytest.approx(0.11)
        assert grid[3][7] == pytest.approx(0.22)
        assert grid[9][4] == pytest.approx(0.33)
        assert grid[7][3] is None          # not transposed
