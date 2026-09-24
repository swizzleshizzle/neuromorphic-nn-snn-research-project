"""Tests for `scripts/seed_effect.py`.

Every expected value here is hand-computable, because the whole point of the script is to
replace a hand-wave ("that block looks low") with an exact number. A test that only
asserted "p is between 0 and 1" would pass against every defect this file exists to catch.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "seed_effect", Path(__file__).resolve().parents[2] / "scripts" / "seed_effect.py"
)
seed_effect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed_effect)


class TestPerSeedEffect:
    def test_removes_each_arms_own_level(self):
        """The arms here differ by exactly 0.5 in level and agree perfectly on seeds.

        Residualising must therefore return the seed pattern centred on zero. Skipping the
        residualisation returns the raw average, which is this plus 0.35, so the assertion
        below fails against that defect rather than merely looking different.
        """
        arms = {
            ("low", 5): {0: 0.1, 1: 0.2, 2: 0.3},
            ("high", 5): {0: 0.6, 1: 0.7, 2: 0.8},
        }
        effect = seed_effect.per_seed_effect(arms)
        assert effect == pytest.approx({0: -0.1, 1: 0.0, 2: 0.1})

    def test_a_seed_missing_from_one_arm_uses_only_the_arms_that_have_it(self):
        arms = {
            ("a", 5): {0: 0.1, 1: 0.3},          # mean 0.2 -> residuals -0.1, +0.1
            ("b", 5): {0: 0.5, 1: 0.7, 2: 0.9},  # mean 0.7 -> residuals -0.2, 0.0, +0.2
        }
        effect = seed_effect.per_seed_effect(arms)
        assert effect[0] == pytest.approx((-0.1 + -0.2) / 2)
        assert effect[2] == pytest.approx(0.2)


class TestExactBlockP:
    def test_hand_computed_p_on_a_four_seed_split(self):
        """Pool [0, 0, 1, 1] split 2 and 2.

        Of the six ways to choose the second block, exactly two reach |difference| >= 1.0:
        the observed split and its mirror. So p is 2/6, and the enumeration must cover all
        six. A test asserting only `p < 0.5` would pass against an off-by-one here.
        """
        values = {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0}
        observed, p, count = seed_effect.exact_block_p(values, [0, 1], [2, 3])
        assert observed == pytest.approx(1.0)
        assert count == 6
        assert p == pytest.approx(2 / 6)

    def test_is_two_sided_so_swapping_the_blocks_only_flips_the_sign(self):
        values = {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0}
        fwd = seed_effect.exact_block_p(values, [0, 1], [2, 3])
        rev = seed_effect.exact_block_p(values, [2, 3], [0, 1])
        assert rev[0] == pytest.approx(-fwd[0])
        assert rev[1] == pytest.approx(fwd[1])

    def test_exchangeable_data_cannot_produce_a_small_p(self):
        """Every seed identical, so every partition ties the observed difference of zero.

        p must be exactly 1.0. A one-sided comparison, or a `>` where the code needs `>=`,
        drops this below 1.0.
        """
        values = {s: 0.4 for s in range(6)}
        observed, p, _ = seed_effect.exact_block_p(values, [0, 1, 2], [3, 4, 5])
        assert observed == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_uses_only_seeds_present_in_the_values(self):
        values = {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0}
        assert seed_effect.exact_block_p(values, [0, 1, 99], [2, 3])[2] == 6

    def test_refuses_a_block_too_small_to_permute(self):
        with pytest.raises(ValueError):
            seed_effect.exact_block_p({0: 0.1, 1: 0.2, 2: 0.3}, [0], [1, 2])


class TestReproducibility:
    def test_perfectly_agreeing_arms_score_plus_one(self):
        base = {s: s / 10 for s in range(4)}
        arms = {("a", 5): dict(base), ("b", 5): {s: v + 0.5 for s, v in base.items()}}
        mean_r, n_pairs, n_pos = seed_effect.reproducibility(arms, min_common=4)
        assert mean_r == pytest.approx(1.0)
        assert (n_pairs, n_pos) == (1, 1)

    def test_disagreeing_arms_score_minus_one_and_count_as_negative(self):
        """A seed effect that is noise would average near zero here, not +1.

        This is the case that gives the reproducibility number its meaning, so it has to be
        able to come out negative.
        """
        arms = {
            ("a", 5): {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
            ("b", 5): {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1},
        }
        mean_r, n_pairs, n_pos = seed_effect.reproducibility(arms, min_common=4)
        assert mean_r == pytest.approx(-1.0)
        assert (n_pairs, n_pos) == (1, 0)

    def test_arm_pairs_with_too_few_shared_seeds_are_skipped(self):
        arms = {("a", 5): {0: 0.1, 1: 0.2}, ("b", 5): {0: 0.3, 1: 0.4}}
        assert seed_effect.reproducibility(arms, min_common=4)[1] == 0


class TestSpanningArms:
    def test_an_arm_covering_only_one_block_is_excluded(self):
        """This is the depth-7 confound in miniature.

        An arm present in one block only cannot be residualised against the other, so its
        level and the block are the same variable. Including it would manufacture a block
        difference out of an arm difference.
        """
        records = {
            ("both", 5): {s: 0.3 for s in list(range(12)) + list(range(14, 24))},
            ("one_block_only", 5): {s: 0.9 for s in range(12)},
            ("wrong_depth", 7): {s: 0.3 for s in list(range(12)) + list(range(14, 24))},
        }
        arms = seed_effect.spanning_arms(records, 5, list(range(12)), list(range(14, 24)))
        assert set(arms) == {("both", 5)}


class TestLoadRecords:
    def test_reads_both_object_and_list_records_and_skips_non_cells(self, tmp_path):
        out = tmp_path / "experiments" / "099_x" / "outputs"
        out.mkdir(parents=True)
        (out / "one.json").write_text(
            json.dumps({"tag": "t", "depth": 5, "seed": 3, "success_rate": 0.25})
        )
        (out / "many.json").write_text(
            json.dumps([{"tag": "t", "depth": 5, "seed": 4, "success_rate": 0.5}])
        )
        (out / "not_a_cell.json").write_text(json.dumps({"selected_lr": 0.0001}))
        (out / "broken.json").write_text("{not json")
        assert seed_effect.load_records(tmp_path) == {("t", 5): {3: 0.25, 4: 0.5}}
