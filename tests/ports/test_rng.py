from __future__ import annotations

import pytest

from ttt.adapters.numpy_rng import NumpyRng
from ttt.adapters.scripted_rng import ScriptedRng
from ttt.ports.rng import Rng, is_permutation

SIZE = 8


class RngConformance:
    @pytest.fixture
    def rng(self):
        raise NotImplementedError

    def test_it_satisfies_the_port(self, rng):
        assert isinstance(rng, Rng)

    def test_integers_land_in_the_half_open_range(self, rng):
        drawn = [rng.integers(3, 5) for _ in range(20)]

        assert set(drawn) <= {3, 4}

    def test_an_empty_integer_range_is_rejected(self, rng):
        with pytest.raises(ValueError):
            rng.integers(4, 4)

    def test_permutation_is_a_permutation(self, rng):
        assert is_permutation(rng.permutation(SIZE), SIZE)

    def test_permutation_of_zero_is_empty(self, rng):
        assert rng.permutation(0) == []

    def test_shuffle_keeps_every_element(self, rng):
        items = list(range(SIZE))

        rng.shuffle(items)

        assert sorted(items) == list(range(SIZE))

    def test_random_is_a_unit_interval_float(self, rng):
        drawn = [rng.random() for _ in range(20)]

        assert all(0.0 <= value < 1.0 for value in drawn)

    def test_the_torch_generator_is_reproducible_per_seed(self, rng):
        import torch

        first = torch.rand(4, generator=rng.torch_generator(11))
        second = torch.rand(4, generator=rng.torch_generator(11))

        assert torch.equal(first, second)


class TestNumpyRng(RngConformance):
    @pytest.fixture
    def rng(self):
        return NumpyRng(seed=0)

    def test_the_same_seed_gives_the_same_permutation(self):
        assert NumpyRng(seed=7).permutation(SIZE) == NumpyRng(seed=7).permutation(SIZE)

    def test_different_seeds_give_different_permutations(self):
        assert NumpyRng(seed=1).permutation(64) != NumpyRng(seed=2).permutation(64)


class TestScriptedRng(RngConformance):
    @pytest.fixture
    def rng(self):
        return ScriptedRng()

    def test_an_unscripted_draw_takes_the_neutral_end(self, rng):
        assert (rng.integers(3, 9), rng.random(), rng.permutation(3)) == (
            3,
            0.0,
            [0, 1, 2],
        )

    def test_a_short_script_cycles(self):
        rng = ScriptedRng(integers=(4, 5))

        assert [rng.integers(0, 10) for _ in range(5)] == [4, 5, 4, 5, 4]

    def test_a_draw_outside_the_requested_range_is_rejected(self):
        with pytest.raises(ValueError, match="does not match the schedule"):
            ScriptedRng(integers=(99,)).integers(0, 10)

    def test_a_scripted_non_permutation_is_rejected(self):
        with pytest.raises(ValueError, match="not a permutation"):
            ScriptedRng(permutations=([0, 0, 1],)).permutation(3)

    def test_shuffle_follows_the_scripted_permutation(self):
        items = ["a", "b", "c"]

        ScriptedRng(permutations=([2, 0, 1],)).shuffle(items)

        assert items == ["c", "a", "b"]
