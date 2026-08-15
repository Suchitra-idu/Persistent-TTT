"""filter_num_proc: the threshold below which multiprocessing isn't worth it."""

from __future__ import annotations

from ttt.adapters.hf_table import MAX_FILTER_PROCS, MIN_ROWS_TO_PARALLELIZE, filter_num_proc


def test_a_small_table_stays_single_process():
    assert filter_num_proc(MIN_ROWS_TO_PARALLELIZE - 1) is None


def test_a_large_table_parallelizes():
    assert filter_num_proc(MIN_ROWS_TO_PARALLELIZE) is not None


def test_the_proc_count_never_exceeds_the_cap():
    assert filter_num_proc(10_000_000) <= MAX_FILTER_PROCS
