from __future__ import annotations

import dataclasses

from ttt.adapters.local_storage import LocalStorage
from ttt.app import evidence, relatedness_eval
from ttt.app.relatedness_eval import RelatedRow
from ttt.core.types import COLD_CARRY, COLD_CARRY_OFF, EvalRow
from ttt.experiments import mixed_probe_v1

ARMS = (relatedness_eval.RESET, relatedness_eval.RELATED, relatedness_eval.OTHER_SOURCE, mixed_probe_v1.CHAIN_ARM)


def write(folder, *, gains, slice_gain, n=12):
    related = [
        RelatedRow(arm=arm, source="Wikipedia", doc_idx=i, target_title="t", prime_title="p",
                   n_tokens=100, ppl=20.0 - gains.get(arm, 0.0))
        for i in range(n) for arm in ARMS
    ]
    regimes = [
        EvalRow(source=s, doc_idx=i, regime=regime, n_tokens=100,
                ppl=15.0 - (slice_gain if regime == COLD_CARRY else 0.0))
        for s in ("A", "B") for i in range(n) for regime in (COLD_CARRY_OFF, COLD_CARRY)
    ]
    storage = LocalStorage(folder)
    evidence.save_json(storage, "raw/relatedness.json", [dataclasses.asdict(r) for r in related])
    evidence.save_json(storage, "raw/regimes.json", [dataclasses.asdict(r) for r in regimes])
    return folder


def by_measure(estimates):
    return {e.measure: e.interval.point for e in estimates}


def test_each_contrast_is_the_candidates_gain_minus_the_baselines(tmp_path):
    base = write(tmp_path / "b", gains={"related": 0.5, "other_source": -0.4, "chain_10": -0.6}, slice_gain=0.3)
    cand = write(tmp_path / "c", gains={"related": 0.4, "other_source": 0.1, "chain_10": 0.0}, slice_gain=0.2)

    points = by_measure(mixed_probe_v1.contrasts(base, cand, n_boot=50))

    assert round(points["related_gain_change"], 6) == -0.1
    assert round(points["other_source_gain_change"], 6) == 0.5
    assert round(points["chain_10_gain_change"], 6) == 0.6
    assert round(points[mixed_probe_v1.BETWEEN_SLICES], 6) == -0.1


def test_removing_harm_needs_positive_other_source_and_chain_without_losing_related(tmp_path):
    base = write(tmp_path / "b", gains={"related": 0.5, "other_source": -0.4, "chain_10": -0.6}, slice_gain=0.3)
    cand = write(tmp_path / "c", gains={"related": 0.5, "other_source": 0.1, "chain_10": 0.0}, slice_gain=0.3)

    changes = mixed_probe_v1.contrasts(base, cand, n_boot=50)

    assert mixed_probe_v1.verdict(changes, mixed_probe_v1.own_gains(cand, n_boot=50)) == (
        "training across documents removes the harm"
    )


def test_losing_the_related_gain_means_the_harm_is_not_shown_removed(tmp_path):
    base = write(tmp_path / "b", gains={"related": 0.5, "other_source": -0.4, "chain_10": -0.6}, slice_gain=0.3)
    cand = write(tmp_path / "c", gains={"related": 0.2, "other_source": 0.1, "chain_10": 0.0}, slice_gain=0.3)

    changes = mixed_probe_v1.contrasts(base, cand, n_boot=50)

    assert mixed_probe_v1.verdict(changes, mixed_probe_v1.own_gains(cand, n_boot=50)) == (
        "the harm is not shown to be removed"
    )


def test_near_zero_related_and_slice_gains_mean_the_carry_is_ignored(tmp_path):
    base = write(tmp_path / "b", gains={"related": 0.5, "other_source": -0.4, "chain_10": -0.6}, slice_gain=0.3)
    cand = write(tmp_path / "c", gains={"related": 0.005, "other_source": 0.0, "chain_10": 0.0}, slice_gain=0.0)

    changes = mixed_probe_v1.contrasts(base, cand, n_boot=50)

    assert mixed_probe_v1.verdict(changes, mixed_probe_v1.own_gains(cand, n_boot=50)) == (
        "the probe learned to ignore the carry"
    )


def test_the_probe_writes_its_table_and_verdict_into_the_candidate_bundle(tmp_path):
    base = write(tmp_path / "b", gains={"related": 0.5}, slice_gain=0.3)
    cand = write(tmp_path / "c", gains={"related": 0.5}, slice_gain=0.3)

    tables = mixed_probe_v1.probe(base, cand, n_boot=20, announce=[].append)

    assert (tables / "mixed_probe.csv").exists()
    assert "single-seed probe" in (tables / "mixed_probe_verdict.md").read_text()
