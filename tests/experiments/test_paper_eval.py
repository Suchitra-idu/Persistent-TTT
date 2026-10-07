"""Smoke for the evidence bundle: fake engine in, folder of files out."""

from __future__ import annotations

import dataclasses
import json

import pytest

from tests.experiments import _builders
from ttt.experiments import compare_checkpoints_v1, paper_eval_v1
from ttt.experiments.paper_eval_v1 import Plan

PLAN = Plan(n_docs=3, n_slices=2, n_chain=3, n_repeats=3, n_primes=1, n_targets=2)


def payload(plan=PLAN, resolution=None):
    resolution = resolution or _builders.wider_holdout(_builders.resolved(), holdout_last_n=12)
    return paper_eval_v1.run(
        _builders.wider_holdout(_builders.resolved(), holdout_last_n=12),
        engine=_builders.engine(resolution),
        source=_builders.data_source(sources=_builders.SOURCES * 6),
        plan=plan, announce=[].append,
    )


def bundle(tmp_path, name="a", plan=PLAN):
    folder = tmp_path / name
    paper_eval_v1.write_bundle(
        folder, payload(plan), {"bundle": f"{name}/step_1/x", "describe": {}},
        n_boot=20, announce=[].append,
    )
    return folder


def test_every_suite_writes_its_raw_rows():
    assert set(payload()["rows"]) == {"regimes", "regime_slices", "boundary", "repeat", "swap"}


def test_the_layers_come_from_the_built_engine_not_the_requested_config():
    resolution = _builders.wider_holdout(_builders.resolved(), holdout_last_n=12)
    built = dataclasses.replace(resolution, ttt=dataclasses.replace(resolution.ttt, layer_indices=(1, 3)))

    assert payload(resolution=built)["ttt_layers"] == [1, 3]


def test_a_checkpoint_without_ttt_layers_runs_only_the_regimes():
    resolution = _builders.wider_holdout(_builders.resolved(), holdout_last_n=12)
    no_ttt = dataclasses.replace(resolution, ttt=dataclasses.replace(resolution.ttt, layer_indices=()))

    assert payload(resolution=no_ttt)["suites"] == ["regimes"]


def test_a_plan_naming_an_unknown_suite_is_rejected():
    with pytest.raises(ValueError):
        Plan(suites=("regimes", "rular"))


@pytest.mark.slow
def test_the_bundle_has_raw_tables_figures_manifest_and_summary(tmp_path):
    folder = bundle(tmp_path)

    assert {p.name for p in folder.iterdir()} == {"raw", "tables", "figures", "manifest.json", "SUMMARY.md"}


@pytest.mark.slow
def test_the_bundle_draws_every_figure(tmp_path):
    folder = bundle(tmp_path)

    assert {p.name for p in (folder / "figures").iterdir()} == {
        "regime_gaps.png", "boundary_gaps.png", "repeat_curve.png",
        "stability.png", "swap_matrix.png", "domain_correlates.png",
    }


@pytest.mark.slow
def test_the_manifest_records_the_bootstrap(tmp_path):
    manifest = json.loads((bundle(tmp_path) / "manifest.json").read_text())

    assert (manifest["n_boot"], manifest["boot_seed"]) == (20, paper_eval_v1.BOOT_SEED)


@pytest.mark.slow
def test_rerendering_from_raw_reproduces_the_tables(tmp_path):
    folder = bundle(tmp_path)
    before = (folder / "tables/regime_gaps.csv").read_text()

    paper_eval_v1.rerender(str(folder), n_boot=20)

    assert (folder / "tables/regime_gaps.csv").read_text() == before


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [("run-a/step_600", ("run-a", "step_600")), ("step_600", ("default-run", "step_600"))],
)
def test_bundle_parts_split_a_checkpoint_into_run_and_step(checkpoint, expected):
    assert paper_eval_v1.bundle_parts(checkpoint, "default-run") == expected


@pytest.mark.slow
def test_comparing_a_bundle_with_itself_finds_no_gap(tmp_path):
    folder = bundle(tmp_path)

    out = compare_checkpoints_v1.compare(folder, folder, out_dir=tmp_path / "cmp", n_boot=20, announce=[].append)
    pooled = next(
        line for line in (out / "tables/checkpoints.csv").read_text().splitlines()
        if line.startswith("fresh_sanity,ALL,")
    )

    assert pooled.split(",")[3] == "0"


def test_config_diff_lists_only_keys_that_differ():
    assert compare_checkpoints_v1.config_diff({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4}) == [
        ("b", 2, 3), ("c", None, 4),
    ]
