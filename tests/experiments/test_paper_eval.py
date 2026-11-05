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


def _holdout(plan, resolution=None):
    resolution = resolution or _builders.wider_holdout(_builders.resolved(), holdout_last_n=12)
    eng = _builders.engine(resolution)
    resolution = paper_eval_v1._with_docs(resolution, plan.n_docs)
    source = _builders.data_source(sources=_builders.SOURCES * 6)
    return paper_eval_v1._holdout_by_source(resolution, eng, source, plan, [].append), resolution, eng, source


def test_a_fresh_holdout_never_repeats_the_excluded_seeds_documents():
    plan = dataclasses.replace(PLAN, n_docs=1, exclude_holdout_seed=0)
    fresh, resolution, eng, source = _holdout(plan)
    old, *_ = _holdout(dataclasses.replace(PLAN, n_docs=1))

    old_ids = {d.token_ids for docs in old.values() for d in docs}
    assert fresh and old_ids
    assert all(d.token_ids not in old_ids for docs in fresh.values() for d in docs)


def test_without_an_exclusion_the_holdout_is_the_seeds_own_pick():
    first, *_ = _holdout(PLAN)
    again, *_ = _holdout(PLAN)

    assert {s: [d.token_ids for d in docs] for s, docs in first.items()} == {
        s: [d.token_ids for d in docs] for s, docs in again.items()
    }


def _case(index):
    from ttt.app.relatedness_eval import Case

    return Case(index=index, target_title=f"T{index}", target=(1, 2), primes={"self": (3,)},
                prime_titles={"self": "T"}, chain=((4, 5),))


def test_cases_are_built_once_and_reloaded_identically():
    from ttt.experiments._wiki_relatedness import Pools

    eng = _builders.engine()
    plan = dataclasses.replace(PLAN, cases_file="fresh")
    builds = []

    def build():
        builds.append(1)
        return [_case(0), _case(1)], [], Pools(generic=((7, 8),), excluded=3)

    first = paper_eval_v1._cached_cases(eng, plan, [].append, build)
    second = paper_eval_v1._cached_cases(eng, plan, [].append, build)

    assert len(builds) == 1
    assert first == second == ([_case(0), _case(1)], [(7, 8)])


def test_without_a_cases_file_the_cases_are_rebuilt_every_time():
    from ttt.experiments._wiki_relatedness import Pools

    eng = _builders.engine()
    builds = []

    def build():
        builds.append(1)
        return [_case(0)], [], Pools()

    paper_eval_v1._cached_cases(eng, PLAN, [].append, build)
    paper_eval_v1._cached_cases(eng, PLAN, [].append, build)

    assert len(builds) == 2


def _shingle_run(monkeypatch, plan, **overrides):
    from ttt.app import relatedness_eval

    decoded = []
    monkeypatch.setattr(paper_eval_v1, "WIKIPEDIA", "FixtureProse")
    monkeypatch.setattr(relatedness_eval, "shingles", lambda text: decoded.append(text) or {text})
    resolution = _builders.resolved(**overrides)
    seen = paper_eval_v1._training_shingles(
        resolution, _builders.engine(resolution), _builders.data_source(sources=_builders.SOURCES * 6),
        plan, [].append,
    )
    return seen, decoded


def test_training_overlap_reads_the_wikipedia_training_documents_of_each_seed(monkeypatch):
    seen, decoded = _shingle_run(monkeypatch, dataclasses.replace(PLAN, overlap_seeds="42,43"), limit_docs=6)

    assert decoded and seen == set(decoded)


def test_no_overlap_seeds_means_no_training_text_is_read(monkeypatch):
    seen, decoded = _shingle_run(monkeypatch, PLAN, limit_docs=6)

    assert (seen, decoded) == (set(), [])


def test_an_overlap_check_without_the_training_document_limit_is_an_error(monkeypatch):
    with pytest.raises(ValueError, match="limit_docs"):
        _shingle_run(monkeypatch, dataclasses.replace(PLAN, overlap_seeds="42"))


def test_token_gain_is_an_opt_in_suite():
    assert paper_eval_v1.Plan(suites=(paper_eval_v1.TOKEN_GAIN,)).suites == ("token_gain",)


def test_token_gain_rows_survive_the_analysis():
    row = {"arm": "reset", "doc_idx": 0, "target_title": "T", "losses": [1.0, 2.0]}

    assert paper_eval_v1.analyse({"token_gain": [row]}).relatedness == ()
