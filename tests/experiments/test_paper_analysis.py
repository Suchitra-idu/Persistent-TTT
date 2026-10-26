"""Smoke for the cross-checkpoint analysis: fake bundles in, the paper folder out."""

from __future__ import annotations

import json

import pytest

from tests.experiments.test_paper_eval import payload
from ttt.experiments import paper_analysis_v1, paper_eval_v1

CLIP = {"clip_enabled": "True", "clip_tau": "20.0"}
TABLES = ("T1_main_gaps", "T2_per_source", "T3_persistence", "T4_by_position",
          "T5_state_and_gate", "T6_clip_saturation")


@pytest.fixture(scope="module")
def rows():
    return payload()


def write_bundle(results, rows, run, step, stamp="20260101-000000", describe=None):
    paper_eval_v1.write_bundle(
        results / run / f"step_{step}" / stamp, rows,
        {"bundle": f"{run}/step_{step}/{stamp}", "describe": describe or {}},
        n_boot=10, announce=[].append,
    )


@pytest.fixture
def results(tmp_path, rows):
    root = tmp_path / "results"
    for step in (1, 2):
        write_bundle(root, rows, "control", step)
        write_bundle(root, rows, "ttt", step, describe=CLIP)
    return root


def run_analysis(results, out, argv=()):
    norms = out.parent / "w0_norms.json"
    norms.write_text(json.dumps({"norms": {"1": 40.0, "3": 60.0}}))
    return paper_analysis_v1.main([
        "--control", "control", "--run", "rule=ttt", "--steps", "1,2",
        "--w0-norms", str(norms), "--results", str(results), "--out", str(out),
        "--n-boot", "10", *argv,
    ])


def test_newest_bundle_picks_the_latest_stamp(tmp_path, rows):
    write_bundle(tmp_path, rows, "run", 1, stamp="20260101-000000")
    write_bundle(tmp_path, rows, "run", 1, stamp="20260202-000000")

    assert paper_analysis_v1.newest_bundle(tmp_path, "run", 1).path.name == "20260202-000000"


def test_newest_bundle_skips_a_newer_bundle_without_the_needed_rows(tmp_path, rows):
    write_bundle(tmp_path, rows, "run", 1, stamp="20260101-000000")
    write_bundle(tmp_path, rows, "run", 1, stamp="20260202-000000")
    (tmp_path / "run" / "step_1" / "20260202-000000" / "raw" / "regimes.json").unlink()

    assert paper_analysis_v1.newest_bundle(tmp_path, "run", 1).path.name == "20260101-000000"


def test_a_missing_bundle_names_the_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="step_7"):
        paper_analysis_v1.newest_bundle(tmp_path, "run", 7)


def test_the_analysis_writes_every_table_and_a_summary(results, tmp_path):
    out = run_analysis(results, tmp_path / "paper")

    assert all((out / "tables" / f"{t}.{ext}").is_file() for t in TABLES for ext in ("md", "csv"))
    assert (out / "SUMMARY.md").is_file()


def test_the_analysis_draws_every_figure(results, tmp_path):
    pytest.importorskip("matplotlib")
    out = run_analysis(results, tmp_path / "paper")

    assert {p.name for p in (out / "figures").iterdir()} == {
        "F1_gains_vs_step.pdf", "F2_by_position.pdf", "F3_repeat_mechanism.pdf",
        "F4_per_source_rule.pdf",
    }


def test_the_manifest_records_every_bundle_read(results, tmp_path):
    out = run_analysis(results, tmp_path / "paper")

    manifest = json.loads((out / "manifest.json").read_text())
    assert {(b["run"], b["step"]) for b in manifest["bundles"]} == {
        (run, step) for run in ("control", "ttt") for step in (1, 2)
    }
    assert manifest["clip_threshold"] == pytest.approx(20.0 / 50.0)


def test_files_the_analysis_does_not_write_are_left_alone(results, tmp_path):
    out = tmp_path / "paper"
    out.mkdir()
    (out / "FINDINGS.md").write_text("hand-written")

    run_analysis(results, out)

    assert (out / "FINDINGS.md").read_text() == "hand-written"


def test_a_run_must_be_given_as_label_equals_run(results, tmp_path):
    with pytest.raises(SystemExit, match="label=run"):
        run_analysis(results, tmp_path / "paper", argv=("--run", "unlabelled"))
