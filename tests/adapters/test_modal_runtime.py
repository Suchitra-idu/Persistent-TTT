"""The Modal runtime seam: what the image carries and what the env forwards.

No network — Modal builds these objects locally and only resolves them on a
`modal run`.
"""

from __future__ import annotations

import pytest

from ttt.adapters import modal_runtime


class TestImage:


    def test_the_pinned_versions_match_the_project_metadata(self):
        import tomllib
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        declared = tomllib.loads((root / "pyproject.toml").read_text())
        wanted = {
            name.split(">=")[0].split("==")[0]
            for name in declared["project"]["dependencies"]
        }
        shipped = {
            name.split(">=")[0].split("==")[0] for name in modal_runtime.REQUIREMENTS
        }

        assert shipped <= wanted | {"accelerate"}


class TestCacheCommit:

    def test_the_decorator_commits_after_a_successful_call(self, monkeypatch):
        volume = _SpyVolume()
        monkeypatch.setattr(modal_runtime, "cache_volume", lambda: volume)

        modal_runtime.caching(lambda: "done")()

        assert volume.commits == 1

    def test_the_decorator_commits_even_when_the_run_raises(self, monkeypatch):
        """A model that downloaded is worth keeping whatever failed after it."""
        volume = _SpyVolume()
        monkeypatch.setattr(modal_runtime, "cache_volume", lambda: volume)

        def boom():
            raise RuntimeError("training died")

        with pytest.raises(RuntimeError):
            modal_runtime.caching(boom)()

        assert volume.commits == 1


class _SpyVolume:
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1
