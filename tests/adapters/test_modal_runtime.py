"""The Modal runtime seam: what the image carries and what the env forwards.

No network — Modal builds these objects locally and only resolves them on a
`modal run`.
"""

from __future__ import annotations

import pytest

from ttt.adapters import modal_runtime


class TestForwardedEnvironment:
    def test_it_forwards_the_settings_the_container_must_agree_on(self):
        forwarded = modal_runtime.forwarded_env({"TTT_DATASET": "slimpajama-6b"})

        assert forwarded == {"TTT_DATASET": "slimpajama-6b"}

    def test_it_forwards_nothing_it_was_not_given(self):
        assert modal_runtime.forwarded_env({}) == {}

    def test_it_ignores_unrelated_variables(self):
        assert modal_runtime.forwarded_env({"PATH": "/usr/bin"}) == {}

    def test_every_forwarded_key_is_one_the_cli_reads(self):
        from ttt import cli

        config_keys = {cli.ENV_DATASET, cli.ENV_MODEL_SIZE, cli.ENV_BASE_MODEL}

        assert config_keys <= set(modal_runtime.FORWARDED_KEYS)


class TestImage:
    def test_it_builds_without_a_network(self):
        assert modal_runtime.build_image({}) is not None

    def test_extra_packages_survive_the_local_source_step(self):
        """Modal rejects a build step after `add_local_*`, so extras have to go in
        before it — a caller chaining `.pip_install` onto the result gets an error
        only at `modal run`."""
        assert modal_runtime.build_image({}, extra_packages=("pytest>=8",)) is not None

    def test_local_files_can_still_be_added_after_the_extras(self):
        image = modal_runtime.build_image({}, extra_packages=("pytest>=8",))

        assert image.add_local_file("pyproject.toml", "/root/pyproject.toml")

    @pytest.mark.parametrize("cut", ["bitsandbytes", "flash-attn", "flash_attn"])
    def test_the_dependencies_d11_removed_are_absent(self, cut):
        assert not any(cut in package for package in modal_runtime.REQUIREMENTS)

    @pytest.mark.parametrize("needed", ["torch", "transformers", "peft", "datasets"])
    def test_the_dependencies_the_adapters_import_are_present(self, needed):
        assert any(package.startswith(needed) for package in modal_runtime.REQUIREMENTS)

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


class TestStorage:
    def test_the_checkpoint_storage_is_rooted_at_the_mount(self):
        volume = object()

        storage = modal_runtime.checkpoint_storage(volume)

        assert str(storage.root) == modal_runtime.CKPT_MOUNT

    def test_it_wraps_the_volume_it_was_given(self):
        volume = object()

        assert modal_runtime.checkpoint_storage(volume).volume is volume
