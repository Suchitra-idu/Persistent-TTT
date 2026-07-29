"""Runs the gpu tier on a real H100: `make test-gpu-modal`.

Not a test module — pytest never collects this, Modal invokes it. It lives here
so the runner and what it runs cannot drift apart.

The HF cache rides on a volume, so the base model downloads once on Modal's link
rather than every container start.
"""

from __future__ import annotations

import modal

from ttt.adapters import modal_runtime

TIMEOUT_S = 30 * 60
TEST_REQUIREMENTS = ("pytest>=8", "hypothesis>=6")

app = modal.App("ttt-gpu-smoke")

image = (
    modal_runtime.build_image(extra_packages=TEST_REQUIREMENTS)
    .add_local_dir("tests", "/root/tests")
    .add_local_file("pyproject.toml", "/root/pyproject.toml")
)


@app.function(
    image=image,
    gpu=modal_runtime.GPU,
    volumes={modal_runtime.HF_CACHE_MOUNT: modal_runtime.cache_volume()},
    secrets=modal_runtime.secrets(),
    timeout=TIMEOUT_S,
)
def smoke() -> int:
    import subprocess
    import sys

    import torch

    print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/gpu",
         "-o", "addopts=", "-m", "gpu", "-v", "-s"],
        cwd="/root",
    )
    # Commit even on failure: a download that succeeded is worth keeping whatever
    # the assertions did.
    modal_runtime.cache_volume().commit()
    return completed.returncode


@app.local_entrypoint()
def main() -> None:
    code = smoke.remote()
    if code:
        raise SystemExit(code)
