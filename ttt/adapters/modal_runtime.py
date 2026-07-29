"""The Modal runtime: image, volumes, secrets, and the decorators over them.

Two dependencies the old image carried are gone (D11): bitsandbytes, and a
flash-attn wheel pinned to torch2.8+cu12+cp311+cxx11abiTRUE — the most fragile
line in the build. Attention is `sdpa` everywhere.
"""

from __future__ import annotations

import os

import modal

from ttt.adapters.modal_storage import ModalStorage

CKPT_VOLUME = "ttt-checkpoints"
HF_CACHE_VOLUME = "hf-hub-cache"
CKPT_MOUNT = "/ckpt"
HF_CACHE_MOUNT = "/hf-cache"

GPU = "H100"
PYTHON_VERSION = "3.11"

REQUIREMENTS = (
    "torch==2.8.0",
    "transformers>=4.56",
    "peft>=0.17",
    "datasets>=4.0",
    "accelerate>=1.0",
    "wandb>=0.21",
)

# The env lives on the laptop `modal run` is invoked from, not on the Modal
# machine, so anything the container must agree with is captured at image
# definition time and baked in.
FORWARDED_KEYS = ("TTT_DATASET", "TTT_MODEL_SIZE", "TTT_BASE_MODEL", "HF_TOKEN")


def forwarded_env(environ=None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    return {key: source[key] for key in FORWARDED_KEYS if key in source}


def build_image(environ=None, *, extra_packages=()) -> modal.Image:
    """`extra_packages` exists because Modal rejects any build step after an
    `add_local_*`, so a caller cannot pip_install onto the returned image."""
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install(*REQUIREMENTS, *extra_packages)
        .env({"HF_HOME": HF_CACHE_MOUNT, **forwarded_env(environ)})
        .add_local_python_source("ttt")
    )


def checkpoint_volume() -> modal.Volume:
    return modal.Volume.from_name(CKPT_VOLUME, create_if_missing=True)


def cache_volume() -> modal.Volume:
    return modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


def secrets() -> list:
    return [modal.Secret.from_name("wandb"), modal.Secret.from_name("huggingface")]


def checkpoint_storage(volume=None) -> ModalStorage:
    """Built here so the mount path and the volume cannot drift apart."""
    return ModalStorage(volume or checkpoint_volume(), CKPT_MOUNT)
