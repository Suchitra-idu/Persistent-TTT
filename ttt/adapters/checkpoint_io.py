"""Checkpoint serialization, over the Storage port.

Keys are stripped to base-model-relative so a checkpoint loads with or without
a PEFT wrap, and shapes are checked before any copy so a config change reports
"incompatible" rather than failing halfway through with a broadcast error.
"""

from __future__ import annotations

import io
from typing import Iterable

import torch

from ttt.core import naming

TTT_PARAMS = "ttt_params.pt"


def save_ttt_params(
    model: torch.nn.Module, storage, path: str, *, layer_indices: Iterable[int]
) -> int:
    ttt_down = naming.ttt_down_suffixes(layer_indices)
    tensors = {
        naming.strip_peft_prefix(name): parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if naming.is_checkpoint_key(name, ttt_down)
    }
    storage.write_bytes(path, _dump(tensors))
    return len(tensors)


def load_ttt_params(model: torch.nn.Module, storage, path: str) -> int:
    saved = _load(storage.read_bytes(path))
    targets = {
        naming.strip_peft_prefix(name): parameter
        for name, parameter in model.named_parameters()
    }
    missing = sorted(set(saved) - set(targets))
    if missing:
        raise RuntimeError(
            f"checkpoint has {len(missing)} tensor(s) the model does not, first "
            f"{missing[0]!r}; layer indices likely disagree with the checkpoint"
        )
    for key, tensor in saved.items():
        target = targets[key]
        if tuple(tensor.shape) != tuple(target.shape):
            raise RuntimeError(
                f"checkpoint tensor {key!r} has shape {tuple(tensor.shape)}, "
                f"model parameter has {tuple(target.shape)}; rebuild the checkpoint"
            )
    for key, tensor in saved.items():
        target = targets[key]
        target.data.copy_(tensor.to(target.device, target.dtype))
    return len(saved)


def keys_in(storage, path: str) -> tuple[str, ...]:
    return tuple(sorted(_load(storage.read_bytes(path))))


def _dump(payload) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _load(data: bytes):
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
