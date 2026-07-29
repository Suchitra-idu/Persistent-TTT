"""Checkpoints written by the old tree load into the new one, and back (Phase 6.3).

This is what decides whether the runs already on the volume survive cutover. The
old side writes to a real path with `torch.save`; the new side reads through the
Storage port, so the bytes cross between them rather than the objects.
"""

from __future__ import annotations

import pytest
import torch
import ttt_wiring

from tests.parity import _builders as build
from tests.ports import _builders as ports
from ttt.adapters import checkpoint_io
from ttt.adapters.in_memory_storage import InMemoryStorage
from ttt.core.types import Carry

pytestmark = pytest.mark.parity

TTT_PARAMS = checkpoint_io.TTT_PARAMS
CARRIES = checkpoint_io.CARRIES


def old_saved(tmp_path):
    """A checkpoint written by the pre-rebuild code, as bytes in new storage."""
    model, cfg = build.old_model()
    path = tmp_path / TTT_PARAMS
    ttt_wiring.save_ttt_state_dict(model, str(path), cfg)
    storage = InMemoryStorage()
    storage.write_bytes(TTT_PARAMS, path.read_bytes())
    return storage, model, cfg


class TestTttParams:
    def test_the_key_sets_are_identical(self, tmp_path):
        storage, _, _ = old_saved(tmp_path)
        new_model, new_cfg = ports.tiny_model()
        checkpoint_io.save_ttt_params(
            new_model, storage, "new.pt", layer_indices=new_cfg.layer_indices
        )

        assert checkpoint_io.keys_in(storage, TTT_PARAMS) == checkpoint_io.keys_in(
            storage, "new.pt"
        )

    def test_an_old_checkpoint_loads_into_the_new_model(self, tmp_path):
        storage, _, _ = old_saved(tmp_path)
        new_model, _ = ports.tiny_model()

        loaded = checkpoint_io.load_ttt_params(new_model, storage, TTT_PARAMS)

        assert loaded == len(checkpoint_io.keys_in(storage, TTT_PARAMS))

    def test_the_loaded_values_are_the_ones_that_were_saved(self, tmp_path):
        storage, old_model, _ = old_saved(tmp_path)
        new_model, _ = ports.tiny_model()

        checkpoint_io.load_ttt_params(new_model, storage, TTT_PARAMS)

        assert torch.equal(
            new_model.model.layers[1].mlp.w_target,
            old_model.model.layers[1].mlp.w_target,
        )

    def test_every_ttt_tensor_crosses_over(self, tmp_path):
        storage, _, _ = old_saved(tmp_path)

        keys = checkpoint_io.keys_in(storage, TTT_PARAMS)

        assert all(
            any(suffix in key for key in keys) for suffix in ports.TTT_TENSOR_SUFFIXES
        )

    def test_a_new_checkpoint_loads_into_the_old_model(self, tmp_path):
        new_model, new_cfg = ports.tiny_model()
        storage = InMemoryStorage()
        checkpoint_io.save_ttt_params(
            new_model, storage, TTT_PARAMS, layer_indices=new_cfg.layer_indices
        )
        path = tmp_path / TTT_PARAMS
        path.write_bytes(storage.read_bytes(TTT_PARAMS))
        old_model, _ = build.old_model()

        ttt_wiring.load_ttt_state_dict(old_model, str(path))

        assert torch.equal(
            old_model.model.layers[1].mlp.w_target,
            new_model.model.layers[1].mlp.w_target,
        )

    def test_a_checkpoint_from_other_layer_indices_is_rejected(self, tmp_path):
        storage, _, _ = old_saved(tmp_path)
        other, _ = ports.tiny_model(layer_indices=(0, 2))

        with pytest.raises(RuntimeError, match="layer indices likely disagree"):
            checkpoint_io.load_ttt_params(other, storage, TTT_PARAMS)


class TestPerSourceCarries:
    def payload(self, tmp_path):
        carries = {
            "RedPajamaArXiv": {1: torch.randn(1, 4, 6), 3: torch.randn(1, 4, 6)},
            "RedPajamaBook": {1: torch.randn(1, 4, 6), 3: torch.randn(1, 4, 6)},
        }
        path = tmp_path / CARRIES
        ttt_wiring.save_per_source_carries(
            carries, str(path), meta={"n_updates": {"RedPajamaArXiv": 42}}
        )
        storage = InMemoryStorage()
        storage.write_bytes(CARRIES, path.read_bytes())
        return storage, carries

    def test_the_sources_come_back(self, tmp_path):
        storage, carries = self.payload(tmp_path)

        loaded, _ = checkpoint_io.load_carries(storage, CARRIES)

        assert set(loaded) == set(carries)

    def test_the_layer_indices_come_back_as_base_model_indices(self, tmp_path):
        storage, _ = self.payload(tmp_path)

        loaded, _ = checkpoint_io.load_carries(storage, CARRIES)

        assert loaded["RedPajamaArXiv"].layer_indices == (1, 3)

    def test_the_tensors_come_back_unchanged(self, tmp_path):
        storage, carries = self.payload(tmp_path)

        loaded, _ = checkpoint_io.load_carries(storage, CARRIES)

        assert torch.equal(
            loaded["RedPajamaArXiv"].get(1), carries["RedPajamaArXiv"][1]
        )

    def test_the_update_counts_come_back(self, tmp_path):
        storage, _ = self.payload(tmp_path)

        _, meta = checkpoint_io.load_carries(storage, CARRIES)

        assert meta["n_updates"] == {"RedPajamaArXiv": 42}

    def test_a_new_carry_file_loads_into_the_old_reader(self, tmp_path):
        storage = InMemoryStorage()
        carry = Carry(deltas={1: torch.randn(1, 4, 6), 3: torch.randn(1, 4, 6)})
        checkpoint_io.save_carries(
            {"RedPajamaArXiv": carry}, storage, CARRIES, n_updates={"RedPajamaArXiv": 7}
        )
        path = tmp_path / CARRIES
        path.write_bytes(storage.read_bytes(CARRIES))

        loaded, meta = ttt_wiring.load_per_source_carries(str(path))

        assert set(loaded) == {"RedPajamaArXiv"} and meta["n_updates"] == (
            {"RedPajamaArXiv": 7}
        )

    def test_the_old_reader_gets_the_same_tensors(self, tmp_path):
        storage = InMemoryStorage()
        carry = Carry(deltas={1: torch.randn(1, 4, 6)})
        checkpoint_io.save_carries({"src": carry}, storage, CARRIES)
        path = tmp_path / CARRIES
        path.write_bytes(storage.read_bytes(CARRIES))

        loaded, _ = ttt_wiring.load_per_source_carries(str(path))

        assert torch.equal(loaded["src"][1], carry.get(1))

    def test_a_missing_file_is_empty_on_both_sides(self, tmp_path):
        assert ttt_wiring.load_per_source_carries(
            str(tmp_path / "absent.pt")
        ) == checkpoint_io.load_carries(InMemoryStorage(), "absent.pt")


def test_the_carry_key_scheme_is_the_base_model_layer_index(tmp_path):
    """D14 defect 2: the old tree also keyed `export_fast_weights` by enumeration
    order, and mixing the two loaded half the modules with another layer's delta.
    The new tree has one representation, so the mix is unrepresentable."""
    storage = InMemoryStorage()
    model, cfg = ports.tiny_model()
    from ttt.adapters.torch_fast_weights import TorchFastWeights

    fast_weights = TorchFastWeights(model, cfg)
    fast_weights.set_mode(evolve=True, stream=False, session=True)
    model(input_ids=torch.tensor([ports.token_ids(12)]))
    fast_weights.advance_carry()
    checkpoint_io.save_carries({"src": fast_weights.snapshot()}, storage, CARRIES)

    loaded, _ = checkpoint_io.load_carries(storage, CARRIES)

    assert loaded["src"].layer_indices == ports.LAYER_INDICES
