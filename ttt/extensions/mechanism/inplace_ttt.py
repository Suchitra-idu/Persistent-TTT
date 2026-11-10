"""InPlaceTTTMLP — the gated-MLP replacement, over the core.ttt_math kernels.

The chunked scan carries state across items via TBPTT. Assembly onto a real
model is Ring 3's job.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from ttt.core import carry as carry_math
from ttt.core import ttt_math
from ttt.core.config.ttt import TTTConfig


class InPlaceTTTMLP(nn.Module):
    def __init__(self, original_mlp: nn.Module, hidden_size: int, cfg: TTTConfig):
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn
        self.cfg = cfg

        # Captured before a checkpoint overwrites down_proj, so the `fresh`
        # regime can measure the untouched base model. Never checkpointed.
        self.register_buffer(
            "pretrained_down",
            original_mlp.down_proj.weight.detach().clone(),
            persistent=False,
        )
        self.pretrained = False

        # gamma frozen: a hard rescale, not one the optimizer can re-inflate.
        self.v_source_norm = nn.RMSNorm(hidden_size)
        self.v_source_norm.weight.requires_grad_(False)

        self.target_conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=cfg.conv_kernel_size,
            groups=hidden_size,
            bias=False,
        )
        with torch.no_grad():
            self.target_conv.weight.zero_()
            self.target_conv.weight[:, :, -1] = 1.0

        # Zero-init: at step 0 the layer is bit-exact identity to the base MLP.
        self.w_target = nn.Parameter(torch.zeros(hidden_size, hidden_size))

        if cfg.output_gate:
            self.output_gate = nn.Linear(hidden_size, 1, bias=True)
            with torch.no_grad():
                # Zeroed rather than jittered: one output unit has no symmetry
                # to break, and construction stays free of the global RNG (D1).
                self.output_gate.weight.zero_()
                self.output_gate.bias.fill_(cfg.output_gate_bias_init)
        else:
            self.output_gate = None

        self.output_gate_carry = (
            nn.Linear(hidden_size, 1, bias=True) if cfg.carry_gate else None
        )
        if self.output_gate_carry is not None:
            with torch.no_grad():
                self.output_gate_carry.weight.zero_()
                self.output_gate_carry.bias.fill_(cfg.output_gate_bias_init)
        # Zero-init, so the carry path starts as an exact no-op and enters only
        # if training finds it useful: a clipped carry is several times W0's
        # largest singular value, and an open gate alone lets it wreck the output.
        self.output_gate_carry_scale = (
            nn.Parameter(torch.zeros(1)) if cfg.carry_gate else None
        )

        self.gate_mean: float | None = None
        self.gate_std: float | None = None
        self.carry_gate_mean: float | None = None

        self.evolve = True
        self.session_mode = False

        # Staged in forward, promoted by advance_carry after backward. Staging
        # is idempotent so gradient-checkpointing recompute is harmless.
        self.carried: torch.Tensor | None = None
        self._next_carried: torch.Tensor | None = None
        self.recomputing = False

    def reset_carry(self) -> None:
        self.carried = None
        self._next_carried = None

    def advance_carry(self) -> None:
        """Call once per work item, after backward."""
        if self._next_carried is not None:
            self.carried = self._next_carried
            self._next_carried = None

    def detach_carry(self) -> None:
        """Cut the carry out of the graph; required once the weights it was built with change."""
        if self.carried is not None:
            self.carried = self.carried.detach()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # gate_proj / up_proj called as modules so a LoRA path lands in z.
        z = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        if self.pretrained:
            return F.linear(z, self.pretrained_down, self.down_proj.bias)
        if not self.evolve:
            return self.down_proj(z)
        return self._scan_forward(z, hidden_states)

    def _carried_for(self, batch: int) -> torch.Tensor | None:
        if not self.session_mode or self.carried is None:
            return None
        if self.carried.shape[0] != batch:
            raise RuntimeError(
                f"batch size changed mid-session: carry has B="
                f"{self.carried.shape[0]}, input has B={batch}"
            )
        return self.carried

    def _targets(self, source: torch.Tensor) -> torch.Tensor:
        x = F.pad(source.transpose(1, 2), (self.cfg.conv_kernel_size - 1, 0))
        return self.target_conv(x).transpose(1, 2) @ self.w_target

    def _gated(self, ttt_term: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.output_gate is None:
            return ttt_term
        gate = torch.sigmoid(self.output_gate(hidden_states))
        # A gate pinned near 0 means the TTT term cannot be reaching the output.
        with torch.no_grad():
            self.gate_mean = float(gate.mean())
            self.gate_std = float(gate.std(correction=0))
        return gate * ttt_term

    def _scan_forward(
        self, z: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # down_proj.weight is used functionally here: it is the fast weight W0.
        base_out = z @ self.down_proj.weight.T
        carried = self._carried_for(z.shape[0])

        # One chunk with nothing carried in contributes nothing (both rules
        # freeze the first chunk at zero), but in a session its delta is still
        # what the next item starts from.
        if z.shape[1] <= self.cfg.chunk_size and carried is None and not self.session_mode:
            return base_out

        v = self._targets(self.v_source_norm(hidden_states))
        through_scan = None if self.cfg.carry_gate else carried
        tau = self.cfg.effective_clip_tau
        if self.cfg.update_rule == "delta_chunk":
            ttt_out, item_delta = ttt_math.chunked_delta_scan(
                z,
                v,
                chunk_size=self.cfg.chunk_size,
                eta=self.cfg.eta,
                normalize_delta_by_chunk=self.cfg.normalize_delta_by_chunk,
                carried=through_scan,
                clip_tau=tau,
                truncate_every=self.cfg.truncate_every,
            )
        else:
            ttt_out, item_delta = ttt_math.scan(
                z,
                v,
                chunk_size=self.cfg.chunk_size,
                eta=self.cfg.eta,
                normalize_delta_by_chunk=self.cfg.normalize_delta_by_chunk,
                carried=through_scan,
                clip_tau=tau,
            )

        if self.session_mode and not self.recomputing:
            # fp32: bf16 drifts over a long session. The detach is the TBPTT
            # boundary unless carry_backprop moves it to the optimizer step.
            backprop = self.cfg.carry_backprop and self.training
            delta = item_delta.float() if backprop else item_delta.detach().float()
            self._next_carried = carry_math.advance(
                self.carried, delta, decay=self.cfg.carried_decay
            )

        out = base_out + self._gated(ttt_out, hidden_states)
        if self.cfg.carry_gate and carried is not None:
            out = out + self._carried_term(z, carried, hidden_states)
        return out

    def _carried_term(
        self, z: torch.Tensor, carried: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        state = carried.to(z.dtype)
        tau = self.cfg.effective_clip_tau
        if tau is not None:
            state = ttt_math.frobenius_clip(state.unsqueeze(1), eta=self.cfg.eta, tau=tau).squeeze(1)
        term = self.cfg.eta * torch.einsum("bnf,bdf->bnd", z, state)
        gate = torch.sigmoid(self.output_gate_carry(hidden_states))
        with torch.no_grad():
            self.carry_gate_mean = float(gate.mean())
        return self.output_gate_carry_scale.to(term.dtype) * gate * term


def iter_ttt_modules(model: nn.Module):
    for module in model.modules():
        if isinstance(module, InPlaceTTTMLP):
            yield module


class RecomputeAsForward:
    """Recompute context for non-reentrant gradient checkpointing.

    Built at forward time, entered on recompute. With carry_backprop a layer can
    be recomputed during a later item's backward, after its carry has moved on;
    this restores the carry it actually read and stops the recompute restaging.
    """

    def __init__(self, modules) -> None:
        self._seen = [(m, m.carried) for m in modules]

    def __enter__(self) -> None:
        self._live = [(m, m.carried) for m, _ in self._seen]
        for module, carried in self._seen:
            module.carried = carried
            module.recomputing = True

    def __exit__(self, *exc) -> None:
        for module, carried in self._live:
            module.carried = carried
            module.recomputing = False


def checkpoint_context_fn(model: nn.Module):
    """`context_fn` for torch.utils.checkpoint over a model with TTT layers."""
    modules = list(iter_ttt_modules(model))
    return lambda: (contextlib.nullcontext(), RecomputeAsForward(modules))


@contextlib.contextmanager
def pretrained_mlps(model: nn.Module):
    """Every TTT layer runs as the base MLP: pretrained W_down, no fast weight."""
    modules = list(iter_ttt_modules(model))
    for module in modules:
        module.pretrained = True
    try:
        yield
    finally:
        for module in modules:
            module.pretrained = False
