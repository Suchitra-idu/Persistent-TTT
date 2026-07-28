"""InPlaceTTTMLP — the gated-MLP replacement, over the core.ttt_math kernels.

Two paths over one state family each: `_scan_forward` trains (chunked scan,
carry across items via TBPTT), `_stream_forward` generates (apply-then-buffer,
one chunk at a time). Assembly onto a real model is Ring 3's job.

Reset ladder:
    reset_stream()      stream delta + pending + conv left-context
    reset_v_context()   conv left-context only, so the stream delta survives
    reset_carry()       the session carry, independent of both
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ttt.core import carry as carry_math
from ttt.core import ttt_math
from ttt.core.config.ttt import TTTConfig


@dataclass
class StreamState:
    """The generation-path fast weight and its partially filled chunk."""

    delta: torch.Tensor | None = None
    pending_z: list = field(default_factory=list)
    pending_v: list = field(default_factory=list)
    pending_tokens: int = 0


class InPlaceTTTMLP(nn.Module):
    def __init__(self, original_mlp: nn.Module, hidden_size: int, cfg: TTTConfig):
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn
        self.cfg = cfg

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
                # Zeroed rather than jittered: the gate has one output unit, so
                # there is no symmetry to break, and construction stays free of
                # the global RNG (D1).
                self.output_gate.weight.zero_()
                self.output_gate.bias.fill_(cfg.output_gate_bias_init)
        else:
            self.output_gate = None

        self.gate_mean: float | None = None
        self.gate_std: float | None = None

        self.evolve = True
        self.stateful = False
        self.session_mode = False

        self.stream = StreamState()
        self._hidden_context: torch.Tensor | None = None

        # Staged in forward, promoted by advance_carry after backward. Staging
        # is idempotent so gradient-checkpointing recompute is harmless.
        self.carried: torch.Tensor | None = None
        self._next_carried: torch.Tensor | None = None

    def reset_stream(self) -> None:
        self.stream = StreamState()
        self._hidden_context = None

    def reset_v_context(self) -> None:
        self._hidden_context = None

    def reset_carry(self) -> None:
        self.carried = None
        self._next_carried = None

    def advance_carry(self) -> None:
        """Call once per work item, after backward."""
        if self._next_carried is not None:
            self.carried = self._next_carried
            self._next_carried = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # gate_proj / up_proj called as modules so a LoRA path lands in z.
        z = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        if not self.evolve and not self.stateful:
            return self.down_proj(z)
        if self.stateful:
            return self._stream_forward(z, hidden_states)
        return self._scan_forward(z, hidden_states)

    def _clip_tau(self) -> float | None:
        if self.cfg.clip_at_inference_only and self.training:
            return None
        return self.cfg.effective_clip_tau

    def _carried_for(self, batch: int) -> torch.Tensor | None:
        if not self.session_mode or self.carried is None:
            return None
        if self.carried.shape[0] != batch:
            raise RuntimeError(
                f"batch size changed mid-session: carry has B="
                f"{self.carried.shape[0]}, input has B={batch}"
            )
        return self.carried

    def _targets(
        self, source: torch.Tensor, left_context: torch.Tensor | None
    ) -> torch.Tensor:
        kernel = self.cfg.conv_kernel_size
        streaming = left_context is not None
        # Streaming forces causal padding: there are no future tokens yet.
        if self.cfg.v_bidirectional and not streaming:
            left_pad = kernel // 2
            right_pad = kernel - 1 - left_pad
        else:
            left_pad, right_pad = kernel - 1, 0

        x = source
        if streaming:
            x = torch.cat([left_context.to(source.dtype), source], dim=1)
            left_pad = max(0, left_pad - left_context.shape[1])

        x = x.transpose(1, 2)
        if left_pad or right_pad:
            x = F.pad(x, (left_pad, right_pad))
        v = self.target_conv(x).transpose(1, 2)
        if streaming:
            v = v[:, -source.shape[1] :, :]
        return v @ self.w_target

    def _update_hidden_context(self, source: torch.Tensor) -> None:
        width = self.cfg.conv_kernel_size - 1
        if width <= 0:
            return
        joined = (
            source
            if self._hidden_context is None
            else torch.cat([self._hidden_context, source], dim=1)
        )
        self._hidden_context = joined[:, -width:, :].detach()

    def _gated(
        self, ttt_term: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.output_gate is None:
            return ttt_term
        gate = torch.sigmoid(self.output_gate(hidden_states))
        # Under no_grad so eval and chat record them too (D14 turn diagnostics):
        # a gate pinned near 0 means the TTT term cannot be reaching the output.
        with torch.no_grad():
            self.gate_mean = float(gate.mean())
            self.gate_std = float(gate.std())
        return gate * ttt_term

    def _scan_forward(
        self, z: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # down_proj.weight is used functionally here: it is the fast weight W0.
        base_out = z @ self.down_proj.weight.T
        carried = self._carried_for(z.shape[0])

        # One chunk with nothing carried in contributes nothing — but in a
        # session its delta is still what the next item starts from.
        if (
            z.shape[1] <= self.cfg.chunk_size
            and carried is None
            and not self.session_mode
        ):
            return base_out

        v = self._targets(self.v_source_norm(hidden_states), left_context=None)
        ttt_out, item_delta = ttt_math.scan(
            z,
            v,
            chunk_size=self.cfg.chunk_size,
            eta=self.cfg.eta,
            normalize_delta_by_chunk=self.cfg.normalize_delta_by_chunk,
            carried=carried,
            clip_tau=self._clip_tau(),
        )

        if self.session_mode:
            # fp32 and detached: bf16 drifts over a long session, and the detach
            # is the TBPTT boundary.
            self._next_carried = carry_math.advance(
                self.carried,
                item_delta.detach().float(),
                decay=self.cfg.carried_decay,
            )

        return base_out + self._gated(ttt_out, hidden_states)

    @torch.no_grad()
    def _stream_forward(
        self, z: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        source = self.v_source_norm(hidden_states)
        v = self._targets(source, left_context=self._hidden_context)
        # Buffer the post-norm source so stream and scan see the same v.
        self._update_hidden_context(source)

        w0 = self.down_proj.weight
        state = self.stream
        chunk = self.cfg.chunk_size
        n_tokens = z.shape[1]
        outputs = []
        pos = 0

        while pos < n_tokens:
            take = min(chunk - state.pending_tokens, n_tokens - pos)
            z_part = z[:, pos : pos + take]
            out = z_part @ w0.T
            if state.delta is not None:
                out = out + self._gated(
                    ttt_math.stream_apply(z_part, state.delta, eta=self.cfg.eta),
                    hidden_states[:, pos : pos + take],
                )
            outputs.append(out)

            if self.evolve:
                state.pending_z.append(z_part)
                state.pending_v.append(v[:, pos : pos + take])
                state.pending_tokens += take
                if state.pending_tokens == chunk:
                    self._commit_chunk()
            pos += take

        return torch.cat(outputs, dim=1)

    def _commit_chunk(self) -> None:
        state = self.stream
        delta = ttt_math.stream_chunk_delta(
            torch.cat(state.pending_v, dim=1).float(),
            torch.cat(state.pending_z, dim=1).float(),
            chunk_size=self.cfg.chunk_size,
            normalize=self.cfg.normalize_delta_by_chunk,
        )
        total = delta if state.delta is None else state.delta + delta
        tau = self._clip_tau()
        state.delta = (
            total
            if tau is None
            else ttt_math.frobenius_clip(total, eta=self.cfg.eta, tau=tau)
        )
        state.pending_z.clear()
        state.pending_v.clear()
        state.pending_tokens = 0


def iter_ttt_modules(model: nn.Module):
    for module in model.modules():
        if isinstance(module, InPlaceTTTMLP):
            yield module
