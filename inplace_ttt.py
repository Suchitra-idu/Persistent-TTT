"""
In-Place Test-Time Training (In-Place TTT) for HuggingFace Qwen3 models.
Drop-in replacement for the gated MLP block on a subset of layers.
Model-size invariant; TTT layer schedule derived from num_hidden_layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ttt_config import TTTConfig


class EmbeddingTap:
    def __init__(self, conv_kernel_size: int):
        self.context_len = conv_kernel_size - 1
        self.current: Optional[torch.Tensor] = None
        self.prev_context: Optional[torch.Tensor] = None
        self._rolling: Optional[torch.Tensor] = None
        self.stateful = False

    def hook(self, _module, _inputs, output):
        self.current = output
        if self.stateful:
            self.prev_context = self._rolling
            joined = output if self._rolling is None else torch.cat(
                [self._rolling, output], dim=1
            )
            self._rolling = joined[:, -self.context_len:, :].detach()
        else:
            self.prev_context = None
        return None

    def reset_stream(self):
        self._rolling = None
        self.prev_context = None


@dataclass
class TTTState:
    delta: Optional[torch.Tensor] = None
    pending_z: list = field(default_factory=list)
    pending_v: list = field(default_factory=list)
    pending_tokens: int = 0


class InPlaceTTTMLP(nn.Module):
    def __init__(self, original_mlp: nn.Module, hidden_size: int,
                 cfg: TTTConfig, tap: EmbeddingTap):
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn

        self.cfg = cfg
        self.tap = tap

        # gamma frozen at 1.0 so the norm is a hard rescale, not a soft suggestion
        # the optimizer could re-inflate to defeat magnitude control.
        if cfg.v_source == "hidden_state":
            self.v_source_norm = nn.RMSNorm(hidden_size)
            self.v_source_norm.weight.requires_grad_(False)
        else:
            self.v_source_norm = None

        self.target_conv = nn.Conv1d(
            in_channels=hidden_size, out_channels=hidden_size,
            kernel_size=cfg.conv_kernel_size, groups=hidden_size, bias=False,
        )
        with torch.no_grad():
            self.target_conv.weight.zero_()
            self.target_conv.weight[:, :, -1] = 1.0

        # Zero-init W_target: at step 0 the TTT layer is bit-exact identity to base MLP.
        self.w_target = nn.Parameter(torch.zeros(hidden_size, hidden_size))

        if cfg.output_gate:
            self.output_gate = nn.Linear(hidden_size, 1, bias=True)
            with torch.no_grad():
                self.output_gate.bias.fill_(cfg.output_gate_bias_init)
                nn.init.normal_(self.output_gate.weight, std=1e-3)
        else:
            self.output_gate = None

        self._gate_l2: Optional[torch.Tensor] = None

        self.ttt_evolve = True
        self.stateful = False
        self.state = TTTState()

        self._hidden_context: Optional[torch.Tensor] = None

        # Session-persistent training (TBPTT). _next_carried staged in forward,
        # promoted by advance_session_state after backward; idempotent staging
        # survives gradient-checkpointing recompute.
        self.session_mode = False
        self.carried_delta: Optional[torch.Tensor] = None
        self._next_carried: Optional[torch.Tensor] = None

    def _targets(self, source: torch.Tensor,
                 left_context: Optional[torch.Tensor]) -> torch.Tensor:
        K = self.cfg.conv_kernel_size
        streaming = left_context is not None
        # Streaming forces causal padding (no future tokens across call boundaries).
        if self.cfg.v_bidirectional and not streaming:
            left_pad = K // 2
            right_pad = K - 1 - left_pad
        else:
            left_pad = K - 1
            right_pad = 0

        if streaming:
            x = torch.cat([left_context.to(source.dtype), source], dim=1)
            left_pad = max(0, left_pad - left_context.shape[1])
        else:
            x = source

        x = x.transpose(1, 2)
        if left_pad > 0 or right_pad > 0:
            x = F.pad(x, (left_pad, right_pad))
        v = self.target_conv(x).transpose(1, 2)
        if streaming:
            v = v[:, -source.shape[1]:, :]
        return v @ self.w_target

    def _v_source(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.cfg.v_source == "hidden_state":
            return self.v_source_norm(hidden_states)
        return self.tap.current

    def _v_left_context(self) -> Optional[torch.Tensor]:
        if self.cfg.v_source == "hidden_state":
            return self._hidden_context
        return self.tap.prev_context

    def _update_hidden_context(self, source: torch.Tensor):
        # Per-module buffer: hidden_states differ per layer, unlike the shared tap.
        if self.cfg.v_source != "hidden_state":
            return
        ctx_len = self.cfg.conv_kernel_size - 1
        if ctx_len <= 0:
            return
        joined = source if self._hidden_context is None else torch.cat(
            [self._hidden_context, source], dim=1
        )
        self._hidden_context = joined[:, -ctx_len:, :].detach()

    def reset_stream_state(self):
        self.state = TTTState()
        self._hidden_context = None

    def reset_v_context(self):
        # Soft turn boundary: clears left-context only, keeps state.delta.
        self._hidden_context = None

    def _clip(self, delta: torch.Tensor) -> torch.Tensor:
        if not self.cfg.clip_enabled:
            return delta
        if self.cfg.clip_at_inference_only and self.training:
            return delta
        norm = (self.cfg.eta * delta).norm(p="fro", dim=(-2, -1), keepdim=True)
        scale = (self.cfg.clip_tau / norm.clamp_min(1e-12)).clamp(max=1.0)
        return delta * scale

    def _gated(self, ttt_term: torch.Tensor,
               hidden_states: torch.Tensor) -> torch.Tensor:
        if self.output_gate is None:
            return ttt_term
        gate = torch.sigmoid(self.output_gate(hidden_states))
        if self.training and torch.is_grad_enabled():
            self._gate_l2 = (gate ** 2).mean()
        return gate * ttt_term

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # gate_proj / up_proj called AS MODULES so LoRA path is included in z.
        z = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)

        if not self.ttt_evolve and not self.stateful:
            # down_proj as module is fine here: TTT-layer down_proj never carries LoRA.
            return self.down_proj(z)

        if self.stateful:
            return self._stream_forward(z, hidden_states)
        return self._scan_forward(z, hidden_states)

    def _scan_forward(self, z: torch.Tensor,
                     hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, d_ff = z.shape
        C = self.cfg.chunk_size
        # down_proj.weight used FUNCTIONALLY here as the fast weight W0.
        w0 = self.down_proj.weight
        base_out = z @ w0.T

        carried = None
        if self.session_mode and self.carried_delta is not None:
            if self.carried_delta.shape[0] != B:
                raise RuntimeError(
                    "Batch size changed mid-session; keep B constant "
                    f"(state B={self.carried_delta.shape[0]}, input B={B})."
                )
            carried = self.carried_delta.to(z.dtype)

        if N <= C and carried is None and not self.session_mode:
            return base_out

        v = self._targets(self._v_source(hidden_states),
                          left_context=None)

        n_chunks = (N + C - 1) // C
        pad = n_chunks * C - N
        if pad:
            z = F.pad(z, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))

        zc = z.view(B, n_chunks, C, d_ff)
        vc = v.view(B, n_chunks, C, -1)

        deltas = torch.einsum("bkcd,bkcf->bkdf", vc, zc)
        if self.cfg.normalize_delta_by_chunk:
            # Divide by ACTUAL non-padded token count; last chunk may be short.
            chunk_sizes = [C] * n_chunks
            if pad:
                chunk_sizes[-1] = C - pad
            chunk_sizes = torch.tensor(
                chunk_sizes, dtype=deltas.dtype, device=deltas.device,
            ).clamp_min(1).view(1, n_chunks, 1, 1)
            deltas = deltas / chunk_sizes
        # Exclusive cumsum => strict chunk causality.
        cum = deltas.cumsum(dim=1)
        cum = torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], dim=1)
        if carried is not None:
            cum = cum + carried.unsqueeze(1)
        cum = self._clip(cum)

        ttt_out = self.cfg.eta * torch.einsum("bkcf,bkdf->bkcd", zc, cum)
        ttt_out = ttt_out.reshape(B, n_chunks * C, -1)[:, :N, :]

        if self.session_mode:
            # fp32 accum: bf16 drifts across many papers. detach = TBPTT boundary.
            total = deltas.sum(dim=1).detach().float()
            self._next_carried = (
                total if self.carried_delta is None
                else self.carried_delta + total
            )

        return base_out + self._gated(ttt_out, hidden_states)

    @torch.no_grad()
    def _stream_forward(self, z: torch.Tensor,
                       hidden_states: torch.Tensor) -> torch.Tensor:
        source = self._v_source(hidden_states)
        v = self._targets(source, left_context=self._v_left_context())
        # Buffer POST-norm source so stream and scan match exactly.
        self._update_hidden_context(source)

        w0 = self.down_proj.weight
        st = self.state
        C = self.cfg.chunk_size
        outputs = []
        pos = 0
        N = z.shape[1]

        while pos < N:
            room = C - st.pending_tokens
            take = min(room, N - pos)
            z_part = z[:, pos:pos + take]
            # Apply-then-update: apply with current state, then buffer.
            out = z_part @ w0.T
            if st.delta is not None:
                ttt_term = self.cfg.eta * (
                    z_part @ st.delta.to(z_part.dtype).transpose(-1, -2)
                )
                out = out + self._gated(ttt_term, hidden_states[:, pos:pos + take])
            outputs.append(out)

            if self.ttt_evolve:
                st.pending_z.append(z_part)
                st.pending_v.append(v[:, pos:pos + take])
                st.pending_tokens += take
                if st.pending_tokens == C:
                    self._commit_chunk()
            pos += take

        return torch.cat(outputs, dim=1)

    def _commit_chunk(self):
        st = self.state
        zc = torch.cat(st.pending_z, dim=1)
        vc = torch.cat(st.pending_v, dim=1)
        # fp32 accum: bf16 drifts across thousands of commits.
        delta = torch.einsum("bcd,bcf->bdf", vc.float(), zc.float())
        if self.cfg.normalize_delta_by_chunk:
            delta = delta / self.cfg.chunk_size
        new = delta if st.delta is None else st.delta + delta
        st.delta = self._clip(new)
        st.pending_z.clear()
        st.pending_v.clear()
        st.pending_tokens = 0


def patch_model_with_ttt(model, cfg: TTTConfig):
    """Replace MLP on cfg.layer_indices with InPlaceTTTMLP. Call BEFORE get_peft_model."""
    tap = EmbeddingTap(cfg.conv_kernel_size)
    if cfg.v_source == "embedding":
        model.get_input_embeddings().register_forward_hook(tap.hook)

    hidden = model.config.hidden_size
    dtype = next(model.parameters()).dtype
    for idx in cfg.layer_indices:
        layer = model.model.layers[idx]
        device = layer.mlp.down_proj.weight.device
        layer.mlp = InPlaceTTTMLP(layer.mlp, hidden, cfg, tap).to(device=device, dtype=dtype)
    model._ttt_tap = tap


def iter_ttt_modules(model):
    for m in model.modules():
        if isinstance(m, InPlaceTTTMLP):
            yield m


def reset_fast_weights(model):
    model._ttt_tap.reset_stream()
    for m in iter_ttt_modules(model):
        m.reset_stream_state()


def reset_v_left_context(model):
    """Soft turn boundary: clears conv left-context, keeps state.delta + pending."""
    model._ttt_tap.reset_stream()
    for m in iter_ttt_modules(model):
        m.reset_v_context()


# Session-persistent training lifecycle (TBPTT carry across papers):
#     reset_session_state(model)
#     for paper in session:
#         loss = model(paper); loss.backward()
#         advance_session_state(model)
def reset_session_state(model):
    for m in iter_ttt_modules(model):
        m.carried_delta = None
        m._next_carried = None


def advance_session_state(model):
    """Call exactly once per paper, AFTER backward."""
    for m in iter_ttt_modules(model):
        if m._next_carried is not None:
            m.carried_delta = m._next_carried
            m._next_carried = None


def state_norms(model, source: str = "session") -> dict:
    """||eta * delta||_F / ||W0||_F keyed by base-model layer index.
    source="session" reads carried_delta (TBPTT), "stream" reads state.delta."""
    modules = list(iter_ttt_modules(model))
    layer_indices = modules[0].cfg.layer_indices if modules else ()
    out = {}
    for layer_idx, m in zip(layer_indices, modules):
        delta = m.carried_delta if source == "session" else m.state.delta
        if delta is None:
            out[layer_idx] = 0.0
            continue
        num = (m.cfg.eta * delta).norm()
        den = m.down_proj.weight.detach().float().norm()
        out[layer_idx] = float(num / den)
    return out


def mean_state_ratio(norms: dict) -> float:
    return sum(norms.values()) / len(norms) if norms else 0.0


def gate_reg_term(model) -> torch.Tensor | float:
    """Returns 0.0 (float) when no gate value stashed, so callers can add unconditionally."""
    accum = None
    for m in iter_ttt_modules(model):
        if m._gate_l2 is not None:
            accum = m._gate_l2 if accum is None else accum + m._gate_l2
    return accum if accum is not None else 0.0


def stream_pending_progress(model) -> tuple[int, int]:
    modules = list(iter_ttt_modules(model))
    if not modules:
        return 0, 0
    m = modules[0]
    return int(m.state.pending_tokens), int(m.cfg.chunk_size)


def export_fast_weights(model) -> dict:
    return {
        i: m.state.delta.clone().cpu()
        for i, m in enumerate(iter_ttt_modules(model))
        if m.state.delta is not None
    }


def import_fast_weights(model, snapshot: dict):
    for i, m in enumerate(iter_ttt_modules(model)):
        if i in snapshot:
            # Keep fp32 on device; cast to activation dtype at apply time.
            m.state.delta = snapshot[i].to(
                device=m.down_proj.weight.device, dtype=torch.float32
            )
