"""
In-Place Test-Time Training (In-Place TTT) for HuggingFace Qwen3 models.

Implements the mechanism from "In-Place Test-Time Training"
(ByteDance Seed, arXiv 2604.06169) as a drop-in replacement for the
gated MLP block on a subset of layers. Model-size invariant: works on
any Qwen3 variant (0.6B, 1.7B, 8B, ...) -- the TTT layer schedule and
LoRA regex are derived from model.config.num_hidden_layers at load time.

Mechanism per TTT layer, for activations Z = silu(gate(H)) * up(H)
and LM-aligned targets V = Conv1D(source) @ W_target, chunked into
non-overlapping chunks of size C. `source` is either the raw embedding
X0 (the paper's choice, cfg.v_source="embedding") or the layer's own
input hidden_states (cfg.v_source="hidden_state"). The conv is causal
by default; setting cfg.v_bidirectional=True turns it into a centered
window (past + current + future tokens) on the scan path.

    apply:   O_[i] = Z_[i] @ (W_down + eta * S_i)^T
    update:  S_{i+1} = S_i + V_[i]^T @ Z_[i]        (S_0 = 0)

so chunk i is processed with updates accumulated strictly from chunks
before it. The cumulative-sum formulation below is mathematically
identical to the sequential apply-then-update loop.

Two execution modes (cohesion -- each mode is one method):
  * stateless scan  -- training and whole-sequence evaluation. Fast
    weights implicitly reset every forward call, which matches the
    paper's per-document W_gatereset when you feed one document per sequence.
    Session mode (see below) opts into cross-call carry; the training
    loop also randomly slices papers so one "call" may be a token-range
    sub-paper rather than a whole document, with carry threaded
    through. See train_utils.build_session_items.
  * stateful stream -- autoregressive inference. Fast weight deltas
    persist across forward calls in a TTTState object, enabling the
    cross-session persistence experiments. Evolution can be switched
    off at any time via set_ttt_evolve(model, False).

Critical wiring rules, do not break these:
  * gate_proj / up_proj are called AS MODULES so the LoRA path is
    included in Z. Never use F.linear(h, self.gate_proj.weight).
  * down_proj is used FUNCTIONALLY (its .weight is the fast weight
    initial state W0). LoRA must never target down_proj on TTT layers;
    build_lora_config() guarantees this via a regex.
  * W_target is zero-initialized, so at step 0 every TTT layer is
    bit-equivalent to the original MLP and the model reproduces the
    base model exactly. Verify with sanity_check_identity().
    Zero W_target also blocks gradient to the conv kernel until
    W_target moves, exactly like LoRA's B=0 init. Expected behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ttt_config import TTTConfig


# ===========================================================================
# Embedding tap. One hook on embed_tokens makes the token embeddings X0
# available to every TTT layer without changing model signatures. Used
# when cfg.v_source="embedding"; the "hidden_state" mode skips the hook
# registration and bypasses the tap entirely.
# ===========================================================================
class EmbeddingTap:
    """Captures embed_tokens output each forward and, in stateful mode,
    keeps a rolling context of the last (conv_kernel_size - 1) embeddings
    so the causal conv has its left context during incremental decoding."""

    def __init__(self, conv_kernel_size: int):
        self.context_len = conv_kernel_size - 1
        self.current: Optional[torch.Tensor] = None   # [B, N_new, d]
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
        return None  # do not modify the output

    def reset_stream(self):
        self._rolling = None
        self.prev_context = None


# ===========================================================================
# Per-layer fast weight state for streaming inference.
# ===========================================================================
@dataclass
class TTTState:
    """Cumulative fast weight update plus the not-yet-committed partial
    chunk. delta has the same shape as down_proj.weight, [d_model, d_ff]."""

    delta: Optional[torch.Tensor] = None
    pending_z: list = field(default_factory=list)   # list of [B, n, d_ff]
    pending_v: list = field(default_factory=list)   # list of [B, n, d_model]
    pending_tokens: int = 0

    def reset(self):
        self.delta = None
        self.pending_z.clear()
        self.pending_v.clear()
        self.pending_tokens = 0


# ===========================================================================
# The TTT-enabled MLP block.
# ===========================================================================
class InPlaceTTTMLP(nn.Module):
    def __init__(self, original_mlp: nn.Module, hidden_size: int,
                 cfg: TTTConfig, tap: EmbeddingTap):
        super().__init__()
        # Reuse the pretrained projections. gate/up will later be wrapped
        # by LoRA; down stays a plain Linear whose weight is W0.
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn

        self.cfg = cfg
        self.tap = tap

        # ---- New TTT components (the only freshly initialized params) ----
        # Depthwise causal conv over the embedding sequence.
        self.target_conv = nn.Conv1d(
            in_channels=hidden_size, out_channels=hidden_size,
            kernel_size=cfg.conv_kernel_size, groups=hidden_size, bias=False,
        )
        # Init the conv as a pass-through of the most recent token. Sane
        # start; W_target = 0 makes the overall init exact-identity anyway.
        with torch.no_grad():
            self.target_conv.weight.zero_()
            self.target_conv.weight[:, :, -1] = 1.0

        # Zero-init: at step 0 V = conv(source) @ 0 = 0, so delta = 0, so
        # ttt_out = 0, so the model is bit-exact identity to the base MLP.
        # The non-zero randn init was tried (commit history) but caused the
        # carry to overfit to training-paper-specific directions; reverting
        # to the paper's zero init removes that pathology. Gradient escape
        # from zero is slower but more honest -- the carry only ever points
        # in a direction the loss explicitly rewards.
        self.w_target = nn.Parameter(torch.zeros(hidden_size, hidden_size))

        # Per-position output gate: output = base + sigmoid(W_g h) * ttt_out.
        # See TTTConfig.output_gate for rationale (opens a new gradient path
        # to W_target through the gate's learning, avoiding cumsum-averaging
        # starvation). Bias init defaults to negative (sigmoid ~ 0.12) so
        # the carry is mostly closed at start and must be EARNED open by
        # gradient; this is "normalization by init" against the overfit
        # mode where a freely-open gate amplifies noise. Weight init tiny
        # so the gate is mostly bias-determined until learning shapes it.
        # None when disabled, so _gated() short-circuits cheaply.
        if cfg.output_gate:
            self.output_gate = nn.Linear(hidden_size, 1, bias=True)
            with torch.no_grad():
                self.output_gate.bias.fill_(cfg.output_gate_bias_init)
                nn.init.normal_(self.output_gate.weight, std=1e-3)
        else:
            self.output_gate = None

        # Stashed by _gated() during forward when the gate is active and
        # the module is in training mode, consumed by gate_reg_term(model)
        # in the training loop to add an L2 penalty on the gate output.
        # None signals "no penalty available" (gate disabled OR last
        # forward was in eval mode); the helper skips silently.
        self._gate_l2: Optional[torch.Tensor] = None

        # Runtime switches, controlled via the module-tree helpers below.
        self.ttt_evolve = True       # False => behaves as a vanilla MLP (+LoRA)
        self.stateful = False        # True  => persist fast weights across calls
        self.state = TTTState()

        # Per-module rolling buffer of past hidden_states, used in stream
        # mode when cfg.v_source="hidden_state" so the causal conv sees
        # real past tokens instead of zero-padding. Only populated when
        # the layer is in stream mode; reset_stream_state() clears it.
        self._hidden_context: Optional[torch.Tensor] = None

        # Session-persistent training (TBPTT-style). carried_delta is the
        # fp32 fast weight delta accumulated over previous papers in the
        # current session. _next_carried is STAGED during forward and
        # promoted by advance_session_state() after backward; staging is
        # idempotent, which keeps gradient-checkpointing recomputation
        # (forward runs twice per backward) from corrupting the state.
        self.session_mode = False
        self.carried_delta: Optional[torch.Tensor] = None
        self._next_carried: Optional[torch.Tensor] = None

    # ----------------------------------------------------------- targets --
    def _targets(self, source: torch.Tensor,
                 left_context: Optional[torch.Tensor]) -> torch.Tensor:
        """V = Conv1D(source) @ W_target, shape [B, N, d_model].

        `source` is the tensor that feeds the conv (X0 from the embedding
        tap when cfg.v_source="embedding", layer-local hidden_states when
        cfg.v_source="hidden_state"); the dispatch lives in _v_source().

        Padding is split between left and right by cfg.v_bidirectional:
          * False (default, causal): left-only pad => each output sees the
            K positions strictly to the left (including itself). Preserves
            chunk-causality so the parallel scan equals the sequential
            apply-then-update.
          * True (centered window): symmetric pad so output at position n
            sees ~K/2 past + n + ~K/2 future tokens. Streaming IGNORES this
            (future tokens are unavailable across call boundaries) -- the
            streaming path is the one with non-None left_context, so we
            detect it here and force-causal.

        `left_context` is the per-stream rolling buffer of source-tensor
        positions from previous calls (None outside streaming); it
        substitutes for left padding so the conv sees real past tokens
        instead of zeros."""
        K = self.cfg.conv_kernel_size
        streaming = left_context is not None
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

        x = x.transpose(1, 2)                           # [B, d, N(+ctx)]
        if left_pad > 0 or right_pad > 0:
            x = F.pad(x, (left_pad, right_pad))
        v = self.target_conv(x).transpose(1, 2)         # [B, N, d]
        if streaming:
            v = v[:, -source.shape[1]:, :]              # keep only new positions
        return v @ self.w_target

    # ---------------------------------------------------- v-source dispatch --
    def _v_source(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Tensor that feeds target_conv. Single source of truth so scan
        and stream paths can't disagree about which signal V is built
        from."""
        if self.cfg.v_source == "hidden_state":
            return hidden_states
        return self.tap.current

    def _v_left_context(self) -> Optional[torch.Tensor]:
        """Rolling left-context buffer matched to _v_source. Same DRY
        rationale: single dispatch site for the v-input pipeline."""
        if self.cfg.v_source == "hidden_state":
            return self._hidden_context
        return self.tap.prev_context

    def _update_hidden_context(self, hidden_states: torch.Tensor):
        """Mirror of EmbeddingTap's rolling buffer, but per-module --
        hidden_states differ between layers, so each TTT module keeps
        its own (conv_kernel_size - 1)-token tail. No-op when v_source
        is "embedding" (the shared tap owns the buffer in that mode);
        call unconditionally from the stream path."""
        if self.cfg.v_source != "hidden_state":
            return
        ctx_len = self.cfg.conv_kernel_size - 1
        if ctx_len <= 0:
            return
        joined = hidden_states if self._hidden_context is None else torch.cat(
            [self._hidden_context, hidden_states], dim=1
        )
        self._hidden_context = joined[:, -ctx_len:, :].detach()

    def reset_stream_state(self):
        """Drop the per-module streaming buffers AND committed fast-weight
        state. Called from the model-tree helper reset_fast_weights()."""
        self.state.reset()
        self._hidden_context = None

    def reset_v_context(self):
        """Soft turn-boundary reset: clear only the conv left-context
        buffer that matches cfg.v_source. Leaves state.delta and
        pending_z/pending_v intact -- that's the cross-turn memory."""
        self._hidden_context = None

    # ---------------------------------------------------------- clipping --
    def _clip(self, delta: torch.Tensor) -> torch.Tensor:
        """Cap ||eta * delta||_F at clip_tau. OFF by default, see
        TTTConfig.clip_enabled for why."""
        if not self.cfg.clip_enabled:
            return delta
        if self.cfg.clip_at_inference_only and self.training:
            return delta
        norm = (self.cfg.eta * delta).norm(p="fro", dim=(-2, -1), keepdim=True)
        scale = (self.cfg.clip_tau / norm.clamp_min(1e-12)).clamp(max=1.0)
        return delta * scale

    # ------------------------------------------------------------- gate --
    def _gated(self, ttt_term: torch.Tensor,
               hidden_states: torch.Tensor) -> torch.Tensor:
        """Multiply the TTT contribution by sigmoid(W_g h) per position when
        output_gate is configured. Single source of truth so both the scan
        and stream paths apply gating identically (DRY -- if you change the
        formulation here it's a one-place edit). When the gate is None
        (cfg.output_gate=False) the term passes through unchanged, so the
        original paper formulation is preserved as the default.

        Side effect during training: stashes mean-square of the gate output
        in self._gate_l2 so gate_reg_term(model) can pick it up and add an
        L2 penalty to the training loss. Skipped under eval/no_grad."""
        if self.output_gate is None:
            return ttt_term
        gate = torch.sigmoid(self.output_gate(hidden_states))   # [B, N, 1]
        if self.training and torch.is_grad_enabled():
            self._gate_l2 = (gate ** 2).mean()
        return gate * ttt_term

    # --------------------------------------------------------- main path --
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        z = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)

        if not self.ttt_evolve and not self.stateful:
            # Pure ablation path, identical to the original MLP (+LoRA on
            # gate/up). Note down_proj called as a module is fine here
            # because TTT-layer down_proj never carries LoRA.
            return self.down_proj(z)

        if self.stateful:
            return self._stream_forward(z, hidden_states)
        return self._scan_forward(z, hidden_states)

    # ----------------------------------------------- stateless scan path --
    def _scan_forward(self, z: torch.Tensor,
                     hidden_states: torch.Tensor) -> torch.Tensor:
        """Training / whole-sequence eval. Parallel chunk scan.

        session_mode off: fast weights implicitly reset every call,
        which is the paper's per-document reset.

        session_mode on: the scan starts from carried_delta (state from
        previous papers in the session, fp32, detached) and the new
        total delta of this sequence is staged for carry-over. Gradients
        never cross the paper boundary, truncated-BPTT style.

        hidden_states feeds the output gate (when configured) AND, when
        cfg.v_source="hidden_state", feeds target_conv via _v_source().
        Keep computing z from it outside this function so the no-evolve
        ablation path can short-circuit without touching scan logic."""
        B, N, d_ff = z.shape
        C = self.cfg.chunk_size
        w0 = self.down_proj.weight                          # [d, d_ff]
        base_out = z @ w0.T                                 # frozen-W0 part

        carried = None
        if self.session_mode and self.carried_delta is not None:
            if self.carried_delta.shape[0] != B:
                raise RuntimeError(
                    "Batch size changed mid-session; keep B constant "
                    f"(state B={self.carried_delta.shape[0]}, input B={B})."
                )
            carried = self.carried_delta.to(z.dtype)        # apply-time cast

        if N <= C and carried is None and not self.session_mode:
            # Single chunk, no carry => no update can ever be applied.
            # Gate isn't needed either; the only TTT term would be zero.
            return base_out

        v = self._targets(self._v_source(hidden_states),
                          left_context=None)                # [B, N, d]

        # Pad to a multiple of C so we can chunk with a reshape.
        n_chunks = (N + C - 1) // C
        pad = n_chunks * C - N
        if pad:
            z = F.pad(z, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))

        zc = z.view(B, n_chunks, C, d_ff)
        vc = v.view(B, n_chunks, C, -1)

        # Per-chunk deltas V^T Z, then an exclusive cumulative sum so chunk
        # i sees only deltas from chunks < i (strict chunk causality).
        deltas = torch.einsum("bkcd,bkcf->bkdf", vc, zc)    # [B, k, d, d_ff]
        if self.cfg.normalize_delta_by_chunk:
            # Divide each chunk by its ACTUAL non-padded token count.
            # All but the last chunk are full (size C); the last chunk has
            # C - pad real tokens. Dividing the last chunk by C would
            # silently halve its contribution -- common when papers
            # aren't a multiple of chunk_size.
            chunk_sizes = [C] * n_chunks
            if pad:
                chunk_sizes[-1] = C - pad
            chunk_sizes = torch.tensor(
                chunk_sizes, dtype=deltas.dtype, device=deltas.device,
            ).clamp_min(1).view(1, n_chunks, 1, 1)
            deltas = deltas / chunk_sizes
        cum = deltas.cumsum(dim=1)
        cum = torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], dim=1)
        if carried is not None:
            # Previous papers' state is visible from chunk 0 onward.
            cum = cum + carried.unsqueeze(1)
        cum = self._clip(cum)

        # O = Z W0^T + eta * Z S^T, avoiding materializing per-chunk W_eff.
        ttt_out = self.cfg.eta * torch.einsum("bkcf,bkdf->bkcd", zc, cum)
        ttt_out = ttt_out.reshape(B, n_chunks * C, -1)[:, :N, :]

        if self.session_mode:
            # Stage this paper's total delta in fp32 (bf16 accumulation
            # drifts over many papers). Detach = the TBPTT boundary.
            total = deltas.sum(dim=1).detach().float()
            self._next_carried = (
                total if self.carried_delta is None
                else self.carried_delta + total
            )

        return base_out + self._gated(ttt_out, hidden_states)

    # --------------------------------------------- stateful stream path --
    @torch.no_grad()
    def _stream_forward(self, z: torch.Tensor,
                       hidden_states: torch.Tensor) -> torch.Tensor:
        """Autoregressive inference. Apply current fast weights, buffer
        the chunk in progress, commit an update each time a full chunk
        completes (only if ttt_evolve is True).

        hidden_states feeds the output gate (when configured), is sliced
        per-position so gating is exact, AND -- when cfg.v_source=
        "hidden_state" -- feeds the conv via _v_source() with the
        per-module rolling buffer as left context."""
        source = self._v_source(hidden_states)
        v = self._targets(source, left_context=self._v_left_context())
        self._update_hidden_context(hidden_states)

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
            # Apply with the CURRENT state, then buffer (apply-then-update).
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
        zc = torch.cat(st.pending_z, dim=1)              # [B, C, d_ff]
        vc = torch.cat(st.pending_v, dim=1)              # [B, C, d]
        # Accumulate in fp32; bf16 drifts over thousands of commits.
        delta = torch.einsum("bcd,bcf->bdf", vc.float(), zc.float())
        if self.cfg.normalize_delta_by_chunk:
            delta = delta / self.cfg.chunk_size
        new = delta if st.delta is None else st.delta + delta
        st.delta = self._clip(new)
        st.pending_z.clear()
        st.pending_v.clear()
        st.pending_tokens = 0


# ===========================================================================
# Model-tree helpers. Everything below operates on a full HF model.
# ===========================================================================
def patch_model_with_ttt(model, cfg: TTTConfig):
    """Replace the MLP on cfg.layer_indices with InPlaceTTTMLP and attach
    the embedding tap. Call BEFORE get_peft_model. The tap is stashed
    on the model as `_ttt_tap` for downstream helpers.

    When cfg.v_source="hidden_state" the tap is still constructed (so the
    module-tree helpers stay uniform) but its forward hook is NOT
    registered -- nothing reads tap.current in that mode, so the hook
    would just allocate per-step."""
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


def set_ttt_evolve(model, evolve: bool):
    """Master switch. False freezes fast weight evolution everywhere;
    already-accumulated state (if any) keeps being applied."""
    for m in iter_ttt_modules(model):
        m.ttt_evolve = evolve


def set_ttt_stateful(model, stateful: bool):
    """True enables the streaming path with persistent fast weights."""
    model._ttt_tap.stateful = stateful
    for m in iter_ttt_modules(model):
        m.stateful = stateful


def reset_fast_weights(model):
    """Drop accumulated fast weight state, e.g. at document or session
    boundaries, restoring the pretrained-plus-LoRA model. Clears BOTH
    the shared embedding tap's rolling buffer (used when v_source=
    "embedding") and the per-module hidden-state buffer (used when
    v_source="hidden_state"), so this is correct in either mode."""
    model._ttt_tap.reset_stream()
    for m in iter_ttt_modules(model):
        m.reset_stream_state()


def reset_v_left_context(model):
    """Soft turn-boundary reset: clear ONLY the conv left-context buffer
    that matches cfg.v_source (the embedding-tap rolling buffer or the
    per-module hidden buffer). Leaves state.delta + pending chunk
    buffers intact -- the chat path relies on these being the only memory
    carried across turns. Use this instead of reset_fast_weights when you
    want the next call's conv to start with zero left context but the
    fast weights to keep accumulating from where they left off."""
    model._ttt_tap.reset_stream()
    for m in iter_ttt_modules(model):
        m.reset_v_context()


# ---------------------------------------------------------------------------
# Session-persistent training lifecycle (TBPTT-style carry across papers).
# Call order per session:
#     reset_session_state(model)
#     for paper in session:
#         loss = model(paper); loss.backward()
#         advance_session_state(model)
# ---------------------------------------------------------------------------
def set_session_mode(model, on: bool):
    for m in iter_ttt_modules(model):
        m.session_mode = on


def reset_session_state(model):
    """Session boundary. Fast weights restart from the pretrained W0."""
    for m in iter_ttt_modules(model):
        m.carried_delta = None
        m._next_carried = None


def advance_session_state(model):
    """Promote the delta staged by the last forward. Call exactly once
    per paper, AFTER backward. Safe under gradient checkpointing since
    staging recomputes to the identical value; only the promotion here
    advances the state."""
    for m in iter_ttt_modules(model):
        if m._next_carried is not None:
            m.carried_delta = m._next_carried
            m._next_carried = None


def session_state_norms(model) -> dict:
    """||eta * carried||_F / ||W0||_F keyed by the actual base-model layer
    index (e.g. 5, 11, 17, ...), not by enumeration order. THE health
    metric for persistence. If this grows without bound across a session,
    you have found the point where a forgetting mechanism becomes mandatory."""
    modules = list(iter_ttt_modules(model))
    layer_indices = modules[0].cfg.layer_indices if modules else ()
    out = {}
    for layer_idx, m in zip(layer_indices, modules):
        if m.carried_delta is None:
            out[layer_idx] = 0.0
            continue
        num = (m.cfg.eta * m.carried_delta).norm()
        den = m.down_proj.weight.detach().float().norm()
        out[layer_idx] = float(num / den)
    return out


def stateful_state_norms(model) -> dict:
    """Same health metric as session_state_norms, but reads the
    STREAMING fast-weight state (m.state.delta) instead of the TBPTT
    carry (m.carried_delta). Use this from the inference chat path,
    where stateful=True is set and state.delta accumulates across
    forward calls; session_state_norms would return 0.0 there because
    carried_delta is only populated when session_mode=True."""
    modules = list(iter_ttt_modules(model))
    layer_indices = modules[0].cfg.layer_indices if modules else ()
    out = {}
    for layer_idx, m in zip(layer_indices, modules):
        if m.state.delta is None:
            out[layer_idx] = 0.0
            continue
        num = (m.cfg.eta * m.state.delta).norm()
        den = m.down_proj.weight.detach().float().norm()
        out[layer_idx] = float(num / den)
    return out


def mean_state_ratio(norms: dict) -> float:
    """Mean of per-layer ||eta * delta||_F / ||W0||_F. Empty dict (no TTT
    modules patched, or norms not computed) is 0.0 so callers can log
    unconditionally."""
    return sum(norms.values()) / len(norms) if norms else 0.0


def gate_reg_term(model) -> torch.Tensor | float:
    """Sum of mean-square gate outputs across TTT layers, ready to be
    added to the training loss multiplied by TRAIN_CFG.gate_reg_weight.
    Returns 0.0 (a Python float, not a tensor) when no module has stashed
    a value -- gate disabled, model in eval, or the no-evolve ablation
    path skipped the gate -- so the caller can do `loss + w * gate_reg`
    unconditionally and the no-op cost is a multiply by zero."""
    accum = None
    for m in iter_ttt_modules(model):
        if m._gate_l2 is not None:
            accum = m._gate_l2 if accum is None else accum + m._gate_l2
    return accum if accum is not None else 0.0


def stream_pending_progress(model) -> tuple[int, int]:
    """Returns (pending_tokens, chunk_size) for the streaming path.
    pending_tokens is how full the not-yet-committed chunk buffer is on
    any one TTT layer (all layers see the same token stream so it
    suffices to read one). When pending_tokens hits chunk_size the
    layer commits a delta into state.delta. Useful for chat-side
    diagnostics: a state_ratio of 0.0 with pending_tokens just shy of
    chunk_size means carry has not been engaged yet but the next
    chunk-worth of tokens will push it over."""
    modules = list(iter_ttt_modules(model))
    if not modules:
        return 0, 0
    m = modules[0]
    return int(m.state.pending_tokens), int(m.cfg.chunk_size)


def export_fast_weights(model) -> dict:
    """Snapshot accumulated deltas for cross-session persistence."""
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


# ===========================================================================
# LoRA wiring and parameter groups.
# ===========================================================================
_PEFT_PREFIX = "base_model.model."
TTT_PARAM_MARKERS = ("target_conv", "w_target", "output_gate")


def ttt_down_suffixes(cfg: TTTConfig) -> set:
    """Parameter-name suffixes of the fully trained fast weight initial
    states (down_proj on TTT layers). Single definition; four call
    sites previously each rebuilt this set (DRY)."""
    return {f"layers.{i}.mlp.down_proj.weight" for i in cfg.layer_indices}


def strip_peft_prefix(name: str) -> str:
    """PEFT wraps the model and prefixes every parameter name; strip it
    so checkpoint keys are relative to the base model and load whether
    or not the model is PEFT-wrapped at the time."""
    return name[len(_PEFT_PREFIX):] if name.startswith(_PEFT_PREFIX) else name


def build_lora_target_regex(num_layers: int, cfg: TTTConfig) -> str:
    """LoRA targets attention + gate/up everywhere, plus down_proj ONLY
    on non-TTT layers. PEFT matches with re.fullmatch on module names
    like 'model.layers.5.mlp.down_proj'. Pure function so the regex,
    whose silent failure mode is LoRA landing on a fast weight, is
    directly unit-testable without PEFT installed."""
    non_ttt = [str(i) for i in range(num_layers) if i not in cfg.layer_indices]
    return (
        r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj)"
        r"|.*\.layers\.(" + "|".join(non_ttt) + r")\.mlp\.down_proj"
    )


def build_lora_config(num_layers: int, cfg: TTTConfig,
                      r: int, alpha: int, dropout: float):
    from peft import LoraConfig

    return LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=build_lora_target_regex(num_layers, cfg),
        bias="none", task_type="CAUSAL_LM",
    )


def _classify_ttt_param(name: str, ttt_down: set) -> str | None:
    """Map a parameter name to its training group, or None for params
    that are not part of TTT/LoRA training. Single source of truth so
    unfreeze and the optimizer grouping can't disagree about which name
    belongs where."""
    if "lora_" in name:
        return "lora"
    if any(marker in name for marker in TTT_PARAM_MARKERS):
        return "new"
    if any(name.endswith(suffix) for suffix in ttt_down):
        return "wdown"
    return None


def unfreeze_ttt_params(model, cfg: TTTConfig):
    """PEFT freezes everything non-LoRA; re-enable grads for the TTT
    trainables, which we manage outside PEFT on purpose (modules_to_save
    has known sharp edges with custom modules)."""
    ttt_down = ttt_down_suffixes(cfg)
    for name, p in model.named_parameters():
        if _classify_ttt_param(name, ttt_down) in ("new", "wdown"):
            p.requires_grad_(True)


def build_param_groups(model, cfg: TTTConfig, lr_lora: float,
                       lr_wdown: float, lr_new: float,
                       wd_full: float, wd_lora: float):
    """Three optimizer groups: LoRA, TTT-layer W_down, fresh modules.
    Also returns the groups by name for gradient-norm monitoring."""
    ttt_down = ttt_down_suffixes(cfg)
    groups = {"lora": [], "wdown": [], "new": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        group = _classify_ttt_param(name, ttt_down)
        if group is None:
            raise RuntimeError(f"Unclassified trainable parameter: {name}")
        groups[group].append(p)
    optim_groups = [
        {"params": groups["lora"], "lr": lr_lora, "weight_decay": wd_lora},
        {"params": groups["wdown"], "lr": lr_wdown, "weight_decay": wd_full},
        {"params": groups["new"], "lr": lr_new, "weight_decay": wd_full},
    ]
    return optim_groups, groups


# ===========================================================================
# Checkpointing of the non-LoRA trainables. LoRA itself is saved with
# model.save_pretrained (the PEFT adapter), these helpers cover the rest.
# Keys are stored relative to the BASE model so loading works whether or
# not the model is wrapped by PEFT at load time.
# ===========================================================================
def save_ttt_state_dict(model, path: str, cfg: TTTConfig):
    ttt_down = ttt_down_suffixes(cfg)
    out = {}
    for name, p in model.named_parameters():
        key = strip_peft_prefix(name)
        if any(m in key for m in TTT_PARAM_MARKERS) or \
           any(key.endswith(s) for s in ttt_down):
            out[key] = p.detach().cpu()
    torch.save(out, path)


def load_ttt_state_dict(model, path: str):
    saved = torch.load(path, map_location="cpu")
    by_suffix = dict(saved)
    loaded = 0
    for name, p in model.named_parameters():
        key = strip_peft_prefix(name)
        if key in by_suffix:
            p.data.copy_(by_suffix[key].to(p.device, p.dtype))
            loaded += 1
    if loaded != len(saved):
        raise RuntimeError(
            f"TTT checkpoint mismatch, saved {len(saved)} tensors, "
            f"loaded {loaded}. Check layer indices match the checkpoint."
        )
