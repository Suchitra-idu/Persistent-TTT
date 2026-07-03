# The TTT Mechanism

Math, code walkthrough, and the two execution paths. Read this before
touching [`inplace_ttt.py`](../inplace_ttt.py).

## The math in six lines

For TTT layers, with:
- `Z = silu(gate_proj(H)) * up_proj(H)` — the standard gated MLP activation
- `V = CausalConv1D(source) @ W_target` — LM-aligned per-position targets
- Chunks of size `C` (default 400)

Per-chunk:

```
apply:    O_i = Z_i @ (W_down + eta · S_i)^T
gate:     O_i = base_out + sigmoid(W_g · h) · (O_i - base_out)
update:   S_{i+1} = S_i + V_i^T @ Z_i / C          (S_0 = 0)
```

Chunk `i` sees only updates from strictly earlier chunks (exclusive
cumsum + causal conv). The cumulative-sum formulation used in code is
mathematically identical to the sequential apply-then-update loop.

### Session mode

Session mode changes exactly two things:

1. `S_0` starts from `carried_delta` (previous items' final state,
   fp32, detached), instead of zero.
2. At the item boundary,
   `carried_delta ← carried_decay · carried_delta + this_item_total`
   (EMA staging). `carried_decay = 1.0` reproduces pure accumulation.

Gradients never cross the item boundary — that's the truncated BPTT
(TBPTT) property. `_next_carried` is staged during forward and promoted
by `advance_session_state()` after backward. Staging is **idempotent**
so gradient-checkpointing recompute (which runs forward twice per
backward) doesn't double-count.

### Output gate

A per-position sigmoid gate modulates the TTT contribution:

```
gate = sigmoid(output_gate(hidden_states))    # shape [B, N, 1]
final = base_out + gate * ttt_out
```

Bias initialized to `-2.0` (sigmoid ≈ 0.12) so the gate starts mostly
closed and must learn to open. Provides an extra gradient path for
`W_target` beyond the "raw linear attention" baseline.

Optional L2 reg (`gate_reg_weight`) penalizes `mean(gate^2)` if you
want to bias the gate closed. Default is 0 (no regularization).

## The two execution paths

### `_scan_forward` — parallel scan (training + whole-sequence eval)

```python
# 1. Compute Z from H (already includes LoRA path)
z = act_fn(gate_proj(H)) * up_proj(H)         # [B, N, d_ff]

# 2. Read the tap. Source is either embeddings or per-layer hidden state.
x0 = self.tap.current                          # [B, N, d]

# 3. If session_mode and we have a prior state, apply it
carried = None
if session_mode and carried_delta is not None:
    carried = carried_delta.to(z.dtype)        # apply-time cast to bf16

# 4. Base output (frozen W0 path)
base_out = z @ w0.T                            # [B, N, d]

# 5. Fast-path: skip TTT if it can't do anything useful
if N <= C and carried is None and not session_mode:
    return base_out

# 6. Compute per-position targets
v = _targets(x0, left_context=None)            # [B, N, d]

# 7. Chunk into [B, n_chunks, C, d_ff] and [B, n_chunks, C, d]
zc = z.view(B, n_chunks, C, d_ff)
vc = v.view(B, n_chunks, C, -1)

# 8. Per-chunk deltas (contract over C)
deltas = einsum("bkcd,bkcf->bkdf", vc, zc)     # [B, n_chunks, d, d_ff]
# Optionally normalize by chunk size
if normalize_delta_by_chunk:
    deltas /= C  # (using actual token count, respecting the last-chunk padding)

# 9. EXCLUSIVE cumsum: chunk i sees only chunks < i
cum = deltas.cumsum(dim=1)
cum = torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], dim=1)

# 10. Add carried state (visible from chunk 0 onward)
if carried is not None:
    cum = cum + carried.unsqueeze(1)

# 11. Clip
cum = self._clip(cum)

# 12. TTT contribution: eta * Z W_delta^T
ttt_out = eta * einsum("bkcf,bkdf->bkcd", zc, cum)
ttt_out = ttt_out.reshape(B, n_chunks * C, -1)[:, :N, :]

# 13. Stage next carried (fp32, detached — TBPTT boundary)
if session_mode:
    total = deltas.sum(dim=1).detach().float()
    self._next_carried = (
        total if carried_delta is None
        else carried_decay * carried_delta + total
    )

# 14. Combine with gate
return base_out + gate(ttt_out, H)
```

Called during training (session-mode threaded via
`set_session_mode`) and whole-sequence eval (session-mode threaded
via `session_perplexity`). Never called during autoregressive
generation.

### `_stream_forward` — incremental stream (autoregressive inference)

```python
@torch.no_grad()
def _stream_forward(z, hidden_states):
    x0 = self.tap.current
    v = _targets(x0, left_context=self.tap.prev_context)
    st = self.state
    C = self.cfg.chunk_size

    while pos < N:
        room = C - st.pending_tokens
        take = min(room, N - pos)
        z_part = z[:, pos:pos + take]

        # apply-then-update
        out = z_part @ w0.T
        if st.delta is not None:
            out = out + eta * (z_part @ st.delta.transpose(-1, -2))
        out = gate(out, hidden_states_slice)

        outputs.append(out)

        if ttt_evolve:
            st.pending_z.append(z_part)
            st.pending_v.append(v[:, pos:pos + take])
            st.pending_tokens += take
            if st.pending_tokens == C:
                self._commit_chunk()          # updates st.delta
        pos += take

    return concat(outputs)
```

Chunks commit as soon as they reach `chunk_size` tokens. The pending
buffer + `state.delta` together are the full fast-weight state at any
moment.

### Why two paths, not one?

The scan is efficient for many-token forward passes (training/eval)
because it can batch chunks in parallel via einsum. The stream is
necessary for autoregressive generation because you don't have future
tokens available and need to commit updates as they arrive.

They're mathematically equivalent when applied to the same token
stream with `chunk_size` alignment. Verified by
[`tests/test_scan_math.py`](../tests/test_scan_math.py).

## The clip

```python
def _clip(self, delta):
    if not self.cfg.clip_enabled:
        return delta
    if self.cfg.clip_at_inference_only and self.training:
        return delta
    norm = (eta * delta).norm(p="fro", dim=(-2, -1), keepdim=True)
    scale = (clip_tau / norm.clamp_min(1e-12)).clamp(max=1.0)
    return delta * scale
```

Frobenius clip on `||eta * cum||_F` per chunk position. `clip_tau=5.0`
by default. When active (i.e., `||eta*cum||_F > 5`), only direction
survives — the magnitude is scaled down to exactly 5.

**Important property (experimentally verified):** at 0.6B, the model
learns a direction that's *useful* in the direction-only regime. Raising
`clip_tau` at inference does NOT help (and hurts if trained with the
lower value) because the model was calibrated for clipped magnitudes.

See [failure-modes.md](failure-modes.md#state-saturation) for more.

## The tap

An `EmbeddingTap` object hooks either the embed_tokens output
(`v_source="embedding"`) or, when `v_source="hidden_state"`, we use
per-layer inputs read via a similar hook mechanism.

Per-layer hook stashes `self.tap.current` on every forward, giving
every TTT layer access to X0 without changing model signatures.
Streaming mode also maintains a rolling `_rolling` buffer of the last
`conv_kernel_size - 1` embeddings so the causal conv has left context
during incremental decoding.

At turn boundaries in chat, `reset_v_left_context()` clears this
rolling buffer so the next turn's causal conv doesn't bleed prior-turn
embeddings into the new turn.

## Init scheme

```python
# target_conv (depthwise causal Conv1D):
#   All weights zero except the LAST position, which is 1.0.
#   This makes V(t) = source(t) @ W_target initially — pass-through.
self.target_conv.weight.zero_()
self.target_conv.weight[:, :, -1] = 1.0

# W_target: zero. This makes V = 0 initially → delta = 0 → TTT term = 0.
self.w_target = nn.Parameter(torch.zeros(hidden_size, hidden_size))

# Output gate (if enabled): near-closed initialization
self.output_gate = nn.Linear(hidden_size, 1, bias=True)
self.output_gate.bias.fill_(-2.0)                # sigmoid(-2) ≈ 0.12
nn.init.normal_(self.output_gate.weight, std=1e-3)
```

**Why zero W_target:** at step 0 every TTT layer is bit-exact identical
to the original gated MLP. `sanity_check` in
[`train_modal.py`](../train_modal.py) verifies this. If sanity_check
fails, the wiring is broken and training would happen on top of a
different model than base Qwen3.

**Downside of zero W_target:** the model sits in a "dead basin" at step
0 — gradient signal to W_target is small until it moves. Escape from
the basin is via the first activation-driven gradient spike. See
[failure-modes.md](failure-modes.md#dead-basin) for what to do if
escape doesn't happen at scale.

## Session lifecycle helpers

Public API in [`inplace_ttt.py`](../inplace_ttt.py):

- **Toggling modes** — done inline on each TTT module (walked via
  `iter_ttt_modules(model)`): `m.session_mode = True|False`,
  `m.stateful = True|False`, `m.ttt_evolve = True|False`. No wrapper
  helpers; the flags are set directly.
- `reset_session_state(model)` — clear `carried_delta` and `_next_carried` on every TTT module (session boundary).
- `advance_session_state(model)` — promote `_next_carried → carried_delta` after backward. Idempotent (safe under gradient checkpointing recompute).
- `state_norms(model, source="session"|"stream")` — per-layer `||eta * delta||_F / ||W_down||_F`. `source="session"` reads `carried_delta`, `source="stream"` reads `state.delta`. Diagnostic.
- `mean_state_ratio(norms)` — mean across TTT layers.

Streaming path (used by chat):
- `reset_fast_weights(model)` — full reset. Clears `state.delta`, pending chunk, and conv left-context on every TTT module + the embedding tap.
- `reset_v_left_context(model)` — soft turn boundary. Clears the causal-conv left-context only; `state.delta` and pending survive.
- `stream_pending_progress(model)` — `(pending_tokens, chunk_size)` for one TTT layer (all layers see the same stream).
- `export_fast_weights(model)` — snapshot the streaming state as a CPU dict.
- `import_fast_weights(model, snapshot)` — restore a snapshot to the model.

See the module docstring in [`inplace_ttt.py`](../inplace_ttt.py) for the
reset-ladder table (which state each reset helper clears vs preserves).

## Related docs

- [architecture.md](architecture.md) — how this fits into the whole system
- [config.md](config.md) — every mechanism knob
- [training.md](training.md) — session mode wiring at training time
- [inference.md](inference.md) — session mode + streaming wiring at eval time
- [failure-modes.md](failure-modes.md) — dead basin, saturation, and gradient collapse
