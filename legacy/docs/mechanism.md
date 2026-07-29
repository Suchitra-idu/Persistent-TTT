# The TTT Mechanism

The math, the code, and every design rationale. This is the most
important doc — everything else in `docs/` presumes the model in this
file. If you touch [`inplace_ttt.py`](../inplace_ttt.py), read this
first.

## Contents

1. [Motivation: fast weights vs slow weights](#motivation-fast-weights-vs-slow-weights)
2. [Notation and shapes](#notation-and-shapes)
3. [The core equations](#the-core-equations)
4. [Component walkthrough](#component-walkthrough)
5. [Chunk causality](#chunk-causality)
6. [The scan path](#the-scan-path)
7. [The stream path](#the-stream-path)
8. [Scan / stream equivalence](#scan--stream-equivalence)
9. [Session mode: TBPTT carry across items](#session-mode-tbptt-carry-across-items)
10. [Idempotence under gradient checkpointing](#idempotence-under-gradient-checkpointing)
11. [Initialization and the dead basin](#initialization-and-the-dead-basin)
12. [The Frobenius clip](#the-frobenius-clip)
13. [The output gate](#the-output-gate)
14. [`v_source`: embedding vs hidden-state](#v_source-embedding-vs-hidden-state)
15. [The tap](#the-tap)
16. [RMSNorm on the V source](#rmsnorm-on-the-v-source)
17. [fp32 boundaries](#fp32-boundaries)
18. [The batch-size-1 invariant](#the-batch-size-1-invariant)
19. [Where gradient goes](#where-gradient-goes)
20. [Public API surface](#public-api-surface)

---

## Motivation: fast weights vs slow weights

The base transformer's weights are **slow weights**: they change only
between training runs (or in this codebase, via LoRA + `down_proj`
fine-tuning during continual pretraining). Everything the model "knows"
lives there.

In-Place TTT adds a **fast weight** `S` on each TTT layer: a small
matrix that gets updated once per chunk of `C=50` tokens during a
forward pass, and is applied to reshape the next chunk's output. The
fast weight is:

- **Per-stream.** Each active sequence carries its own `S`. There is no
  batched fast weight; batching would mix sequences into a single state
  and destroy the per-context adaptation. This is why the code rejects
  `B > 1` at session boundaries.
- **Ephemeral, but persistable.** Within a forward pass it's re-derived
  from scratch (`_scan_forward`). Across items in a session, it can
  carry over (`carried_delta` + TBPTT). During autoregressive
  generation, it accumulates over the whole conversation
  (`state.delta` + pending chunk).
- **Structured as a rank-`d` matrix.** Not a scalar, not a diagonal —
  the fast weight is a full `d × d_ff` matrix that gets added to the
  slow-weight `W_down`, i.e. it's a *modification of the down-projection*.

The key claim: the model can encode useful **short-term memory** about
what it just read into `S`, and the down-projection reads that memory
out. LoRA can't do this — LoRA is a *training-time* modification of
slow weights; it changes with SGD, not with context. TTT provides a
runtime knob LoRA cannot.

### Why TTT on top of LoRA (not instead of)

- LoRA moves the slow weights toward the domain. This shrinks the
  overall loss floor. At 0.6B on arxiv ML papers: base ppl ≈ 37,
  LoRA-only ≈ 31.
- TTT adds a per-context adaptation on top. The fast weight `S` can
  remember what the current session has read.
- LoRA + TTT ≠ either alone. Empirically at 0.6B, the FULL configuration
  buys another 3–7 ppl of *per-paper* gap over LoRA-only, evaluated
  with carry vs no-carry on held-out papers.

The paper we build on is [Test-Time Training with Fast Weight Adaptation
(TTT-Linear)](https://arxiv.org/abs/2407.04620); this codebase
implements the in-place variant (fast weights *inside* the MLP block,
not as a separate module) with several deviations documented in the
sections below.

---

## Notation and shapes

| symbol | shape | meaning |
|---|---|---|
| `B` | scalar | Batch size. Always `1` at session boundaries. |
| `N` | scalar | Sequence length (tokens in this forward). |
| `d` | scalar | Hidden size (`hidden_size`). Qwen3-0.6B: 1024. |
| `d_ff` | scalar | FFN width. Qwen3-0.6B: 3072 (`ffn_intermediate_size`). |
| `C` | scalar | Chunk size (`cfg.chunk_size`). |
| `k` | scalar | Number of chunks per forward: `ceil(N / C)`. |
| `H` | `[B, N, d]` | Hidden states entering this MLP block. |
| `Z` | `[B, N, d_ff]` | Gated activation `silu(gate_proj(H)) * up_proj(H)`. |
| `X0` | `[B, N, d]` | The V-source (embedding output or per-layer hidden). |
| `V` | `[B, N, d]` | Causal-conv-processed X0, then linear via `W_target`. |
| `W_target` | `[d, d]` | Trainable `d × d` linear map from conv-output to V. **Zero-init.** |
| `target_conv` | `[d, 1, K]` | Depthwise Conv1D over X0 (kernel size `K`, `groups=d`). |
| `W_down` | `[d, d_ff]` | The transformer's original `down_proj.weight`. Used **functionally** as `W0`. |
| `S`, `carried_delta`, `_next_carried`, `state.delta` | `[B, d, d_ff]` | Fast weight matrices. Different lifetimes; see later sections. |
| `gate` | `[B, N, 1]` | Sigmoid-gate output modulating the TTT contribution. |

The `V` shape is `[B, N, d]`, not `[B, N, d_ff]`, because `V` and `Z`
get contracted along the `d`-dim vs `d_ff`-dim of the delta matrix. The
delta ends up `[d, d_ff]`, matching `W_down`'s shape. This is
deliberate: the fast weight is a modification of the down-projection,
so its shape must match.

---

## The core equations

### Per-chunk update, in one place

For chunk `i` with contents `Z_i ∈ [B, C, d_ff]` and
`V_i ∈ [B, C, d]`, and running state `S_i`:

```
apply:    O_i = Z_i @ (W_down + eta · S_i)ᵀ           # [B, C, d]
gate:     O_i = base_out_i + sigmoid(W_g · H_i) · (O_i - base_out_i)
update:   S_{i+1} = S_i + (V_iᵀ @ Z_i) / C            # [B, d, d_ff]
```

with `S_0 = 0` (or `S_0 = carried_delta` in session mode).

Equivalently, expanding `apply`:

```
O_i = Z_i @ W_downᵀ  +  eta · Z_i @ S_iᵀ
    ────┬──────      ────────┬──────────
    base output              TTT contribution
```

The base output is what the plain gated MLP would emit (with W_down
frozen). The TTT contribution is the fast-weight correction.

### Why this shape

Look at the dimensions of the update `V_iᵀ @ Z_i`:

- `V_i` is `[B, C, d]`, so `V_iᵀ` is `[B, d, C]`.
- `Z_i` is `[B, C, d_ff]`.
- Product `V_iᵀ @ Z_i` is `[B, d, d_ff]`. ✓ Matches `W_down` shape.

The update is a **rank-`C` update** (since `V_iᵀ Z_i` = sum of `C` outer
products), applied to a `d × d_ff` matrix. Chunk `i` therefore
communicates at most `C` bits of directional information to `S_{i+1}`.

Dividing by `C` (when `normalize_delta_by_chunk=True`) makes `eta`
approximately independent of the choice of `C`: doubling `C` doubles
the sum of outer products but halves the average per-token
contribution, so `S` grows at the same rate per token.

### Why `V_iᵀ @ Z_i` and not `Z_iᵀ @ Z_i` or similar

The rule `S ← V^T Z` is the *outer-product accumulator* form of a
per-token Hebbian-style update. It answers "what direction in `d_ff`
space would you push `W_down`'s rows to reduce the gap between the
current per-token output and the desired `V`?" Concretely: `V` is a
*target* the model wants the down-projection to produce more of, and
`Z` is the current input to the down-projection. Pushing `W_down`'s
rows toward `V` in the `Z` direction is exactly what this outer product
does. See the paper for the fuller derivation as a projection under a
squared-error loss.

Note that `V` in this codebase is **not** the token itself. It is
`causal_conv(X0) @ W_target`, which lets the model **learn** what to
push toward via `W_target`. At init, `V = 0` (since `W_target = 0`), so
the update is zero — the "dead basin" (see below).

---

## Component walkthrough

### `Z`: the gated activation

```python
z = self.act_fn(self.gate_proj(H)) * self.up_proj(H)     # [B, N, d_ff]
```

`gate_proj` and `up_proj` are called **as modules**, not with
`F.linear(H, self.gate_proj.weight)`. This is a subtle wiring rule:
LoRA adapters wrap the modules (via PEFT), so calling as a module
includes LoRA in the computation of `Z`. Calling functionally with
just `.weight` would bypass LoRA and silently produce a mixed model.

### `V`: what the update pushes toward

```python
v = self._targets(source, left_context=...)
# where _targets =
#   x_padded = pad_or_cat(x, left_context)         # [B, N + padding, d]
#   v_raw   = target_conv(x_padded.transpose(1,2)).transpose(1,2)
#             .[:, -N:, :]                          # [B, N, d]
#   return  v_raw @ W_target                        # [B, N, d]
```

`target_conv` is a **depthwise causal Conv1D** with kernel size `K = 8`
(default). "Depthwise" means `groups = d` — each of the `d` output
channels is a Conv1D over the corresponding input channel, not across
channels. This gives each of the `d` "features" its own 8-tap temporal
filter over the last 8 positions of the V source. Cheaper than full
Conv1D by a factor of `d`; expressive enough because W_target does the
cross-channel mixing after.

**Why causal padding.** During training the scan sees the full
sequence but we still pad on the left only (`left_pad = K - 1`) so
position `t`'s conv only sees positions `t-K+1 ... t`. This is the
"chunk causality" invariant — see [below](#chunk-causality).

The optional `v_bidirectional=True` uses symmetric pad instead of
causal pad. This breaks chunk causality: position `t`'s V sees `V(t+1)`,
so the fast weight learned before chunk `i+1` has access to tokens
inside chunk `i+1`. In next-token prediction that is technically a
label leak. The streaming path ignores this flag and stays causal to
avoid future-token peeks during generation. Recommended: leave
`False`.

### `S`: the fast weight

`S` never appears as a `Parameter`. It is a state tensor:

- In `_scan_forward`, `S` at each chunk position is a slice of a
  precomputed `cum` array. It has no gradient tracked to the outer
  optimizer — gradient flows *through* `S` back into `V` and `Z`, but
  `S` itself is not "in" the optimizer.
- In `_stream_forward`, `S = state.delta` is stored as an fp32 tensor
  on the model, mutated in-place inside `torch.no_grad`. No gradient
  path.
- In session mode, `carried_delta` is an fp32 tensor stashed on each
  TTT module, detached at the item boundary (TBPTT).

So `S` is a **runtime state**, not a slow weight. What the outer
optimizer trains is `W_target`, `target_conv`, `W_down` (in TTT layers,
gently), and the LoRA adapters elsewhere. The fast weight is derived
from these at runtime.

### `W_target`: the learned outer-product target

```python
self.w_target = nn.Parameter(torch.zeros(hidden_size, hidden_size))
```

Zero-init. This is the *only* zero-init parameter in the mechanism; it
is what makes `sanity_check` bit-exact identity at step 0 (see
[Initialization](#initialization-and-the-dead-basin) below).

The gradient signal to `W_target` at step 0 is small but nonzero
because it flows through the `Z @ S^T` term when `S` was updated by any
prior chunk's non-zero `V`. But at step 0, `V ≡ 0` (since `W_target = 0`
so `V = conv(X0) @ 0 = 0`), so the fast weight stays zero across the
entire forward. Gradient to `W_target` comes only from the `V = ... @
W_target` path where `V` is used in the delta computation — and the
delta is multiplied by the fast weight in the next chunk's apply, so
the gradient path is chained. The first non-zero step happens after
optimizer.step() nudges `W_target` off zero in some direction; from
there the mechanism "wakes up."

### `target_conv`: temporal receptive field for V

```python
self.target_conv = nn.Conv1d(
    in_channels=d, out_channels=d,
    kernel_size=K, groups=d, bias=False,
)
with torch.no_grad():
    self.target_conv.weight.zero_()
    self.target_conv.weight[:, :, -1] = 1.0
```

Init makes the conv a pure pass-through: at step 0,
`v_raw(t) = x0(t) · 1.0 = x0(t)`. So the temporal receptive field is
"only the current position" until `target_conv` learns something else
during training.

Cross-referencing with `W_target = 0`: at step 0,
`V = v_raw @ W_target = x0 @ 0 = 0` regardless of what the conv does.
So the pass-through init of the conv is not strictly necessary for
identity — it's chosen so that the earliest gradient signal has a
sensible "starting point" (each output channel = input channel from the
current position) instead of random.

### `W_down` as fast-weight origin (aka `W0`)

The transformer's original `down_proj.weight` is not renamed. In the
`_scan_forward` and `_stream_forward` bodies, we write:

```python
w0 = self.down_proj.weight
...
base_out = z @ w0.T
```

So we use `W_down` **functionally as `W0`**, without going through the
module. This is deliberate:

1. The LoRA regex explicitly excludes `down_proj` on TTT layers. If it
   did land there, LoRA would add its adapter to `W0` and the
   fast-weight interpretation would break: the "initial state" you're
   updating is no longer the pretrained down-projection but a
   pretrained + LoRA sum.
2. Functional use also skips the LoRA path even if the regex somehow
   were violated — a belt-and-braces safeguard.

`W_down` **is** trained (that's what the `wdown` param group does),
just gently — LR ≈ `3e-5` vs `1e-5` for LoRA. It's a pretrained fast
weight initial state; we want it to move but not to drift.

### `output_gate` and gate diagnostics

```python
self.output_gate = nn.Linear(d, 1, bias=True)
self.output_gate.bias.fill_(cfg.output_gate_bias_init)     # default -2.0
nn.init.normal_(self.output_gate.weight, std=1e-3)
```

Per-position scalar gate:

```python
gate = torch.sigmoid(self.output_gate(hidden_states))       # [B, N, 1]
final = base_out + gate * ttt_out
```

Two roles:

- **Gradient path for W_target.** Without a gate, the TTT contribution
  is added directly; the gradient signal to `W_target` is entirely
  mediated by the change in loss with respect to `ttt_out`. With a
  gate, the model gains an extra sink: it can partially open the gate
  to see how much benefit `ttt_out` provides, and the gradient to
  `W_target` is scaled by `gate` (not zeroed by it), so the mechanism
  can escape the dead basin more easily.
- **A safety valve.** In failure regimes (state saturated, direction
  useless), the model can close the gate and quietly ignore the TTT
  term.

Diagnostic stats `_gate_mean` / `_gate_std` are stashed under
`torch.no_grad()` on every forward call, including eval. A healthy gate
sits with mean in `(0.1, 0.9)` and std `> 0.05` (i.e. actually
modulating position-to-position). A gate stuck at either extreme means
either fully-off (TTT ignored) or fully-on (gate not modulating,
equivalent to no gate).

See [observability.md](observability.md#gate-mean-per-layer) for the
alert thresholds.

---

## Chunk causality

**Claim:** in `_scan_forward` at chunk `i`, the output tokens see
information only from chunks `< i` (plus the current chunk's `Z_i`).

Proof:

1. `S_i` is defined as `S_i = sum_{j<i} delta_j = sum_{j<i} V_jᵀ Z_j / C`.
   The `cum` construction in code makes this an *exclusive* prefix
   sum: `cum[i] = sum_{j<i} delta[j]`, with `cum[0] = 0`.
2. `V_j` at chunk `j` depends on `X0` at positions inside chunk `j`
   plus the previous `K-1` positions (from the causal conv). All of
   those positions are inside chunk `j` or earlier, so `V_j` doesn't
   see anything past chunk `j`.
3. Applying `S_i` to `Z_i` at chunk `i` therefore only sees information
   from chunks strictly before `i`, plus the intra-chunk `Z_i` itself
   which comes from `H_i`, which is the model's own past.

The key mechanical points in code:

- **Exclusive cumsum.** After `cum = deltas.cumsum(dim=1)`, we shift
  right by one and zero the first entry:

  ```python
  cum = torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], dim=1)
  ```

  So `cum[i]` now equals `sum_{j<i} delta[j]`, not `sum_{j≤i} delta[j]`.

- **Causal conv padding.** In `_targets`:

  ```python
  left_pad = K - 1                    # only when not v_bidirectional
  right_pad = 0
  ```

  This padding ensures position `t`'s conv output only depends on
  `X0[t-K+1 : t+1]`, never on `t+1` or later.

If either goes wrong (bidirectional conv, inclusive cumsum), the loss
computation cheats — the model sees the label tokens through the fast
weight. Symptom: train loss drops fast, eval ppl explodes.

---

## The scan path

`_scan_forward` is the parallel-chunk implementation used for
training and whole-sequence eval. It's more efficient than a Python
loop because it batches all `k` chunk updates into two einsums.

Full walkthrough with shapes:

```python
def _scan_forward(self, z, hidden_states):
    B, N, d_ff = z.shape                        # z:  [B, N, d_ff]
    C = self.cfg.chunk_size
    w0 = self.down_proj.weight                  # w0: [d, d_ff]
    base_out = z @ w0.T                         # [B, N, d]  — plain-MLP output

    # 1. Load carried state if in session mode
    carried = None
    if self.session_mode and self.carried_delta is not None:
        if self.carried_delta.shape[0] != B:
            raise RuntimeError("Batch size changed mid-session")
        carried = self.carried_delta.to(z.dtype)   # [B, d, d_ff], cast to bf16 at apply time

    # 2. Fast-path: nothing to do
    if N <= C and carried is None and not self.session_mode:
        return base_out

    # 3. Compute V from the tap
    v = self._targets(self._v_source(hidden_states),
                      left_context=None)          # v: [B, N, d]

    # 4. Pad to whole chunks
    n_chunks = (N + C - 1) // C
    pad = n_chunks * C - N
    if pad:
        z = F.pad(z, (0, 0, 0, pad))              # zero-pad last chunk
        v = F.pad(v, (0, 0, 0, pad))
    zc = z.view(B, n_chunks, C, d_ff)             # [B, k, C, d_ff]
    vc = v.view(B, n_chunks, C, -1)               # [B, k, C, d]

    # 5. Per-chunk deltas: contract over C
    deltas = torch.einsum("bkcd,bkcf->bkdf", vc, zc)   # [B, k, d, d_ff]
    if self.cfg.normalize_delta_by_chunk:
        # Divide by ACTUAL non-padded token count for the last chunk.
        chunk_sizes = [C] * n_chunks
        if pad:
            chunk_sizes[-1] = C - pad
        chunk_sizes = torch.tensor(chunk_sizes, ...).view(1, n_chunks, 1, 1)
        deltas = deltas / chunk_sizes

    # 6. Exclusive cumsum → strict chunk causality
    cum = deltas.cumsum(dim=1)                    # [B, k, d, d_ff], INCLUSIVE
    cum = torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], dim=1)  # → EXCLUSIVE

    # 7. Add carry (present from chunk 0 onward)
    if carried is not None:
        cum = cum + carried.unsqueeze(1)          # broadcast over k

    # 8. Frobenius clip
    cum = self._clip(cum)                         # magnitude bound, direction preserved

    # 9. TTT contribution: eta · Z · S^T
    ttt_out = self.cfg.eta * torch.einsum("bkcf,bkdf->bkcd", zc, cum)
    ttt_out = ttt_out.reshape(B, n_chunks * C, -1)[:, :N, :]   # strip padding

    # 10. Session mode: stage the next carry
    if self.session_mode:
        total = deltas.sum(dim=1).detach().float()             # fp32 & detach
        self._next_carried = (
            total if self.carried_delta is None
            else self.cfg.carried_decay * self.carried_delta + total
        )

    # 11. Gate and combine
    return base_out + self._gated(ttt_out, hidden_states)
```

### Padding of the last chunk

When `N` isn't a multiple of `C`, the last chunk is zero-padded. The
key subtlety is **`normalize_delta_by_chunk`** — dividing by `C` would
underestimate `V^T Z` for the last chunk relative to full chunks, so
the actual non-padded token count is used. If we skipped this, the
last chunk would silently produce weaker updates.

### The `N <= C` fast path

If the sequence is short enough to fit in one chunk *and* there's no
carry *and* we're not in session mode, the update would be an unused
tail (exclusive cumsum → nothing applied to chunk 0). We return
`base_out` directly. Small optimization, mostly matters for eval on
very short inputs.

---

## The stream path

`_stream_forward` is the incremental path used during autoregressive
generation. The scan path assumes all `N` tokens are available at once;
generation feeds one token at a time (after prefill).

```python
@torch.no_grad()
def _stream_forward(self, z, hidden_states):
    # 1. Fetch left-context and compute V for these new positions.
    source = self._v_source(hidden_states)
    v = self._targets(source, left_context=self._v_left_context())
    self._update_hidden_context(source)  # buffer POST-norm source

    w0 = self.down_proj.weight
    st = self.state
    C = self.cfg.chunk_size
    outputs = []
    pos = 0
    N = z.shape[1]

    while pos < N:
        room = C - st.pending_tokens              # how much more fits before commit
        take = min(room, N - pos)
        z_part = z[:, pos:pos + take]

        # 2. Apply with current state (apply-then-update)
        out = z_part @ w0.T
        if st.delta is not None:
            ttt_term = self.cfg.eta * (
                z_part @ st.delta.to(z_part.dtype).transpose(-1, -2)
            )
            out = out + self._gated(ttt_term, hidden_states[:, pos:pos + take])
        outputs.append(out)

        # 3. Buffer this piece into pending; commit when full
        if self.ttt_evolve:
            st.pending_z.append(z_part)
            st.pending_v.append(v[:, pos:pos + take])
            st.pending_tokens += take
            if st.pending_tokens == C:
                self._commit_chunk()               # updates st.delta

        pos += take

    return torch.cat(outputs, dim=1)
```

`_commit_chunk` folds the buffered `Z` and `V` into `state.delta` in
fp32:

```python
def _commit_chunk(self):
    st = self.state
    zc = torch.cat(st.pending_z, dim=1)
    vc = torch.cat(st.pending_v, dim=1)
    delta = torch.einsum("bcd,bcf->bdf", vc.float(), zc.float())   # fp32
    if self.cfg.normalize_delta_by_chunk:
        delta = delta / self.cfg.chunk_size
    new = delta if st.delta is None else st.delta + delta
    st.delta = self._clip(new)                                     # bound magnitude
    st.pending_z.clear()
    st.pending_v.clear()
    st.pending_tokens = 0
```

Note the **apply-then-update** order (step 2 above): the current token
sees `state.delta` **before** it gets folded into it. That matches the
scan path's exclusive cumsum.

### Why the buffer exists

The stream sees tokens in whatever chunks `.generate` provides (usually
1 token at a time during autoregressive decoding, but the prefill
sends the whole prompt at once). We can only commit a chunk when we
have `C` fresh tokens' worth of `Z` and `V`. So `pending_z` /
`pending_v` accumulate until they hit `C`, then flush.

This means during generation the actual fast weight only updates every
`C` steps. The Z/V buffered before the next commit is applied on each
step against the current `state.delta`, but *itself* isn't committed
yet — it will be, once the buffer fills.

### Left context for the causal conv

Streaming mode adds one wrinkle to `_targets`: the conv needs the last
`K-1` positions of `X0` from the previous forward call to produce
causally-correct `V` for the current call's first `K-1` positions.
Without that context, the conv would treat position 0 of the current
call as position 0 of the sequence, dropping information from prior
positions.

The tap keeps a rolling buffer (`_rolling` for embedding source,
`_hidden_context` per-module for hidden-state source) of exactly
`K - 1` positions. On each stream forward, we prepend it to `X0` and
then slice the conv output back down to the current chunk's positions.

---

## Scan / stream equivalence

**Claim:** for the same token stream with chunk boundaries aligned, the
scan and stream paths produce bit-identical outputs.

Sketch:

- The scan computes `S_i = sum_{j<i} (V_jᵀ Z_j / C)` via exclusive
  cumsum. The stream computes `S_i` via `state.delta` populated by
  successive `_commit_chunk` calls at the end of chunks `0, 1, ..., i-1`.
  Same sum, same normalization, same order (fp32 in both cases).
- Application `O = Z (W_0 + eta·S)^T` is the same in both paths.
- The gate uses the same `hidden_states` slice in both paths.
- The Frobenius clip is applied position-wise in both (per-chunk-position
  in the scan; per-commit in the stream; equivalent when applied to the
  same cumulative state before use).

Verified numerically in [`test_scan_math.py`](../tests/test_scan_math.py)
across:

- Non-session mode (per-forward reset).
- Session mode with prior carry.
- With and without `normalize_delta_by_chunk`.
- Ragged last chunk (`N` not divisible by `C`).

If this test fails, either the scan is wrong or the stream is wrong;
they can't diverge without one being wrong.

---

## Session mode: TBPTT carry across items

Session mode extends the fast weight *across items* in a session, so
subsequent items get the previous items' `S` as their starting state.
This is what makes "the model remembers what it read" work.

### Lifecycle

```
For each session:
    reset_session_state(model)               # carried_delta = None on every layer
    for item in session:
        loss = model(input_ids=item).loss    # scan path stages _next_carried
        loss.backward()                      # gradient flows through S within item
        advance_session_state(model)         # carried_delta ← _next_carried
```

Two things change vs non-session mode:

1. In `_scan_forward`, when `session_mode=True` and `carried_delta` is
   not `None`, we set `S_0 = carried_delta` instead of `0`. This is
   what "the previous item's ending state is the next item's starting
   state" means.
2. At the end of the scan, we stage `_next_carried` based on the sum
   of this item's per-chunk deltas (EMA-mixed with the previous
   `carried_delta` when `carried_decay < 1.0`).

### Gradient does not cross the item boundary

`_next_carried` is `.detach().float()` — this is the TBPTT truncation.
Gradient through the fast weight only exists inside the current item's
forward / backward; the boundary between items is a stop-gradient.

Why? If gradient did cross, backward through item `i+1`'s forward
would need item `i`'s forward computation graph to still be alive.
Sessions can be dozens of items long. Storage would explode.

TBPTT (truncated backprop through time) is the standard solution. The
tradeoff: the model doesn't learn to *plan* multi-item optimization,
but it does learn to *use* whatever state it's given. The training
signal is "given `carried_delta` as-is, produce good outputs on the
current item." That's what we want for a memory mechanism.

### EMA staging (`carried_decay`)

```python
if self.session_mode:
    total = deltas.sum(dim=1).detach().float()             # per-item total delta
    self._next_carried = (
        total if self.carried_delta is None
        else self.cfg.carried_decay * self.carried_delta + total
    )
```

- `carried_decay = 1.0` (default): pure accumulation.
  `carried_delta` after `N` items grows as `sum` of per-item totals.
  With bounded per-item totals this is unbounded — long sessions cause
  `state_ratio` (state magnitude relative to `W_down`) to drift up.
- `carried_decay = 0.95`: half-life ~ 14 items. Bounded plateau ≈
  `per_item_total / (1 - decay) = 20 × per_item_total`.
- `carried_decay = 0.90`: half-life ~ 7 items. Plateau ≈
  `10 × per_item_total`.
- `carried_decay = 0.0`: last-item-only. No session memory.

The Frobenius clip catches this too — even with `decay = 1.0`, once
`||eta · S||_F > clip_tau`, `S` gets rescaled down. But relying on the
clip alone leaves `S` in a permanent "direction-only" regime; the
magnitude is capped, so the mechanism can't emit stronger corrections
even when the direction would benefit from it. `carried_decay < 1.0`
keeps magnitude in a *useful* regime instead of pegged at the clip
limit.

Verified in [`test_session.py`](../tests/test_session.py):
`carried_delta` after `N` items equals
`sum_{i} decay^(N-i-1) · per_item_delta_i`.

### `advance_session_state`

```python
def advance_session_state(model):
    for m in iter_ttt_modules(model):
        if m._next_carried is not None:
            m.carried_delta = m._next_carried
            m._next_carried = None
```

Called *after* backward, so the forward pass's staging is already
complete. Idempotent under gradient-checkpointing recompute — see the
next section.

### `reset_session_state`

Zeros `carried_delta` and `_next_carried` on every TTT module. Called
at session boundaries (start of each session in training, and after
each session in eval). Does **not** touch `state.delta`
(streaming state), which is independent.

---

## Idempotence under gradient checkpointing

Gradient checkpointing (`use_reentrant=False`) recomputes each
checkpointed forward *twice*: once "for real" during the forward pass,
once during backward to reconstruct activations for gradient
computation. If `_scan_forward` staged `_next_carried` naively on every
call, the second recompute would double-count the new item's delta into
`_next_carried` on top of the first computation's value.

We avoid this by making staging **idempotent**:

```python
if self.session_mode:
    total = deltas.sum(dim=1).detach().float()
    self._next_carried = (
        total if self.carried_delta is None
        else self.cfg.carried_decay * self.carried_delta + total
    )
```

- We do **not** accumulate into `_next_carried`. We **overwrite** it
  with the value derived from `carried_delta` (the current head) plus
  this item's total.
- `carried_delta` is not modified inside `_scan_forward` — it's read
  only. So repeated calls to `_scan_forward` for the same item produce
  the same `_next_carried` value.
- `advance_session_state` promotes `_next_carried → carried_delta` and
  clears `_next_carried`. It's called exactly once per item, *after*
  backward. Calling it twice in a row is also idempotent because after
  the first call `_next_carried is None`.

So the recompute during backward stages the same `_next_carried` as
the forward, and `advance_session_state` after backward does the
promotion once. Net effect: exactly one carry step per item.

Tested explicitly in
[`test_session.py::test_staging_is_idempotent_under_recompute`](../tests/test_session.py).

---

## Initialization and the dead basin

### The identity property

- `W_target = 0` → `V = conv(X0) @ 0 = 0` for all positions.
- Consequently `deltas = V^T @ Z = 0`, so `S = 0` throughout the
  forward.
- The apply step becomes `O = Z @ W_0^T + eta · Z @ 0^T = Z @ W_0^T`,
  which is the plain gated MLP output.
- The output gate multiplies `ttt_out` by `sigmoid(gate(H))`, but
  `ttt_out = 0` at init, so the gate value is irrelevant at step 0.

So at step 0, every TTT layer is **bit-exact** identical to the base
Qwen3 MLP. The `sanity_check` entrypoint in `train_modal.py` verifies
this by comparing logits with `W_target = 0` and confirming the max
|logit diff| is `< 1e-3`.

If sanity_check fails, some wiring is broken — probably the LoRA regex
is matching TTT-layer `down_proj`, or `W_target` isn't actually
zero-initialized, or the tap isn't providing the right source. Do not
train until sanity_check is clean.

### The dead basin

The downside of `W_target = 0`: gradient to `W_target` at step 0 is
proportional to the chain `dL/dO · dO/d(ttt_out) · d(ttt_out)/dS
· dS/dV · dV/dW_target`. The middle term `dS/dV = Z^T / C` is nonzero,
but the `dV/dW_target = conv(X0)^T` piece is fine — the real problem is
that `dO/d(ttt_out) = gate(H) · eta` is small (gate near 0.12 at init),
and `d(ttt_out)/dS = eta · Z^T` is proportional to `eta` (default
`7e-2`).

Net effect at step 0: gradient magnitude to `W_target` is
`O(gate · eta² · ||Z||² · ||conv(X0)||)`, i.e. `O(0.1 · 0.005 · 1000 ·
1000) ≈ O(500)` in an unscaled sense — enough to make progress at 0.6B
in 100–200 optimizer steps. At 4B, the same numerator holds but there
are ~5× more parameters in `W_target` (bigger `d`), so per-parameter
gradient is 5–10× smaller. See [scaling.md](scaling.md#1-per-parameter-gradient-dilution).

If the model doesn't escape the basin (grad/new stays flat, state_ratio
stays flat), symptoms are:

- `state_ratio_mean` never rises above `~1e-3`.
- `eval/gap` never becomes positive.
- Loss curve looks like plain LoRA training.

Fixes:

- Bump `lr_new_modules` (the LR on `W_target` and `target_conv`).
- Increase `output_gate_bias_init` (e.g. `-1.0` → sigmoid `0.27`) so
  the initial gate lets more `ttt_out` through, opening the gradient
  path wider.
- Small-random init `W_target` (breaks sanity_check bit-exactness but
  gives a nonzero gradient signal from step 0). Not the default because
  it forfeits the identity invariant.

See [failure-modes.md](failure-modes.md#dead-basin) for the detailed
diagnostic recipe.

### `target_conv` pass-through init

```python
self.target_conv.weight.zero_()
self.target_conv.weight[:, :, -1] = 1.0
```

At step 0, `v_raw(t) = x0(t) · 1.0`, i.e. current position, not a
temporal mix. `W_target = 0` makes this irrelevant to the identity, but
once `W_target` moves off zero, this init means the earliest gradient
signal to `target_conv` is "we currently pass through position `t`;
how would other positions in the kernel help?"

Alternatives (random init) would give the model a random 8-position
temporal filter to start from; the pass-through init is cleaner and
lets the model discover its own temporal filter.

---

## The Frobenius clip

```python
def _clip(self, delta):
    if not self.cfg.clip_enabled:
        return delta
    if self.cfg.clip_at_inference_only and self.training:
        return delta
    norm = (self.cfg.eta * delta).norm(p="fro", dim=(-2, -1), keepdim=True)
    scale = (self.cfg.clip_tau / norm.clamp_min(1e-12)).clamp(max=1.0)
    return delta * scale
```

**Where applied:**

- `_scan_forward`: applied to `cum` at every chunk position (via
  broadcast over `k`). Each chunk sees a clipped fast weight.
- `_stream_forward` via `_commit_chunk`: applied to the newly-updated
  `state.delta` right after the commit. Only the fast weight itself is
  clipped; the previous applies are already done.

**What it computes:**

- `||eta · delta||_F` is the Frobenius norm of the *scaled* delta —
  i.e. how big the TTT correction would be in units of `W_down`.
- If `> clip_tau`, we scale down until it equals `clip_tau`, preserving
  direction.
- If `<= clip_tau`, no-op.

**Why direction preservation.** The information the fast weight carries
is which rows of `W_down` to push in which direction. The magnitude is
"how much." Empirically the model calibrates its expectations to the
clip during training. If you train with `clip_tau = 5` and then serve
with `clip_tau = 30`, the model doesn't benefit — it never learned to
produce corrections useful at magnitude 30. (Verified empirically at
0.6B: clip=30 both trained and evaluated is worse than clip=5.)

Concretely this means: at scale, when `||eta · S||_F` grows past
`clip_tau`, we lose one degree of freedom (magnitude) but the direction
still communicates. In this "direction-only regime," most productive
training happens. This is not a bug; it's the operating point.

Related: `clip_at_inference_only=True` disables clip during training so
you can inspect unclipped state magnitudes. **Not recommended for
production runs** — the model then trains with a different clip regime
than it serves under.

---

## The output gate

Full details of the gate mechanism, expanding [the earlier component
section](#output_gate-and-gate-diagnostics):

### Math

```
gate(H) = sigmoid(W_g · H + b_g)                # b_g init -2.0, W_g std 1e-3
final(t) = base_out(t) + gate(t) · ttt_out(t)   # per position
```

`gate` is a scalar per position. `b_g = -2` gives an initial sigmoid
value of `~0.12`, so `12%` of the TTT contribution reaches the output
at step 0. This creates a gradient path even before `W_target` moves
off zero.

### Why per-position

The alternative would be a global gate (per-layer scalar) or a per-head
gate. Per-position is a reasonable middle ground: it can down-weight
specific positions (e.g. when `ttt_out` at that position looks harmful)
without global suppression. It also gets a per-position gradient signal
back to `output_gate.weight`.

### Regularization

`gate_reg_weight * mean(sigmoid(W_g H)^2)` optionally penalizes the
gate output. Default 0. Useful if the gate stays wide-open and you want
to encourage it to close (equivalent to encouraging sparser TTT use).

### Diagnostics

Stashed on `no_grad` at every forward:

```python
self._gate_mean = float(gate.mean().detach())
self._gate_std = float(gate.std().detach())
```

Read via `gate_stats(model)`. Alerts thresholds documented in
[observability.md](observability.md#gate-mean-per-layer).

---

## `v_source`: embedding vs hidden-state

Two options for what feeds `target_conv`:

### `v_source = "embedding"` (paper reference)

`X0` is the output of `embed_tokens` — the input embedding for each
token. A single tap attached to `embed_tokens` produces one X0, shared
by every TTT layer.

**Pro:** cheap. One tap, one buffer.

**Con:** every TTT layer sees the same `X0`. The mechanism can only
distinguish layers via the different `W_target`s and different `W_down`s
they see.

### `v_source = "hidden_state"` (default)

`X0` at each TTT layer is the layer's own input hidden state (the value
passed to `forward`). Different layers see different `X0`, because
earlier layers' outputs have been transformed by later attention +
MLP.

**Pro:** each layer's V can be much more expressive — deeper layers
build V from more-processed representations.

**Con:** each layer needs its own `_hidden_context` buffer for streaming
mode (the rolling `K-1`-position buffer for the causal conv). Slightly
more memory.

Empirically at 0.6B, `hidden_state` is 1–2 ppl better than `embedding`
on holdout eval. Default is `"hidden_state"`.

### RMSNorm on the V source

```python
if cfg.v_source == "hidden_state":
    self.v_source_norm = nn.RMSNorm(hidden_size)
    self.v_source_norm.weight.requires_grad_(False)   # gamma frozen at 1.0
```

Why norm? Deep layers' hidden states have larger norms than the shallow
ones' (residual stream grows). Without a norm, deeper TTT layers would
get systematically larger V, and eta would be effectively different
per-layer. With norm, all layers see V of comparable scale.

Why gamma frozen? If the norm's learned scale were trainable, the
optimizer could inflate it to counteract the clip on `eta · S`,
effectively defeating magnitude control. Frozen gamma = 1.0 makes the
norm a hard rescale.

---

## The tap

`EmbeddingTap` is a small helper object that provides `X0` on demand
during forward:

```python
class EmbeddingTap:
    def __init__(self, conv_kernel_size):
        self.context_len = conv_kernel_size - 1
        self.current = None                       # last forward's X0
        self.prev_context = None                  # rolling last K-1 for stream
        self._rolling = None
        self.stateful = False
```

For `v_source="embedding"`, an actual forward hook is attached to
`embed_tokens` that populates `self.current` on every forward. For
`v_source="hidden_state"`, there's no shared tap — each TTT layer
maintains its own `_hidden_context` — but the `tap` object still
exists (for the shared `stateful` flag).

`stateful=True` (streaming mode) enables the rolling buffer for the
conv's left context. `stateful=False` (scan mode) leaves the tap
buffer alone.

At turn boundaries in chat, `reset_v_left_context()` clears the
rolling buffers so the next turn's causal conv starts with zero left
context. `state.delta` (the actual fast weight) survives across turns.

---

## RMSNorm on the V source

Covered above under `v_source`. One thing to add: the norm is applied
**before** `target_conv`, so the conv sees rescaled input. This means
V has scale ~1.0 regardless of layer depth. eta then controls the
per-layer contribution uniformly.

---

## fp32 boundaries

Fast weights are stored / accumulated in fp32 even though the model
otherwise runs in bf16:

- `carried_delta`: fp32. Stored on each TTT module in session mode.
- `_next_carried`: fp32. Staged during forward.
- `state.delta`: fp32. Populated by `_commit_chunk` in the stream.
- `_commit_chunk`: casts `Z`, `V` to fp32 *before* the outer product,
  so the accumulation is fp32.
- The scan path: `deltas.sum(dim=1).detach().float()` stages `total` in
  fp32.

Cast to bf16 happens **only at apply time**:

```python
carried = self.carried_delta.to(z.dtype)      # bf16 for apply
```

Why not fp16 or bf16 throughout? Bf16 has 8-bit mantissa (~2-3
significant decimal digits). Accumulating dozens of outer products of
bf16 quantities produces error that grows with the number of chunks. At
chunk_size 50 and sequence length 16384, k = 327 chunks per forward; at
6 items per session, we sum thousands of chunk deltas. Fp32 (~7
decimal digits) has enough headroom; bf16 would drift.

Cost: `carried_delta` at `d = 1024, d_ff = 3072` is
`1024 × 3072 × 4 bytes = 12 MB` per TTT layer. With 14 layers at
0.6B, that's ~170 MB — negligible next to model weights and activations.

---

## The batch-size-1 invariant

Every carry / state tensor has a `B` dimension. If `B` changes
mid-session, the state from the prior `B` no longer matches. The code
enforces this explicitly:

```python
if self.session_mode and self.carried_delta is not None:
    if self.carried_delta.shape[0] != B:
        raise RuntimeError(
            "Batch size changed mid-session; keep B constant "
            f"(state B={self.carried_delta.shape[0]}, input B={B})."
        )
```

In practice, `B = 1` always in this codebase (`micro_batch_size=1`
enforced in training, and inference is single-stream). Batching would
mean either:

- Batching sequences from independent sessions: each needs its own
  carry, so we'd need `B` independent carries per layer. Currently a
  single `carried_delta[B, d, d_ff]` per module, so this is technically
  possible if all sequences enter at the same session start — but
  breaking mid-session is not supported.
- Batching sequences from the *same* session: this would mean the same
  carry serves multiple sequences, which is a semantic ambiguity we
  chose to reject.

**If you want higher throughput at inference,** the right move is more
`TTTInference` replicas (each with `B=1`) rather than `B > 1` in one
replica. Modal will auto-scale replicas on load.

---

## Where gradient goes

Three sinks for the outer-optimizer gradient:

### `lora` (LoRA A + B on attention + gate/up + non-TTT down_proj)

LR ≈ `1e-5`. These are the transformer's own adaptations to the
domain. Gradient signal flows through the usual paths (attention
outputs, MLP gate/up).

`down_proj` on TTT layers is **excluded** from LoRA (enforced by the
regex). This preserves the fast-weight-initial-state interpretation of
`W_down`.

### `wdown` (`down_proj.weight` on TTT layers)

LR ≈ `3e-5`. The pretrained fast-weight initial state. Moves gently
because it starts from a good place — we don't want to overwrite what
Qwen3 already learned.

Gradient reaches `W_down` via `base_out = Z @ W_down^T` at every scan
step. The gradient path is basically the same as normal MLP training,
just applied to a subset of layers.

`W_down`'s functional use (via `.weight`) skips the LoRA path — good,
because we don't want LoRA-through-`W_down` mixing with the
fast-weight interpretation.

### `new` (`W_target` + `target_conv` per TTT layer)

LR ≈ `2e-5`. Fresh modules with no pretrained prior.

Gradient path: `dL/dS` (through the apply step in the next chunk) →
`dS/dV` (chain rule through the outer product) → `dV/dW_target` and
`dV/d(target_conv)`. The `dL/dS` piece is what makes the fast weight
"useful" — the loss reduces when `S` is well-chosen, and gradient flows
back to `W_target` accordingly.

The `output_gate.weight` and `bias` are also in this group.

### Grad group ratios at 0.6B (empirical)

Roughly:

- `lora` ≈ `5e-4` per parameter per step.
- `wdown` ≈ `4e-4` per parameter per step.
- `new` ≈ `1e-4` per parameter per step.

`new` is smaller because gradient is a chain product; more mediating
layers between loss and parameter. At 4B, all three drop ~6–11×; see
[scaling.md](scaling.md#1-per-parameter-gradient-dilution).

### What is NOT trained

- Base Qwen3 weights (embedding, attention, gate/up on TTT layers,
  `down_proj` on non-TTT layers, LayerNorm scales, LM head). Only LoRA
  adapters touch these.
- `v_source_norm.weight` (gamma of the pre-conv RMSNorm) — frozen at
  1.0 by design.
- The tap has no parameters.

---

## Public API surface

Everything importable from [`inplace_ttt.py`](../inplace_ttt.py):

### Module-level classes

- `InPlaceTTTMLP` — the TTT layer itself. Replaces `Qwen3MLP` on TTT
  layers.
- `EmbeddingTap` — shared X0 provider for `v_source="embedding"`.
- `TTTState` — dataclass holding `state.delta`, `pending_z`,
  `pending_v`, `pending_tokens` for one module's streaming state.

### Assembly

- `patch_model_with_ttt(model, cfg)` — replaces `mlp` on every layer in
  `cfg.layer_indices` with an `InPlaceTTTMLP`. Attaches the tap.

### Iteration

- `iter_ttt_modules(model)` — yields every `InPlaceTTTMLP` inside a
  model.

### Reset ladder

- `reset_fast_weights(model)` — full reset. Clears `state.delta`,
  `pending_*`, and the conv left context on every TTT module + the
  embedding tap. Used at eval boundaries and inside `chat_reset`.
- `reset_v_left_context(model)` — soft turn boundary. Clears the conv
  left context only. `state.delta` and pending survive. Used between
  chat turns.
- `reset_session_state(model)` — TBPTT boundary. Clears `carried_delta`
  and `_next_carried` on every TTT module. Independent of the streaming
  state.

### Session mode

- `advance_session_state(model)` — promote `_next_carried →
  carried_delta` on every TTT module. Idempotent. Call **once per
  item, after backward**.

### Diagnostics

- `state_norms(model, source="session"|"stream")` — per-layer
  `||eta · delta||_F / ||W_down||_F`. `source="session"` reads
  `carried_delta`; `source="stream"` reads `state.delta`.
- `mean_state_ratio(norms)` — mean of `state_norms` values.
- `gate_reg_term(model)` — sum of stashed gate L2 penalties across
  modules (tensor). Always returns a tensor, even when nothing stashed.
- `gate_stats(model)` — per-TTT-layer sigmoid gate mean and std.
- `stream_pending_progress(model)` — `(pending_tokens, chunk_size)` for
  one TTT layer. All layers see the same stream, so one is
  representative.

### Snapshot / restore (streaming state)

- `export_fast_weights(model)` — `dict[int, Tensor]` snapshot of every
  TTT layer's `state.delta`. Used by `save_session` in chat.
- `import_fast_weights(model, snapshot)` — restore a snapshot. Used by
  `chat_reset(from_snapshot_name=...)`.

Nothing else in the mechanism module is public. The individual layer's
`forward`, `_scan_forward`, `_stream_forward`, `_commit_chunk`,
`_clip`, `_gated`, `_targets`, `_v_source`, `_v_left_context`,
`_update_hidden_context` are called only by each other — never by
outside code.

---

## Related docs

- [architecture.md](architecture.md) — how the mechanism fits into the whole system
- [config.md](config.md) — every knob and its when-to-tune
- [training.md](training.md) — the outer loop that trains the mechanism
- [inference.md](inference.md) — three-way eval, holdout, chat
- [failure-modes.md](failure-modes.md) — dead basin, saturation, gradient collapse
- [scaling.md](scaling.md) — how the mechanism behaves at 1.7B, 4B, 8B
- [testing.md](testing.md) — which properties of the mechanism are unit-tested
