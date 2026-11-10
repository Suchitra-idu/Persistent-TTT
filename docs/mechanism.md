# The method, in code

This page follows the paper's Method section and points at the code that
implements each part. The paper is `Paper_TMLR/main.tex`; the code paths below
are relative to the repository root.

## Where the layer sits

`InPlaceTTTMLP` ([`ttt/extensions/mechanism/inplace_ttt.py`](../ttt/extensions/mechanism/inplace_ttt.py))
replaces the MLP of every TTT layer of Qwen3-0.6B. The layers are the 14
odd-numbered ones, 1, 3, ..., 27, derived from the model depth by
`derive_layer_indices` in [`ttt/core/config/ttt.py`](../ttt/core/config/ttt.py)
(stride 2, start 1). `patch_model_with_ttt` in
[`ttt/adapters/model_builder.py`](../ttt/adapters/model_builder.py) does the
replacement, then LoRA is added on top.

The layer keeps the pretrained `gate_proj`, `up_proj` and `down_proj`. With
`z = act(gate_proj(h)) * up_proj(h)` the gated MLP activation and `h` the MLP
input:

```
out = W_down z  +  g(h) * eta * S z                       (In-Place TTT term)
              +  alpha * g_c(h) * eta * clip(S_carried) z  (gated carry only)
```

## The fast weight inside one item

An *item* is one slice of a document, or a whole short document. Inside it the
fast weight `S` is written chunk by chunk, `C = 50` tokens per chunk.

| Paper | Code |
|---|---|
| target `V_k`: causal depthwise convolution (kernel 8) over the RMS-normalised hidden states, then a zero-initialised projection `W_target` | `InPlaceTTTMLP._targets`, `target_conv`, `w_target`, `v_source_norm` |
| Hebbian write `ΔS_k = (1/C) V_kᵀ Z_k` | `ttt_math.chunk_deltas` ([`ttt/core/ttt_math.py`](../ttt/core/ttt_math.py)) |
| chunk `k` reads only earlier chunks | `ttt_math.exclusive_cumsum` |
| clip `‖η S‖_F ≤ 20`, applied when the state is read, never fed back | `ttt_math.frobenius_clip` inside `ttt_math.scan` |
| `η S z` added to the output | `ttt_math.apply_state` |
| chunked delta rule, error from the chunk-start state | `ttt_math.chunked_delta_scan` (`update_rule=delta_chunk`) |

Because `W_target` starts at zero, every target is zero at step 0, so the layer
is exactly the base MLP before training. The main gate `g(h) = σ(w_gᵀh + b_g)`
is `output_gate`; its weights start at zero and its bias at
`output_gate_bias_init`, so it starts about `σ(b_g)` open. `_gated` applies it
and records its mean and spread for logging.

The chunked delta rule uses a normalised step `η / (1 + ‖z‖²)` for its
prediction (`_adaptive_eta`), feeds the clip back into the running state, and
detaches the state every `truncate_every` chunks. Only the ungated delta-rule
run uses it.

A single chunk with nothing carried in contributes nothing, because chunk 0
reads an empty state. `_scan_forward` returns the base output directly in that
case, unless a session is running, since the item's write is still needed for
the next item.

## The carry between items

At the end of each item the layer adds the item's total update to a carried
state, with earlier states decaying by 0.9:

```
S_carried <- 0.9 * S_carried + ΔS_item
```

This is `carry.advance` in [`ttt/core/carry.py`](../ttt/core/carry.py), called
from `_scan_forward`. The new state is *staged* in forward
(`_next_carried`) and *promoted* by `advance_carry()` after backward, so a
gradient-checkpointing recompute cannot write it twice. It is kept in float32.

Two ways to read the carried state, set by `carry_gate`:

- **Ungated carry** (`carry_gate=False`): `S_carried` is added to the state the
  scan starts from, so it is read through the same gate and clip as the
  in-slice fast weight. This is In-Place TTT with the reset removed.
- **Gated carry** (`carry_gate=True`): `S_carried` never enters the scan. It is
  read by `_carried_term`: its own clip (`‖η clip(S)‖_F ≤ 20`), its own gate
  `g_c = output_gate_carry`, and a scale `alpha = output_gate_carry_scale`
  initialised to zero, so the carried term is exactly zero at the start.

**Training through the carry** (`carry_backprop=True`) keeps `S_carried` in the
autograd graph, so the loss on a later item also trains what an earlier item
wrote. The training loop cuts the graph every `carry_window = 2` items and at
every optimizer step (`detach_carry`). `RecomputeAsForward` makes gradient
checkpointing replay a layer with the carry it actually read.

The paper's gated carry is therefore:

```
carry_gate=true, carry_backprop=true, output_gate_bias_init=-6
```

The config defaults (`carry_gate=False`, `carry_backprop=False`, bias `-2`)
are the ungated carry, because the recorded ungated runs relied on them.

## Sessions and the reset

A *session* is the list of items whose carry is kept. The carry is reset at
every session boundary (`fast_weights.reset_carry()`). The strategy decides
what a session is ([`ttt/extensions/strategies/`](../ttt/extensions/strategies/)):

- `hybrid` (the default, used by every within-document model): one document per
  session. A document shorter than 2100 tokens is one item; a longer one is cut
  into 2 to 6 slices of at least 1000 tokens. So the carry is trained between
  slices of one document.
- `mixed` (the probe trained across documents): sessions of 5 documents, each
  next document from the same source with probability 0.5, so the carry is
  also trained across document boundaries.

In-Place TTT's own behaviour, a reset at every document boundary, is what every
evaluation compares the carry against.

## The switches an evaluation uses

`FastWeights.set_mode(evolve=..., session=...)` controls the layer:

| Setting | Effect |
|---|---|
| `evolve=False` | the layer is `down_proj(z)`: trained weights, no fast weight |
| `evolve=True, session=False` | fast weights inside an item, no carry |
| `evolve=True, session=True` | fast weights and the carry |
| `pretrained_mlps(model)` | the untouched pretrained `W_down`, no fast weight (the base model) |

The parameter-matched control is trained with `fast_weight_updates=False`, which
trains the same architecture with `evolve=False`: its TTT modules are never used,
so only LoRA and the down-projections are trained.

## What is trained

`TorchCompute` ([`ttt/adapters/torch_compute.py`](../ttt/adapters/torch_compute.py))
puts every trainable parameter into one of three groups, by name
(`naming.classify_param` in [`ttt/core/naming.py`](../ttt/core/naming.py)):

| Group | Parameters | Learning rate |
|---|---|---|
| `lora` | LoRA adapters, rank 16, on attention, `gate_proj` and `up_proj` everywhere, and on `down_proj` of the non-TTT layers only (`naming.lora_target_regex`) | `1e-5` |
| `wdown` | the down-projections of the TTT layers | `3e-5` |
| `new` | `target_conv`, `w_target`, the gates, `alpha` | `6e-2` |

An unclassified trainable parameter is an error, never a silently untrained one.
