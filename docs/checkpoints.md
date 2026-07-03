# Checkpoints and Snapshots

Storage layout, save/load semantics, and compatibility rules.

## Layout

```
ttt-checkpoints Modal volume
├── <run_name>/                       (TrainConfig.run_name, default "ttt-v1.1")
│   ├── step_200/
│   │   ├── adapter/                  PEFT LoRA adapter (via save_pretrained)
│   │   │   ├── adapter_config.json
│   │   │   └── adapter_model.safetensors
│   │   └── ttt_params.pt             torch.save dict — TTT trainables
│   ├── step_400/ ...
│   └── sessions/
│       └── <name>.pt                 exported fast-weight snapshots
├── <other_run>/ ...
└── loss_mask/
    └── reference_wikitext103.pt      unigram counts for loss mask
```

`<CKPT_MOUNT>` is `/ckpt` inside containers.

## What's in each artifact

### `adapter/` — LoRA weights

PEFT `save_pretrained` output. Loadable with
`PeftModel.from_pretrained(base, path, is_trainable=trainable)`.
Includes:
- `adapter_config.json` — rank, alpha, dropout, target_modules regex
- `adapter_model.safetensors` — LoRA A + B matrices for every
  targeted module

**Target modules** (from `build_lora_target_regex()` in
[`ttt_wiring.py`](../ttt_wiring.py)):
```
.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj)
| .*\.layers\.(<non-TTT layer indices>)\.mlp\.down_proj
```

LoRA never touches `down_proj` on TTT layers — that weight is used
functionally as `W0`.

### `ttt_params.pt` — TTT trainables

Dict from `save_ttt_state_dict()` in
[`ttt_wiring.py`](../ttt_wiring.py). Contains, per parameter:
- Key: PEFT prefix stripped, so keys are base-model relative
- Value: CPU tensor

Included:
- `model.layers.<i>.mlp.down_proj.weight` for each TTT layer index
- `model.layers.<i>.mlp.target_conv.weight` for each TTT layer
- `model.layers.<i>.mlp.w_target` for each TTT layer
- `model.layers.<i>.mlp.output_gate.weight` and `.bias` for each TTT layer (if gate enabled)

The number of tensors saved depends on `len(TTT_CFG.layer_indices)`:
- With `output_gate=True`: `n_ttt_layers × 5` tensors
  (down_proj, target_conv, w_target, gate weight, gate bias)
- With `output_gate=False`: `n_ttt_layers × 3` tensors

**Mismatch behavior:** `load_ttt_state_dict()` raises
`RuntimeError("TTT checkpoint mismatch, saved N tensors, loaded M")`
if the number of matched keys differs from the number saved. This
catches:
- Different `TTT_LAYER_STRIDE`/`LAYER_START` between save and load
- Different `MODEL_SIZE` (different `num_hidden_layers`)
- Config changes that removed/added a TTT trainable (e.g., toggling
  `output_gate`)

### `sessions/<name>.pt` — fast-weight snapshot

Dict from `export_fast_weights()` in
[`inplace_ttt.py`](../inplace_ttt.py):
```python
{layer_idx: fp32 delta tensor on CPU}
```

`layer_idx` is the enumeration order (`0..n_ttt_layers-1`), NOT the
base-model layer index. Layers with `state.delta is None` are omitted
from the snapshot.

### `loss_mask/reference_wikitext103.pt` — unigram reference

From `build_reference_counts` in
[`train_modal.py`](../train_modal.py):
```python
{
    "counts": tensor[vocab_size],
    "tokenizer_name": "Qwen/Qwen3-...",
    "dataset_id": "Salesforce/wikitext",
    "dataset_config": "wikitext-103-raw-v1",
    "split": "train",
    "n_docs": int,
    "n_tokens": int,
    "n_unique": int,
    "vocab_size": int,     # = model.config.vocab_size (padded)
}
```

## Save/load code paths

### Training-side save (`train_modal.py`)

```python
def save_checkpoint(model, run_dir, step):
    save_dir = os.path.join(run_dir, f"step_{step}")
    os.makedirs(save_dir, exist_ok=True)
    # LoRA adapter
    model.save_pretrained(os.path.join(save_dir, "adapter"))
    # TTT trainables
    save_ttt_state_dict(model, os.path.join(save_dir, "ttt_params.pt"), TTT_CFG)
    ckpt_vol.commit()   # push to Modal volume
```

Called every `save_every` optimizer steps + once at the end.

### Inference-side load (`model_setup.py`)

```python
patch_model_with_ttt(model, TTT_CFG)     # replace TTT layers, add tap

if adapter_path:
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
else:
    model = get_peft_model(model, build_lora_config(...))   # fresh LoRA at init

if trainable:
    unfreeze_ttt_params(model, TTT_CFG)

if ttt_ckpt_path and os.path.exists(ttt_ckpt_path):
    load_ttt_state_dict(model, ttt_ckpt_path)
```

**Order matters:**
1. TTT patch before LoRA (so LoRA wraps the correct `gate_proj`/`up_proj`)
2. LoRA before `unfreeze_ttt_params` (LoRA freezes non-LoRA, we
   re-enable)
3. TTT state dict load LAST (values overwrite whatever the patch/LoRA
   set)

## Compatibility rules

### Snapshot ↔ slow-weights coupling

A fast-weight snapshot is **only valid for the exact slow weights it
was created under**. `state.delta` is a delta on `W_down` — if
`W_down` has moved (further training), the delta applies to a `W0`
that no longer exists in the current model.

There's no explicit check for this — the snapshot loads silently.
Result: behavior is silently wrong (undefined direction, likely
degraded).

**Rule of thumb:** save snapshots per slow-weight checkpoint. Don't
mix snapshots from step_200 with slow weights from step_400.

### TTT config ↔ ckpt coupling

`ttt_params.pt` is coupled to:
- `TTT_MODEL_SIZE` (determines `num_hidden_layers`)
- `TTT_LAYER_STRIDE` and `TTT_LAYER_START` (determine which layers get TTT patched)
- `output_gate` setting (adds/removes gate params in the save dict)

If any of these change between save and load, the count mismatch guard
in `load_ttt_state_dict` triggers.

**Migration:** if you change layer stride, the old ckpt is no longer
loadable. Options:
- Match the env vars at inference time (set them to what was used at
  training)
- Retrain from scratch with the new stride

### Reference counts ↔ tokenizer coupling

`reference_wikitext103.pt` is coupled to the tokenizer's
`vocab_size`. `load_reference_counts(path, expected_vocab_size)`
raises on mismatch. Rebuild with `build_reference_counts` after any
`BASE_MODEL` or `TTT_MODEL_SIZE` change (they may use different
tokenizers).

### Adapter ↔ model coupling

PEFT tolerates the adapter being loaded onto a compatible base model
regardless of layer stride etc. (it targets modules by regex). But
the LoRA regex was built with the SAVE-time layer stride — the
loaded adapter will silently not touch modules that don't match the
save-time regex.

If layer stride changes, the LoRA regex changes, so some layers that
should have LoRA won't (and vice versa). The fresh side (no ckpt)
builds a new regex from current config; the loaded side uses the
saved regex.

**Bottom line:** treat every ckpt as coupled to its `ttt_config.py`
snapshot. Note the env vars used at training time.

## Ckpt name resolution

At inference, `_ckpt_paths(ckpt)` in
[`infer_modal.py`](../infer_modal.py):

| input | resolves to |
|---|---|
| `""` | `(None, None)` — base model, no LoRA loaded from disk, no TTT weights |
| `"step_400"` | `<CKPT_MOUNT>/<run_name>/step_400/adapter`, `<CKPT_MOUNT>/<run_name>/step_400/ttt_params.pt` |
| `"other_run/step_400"` | `<CKPT_MOUNT>/other_run/step_400/adapter`, `<CKPT_MOUNT>/other_run/step_400/ttt_params.pt` |

`<run_name>` is `TRAIN_CFG.run_name` (default `"ttt-v1.1"`).

## Related docs

- [training.md](training.md) — save cadence
- [inference.md](inference.md) — load path
- [chat.md](chat.md) — snapshot lifecycle
