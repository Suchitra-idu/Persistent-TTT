# Development Guide

Conventions and safety checks for changing this codebase.

## Before you change anything

1. **Read the module docstring** of the file you're editing. Every
   module has one; they call out the traps.
2. **Run the tests locally** to make sure you're on a green baseline.
   ```
   python -m pytest tests/ -q
   ```
3. **Read the relevant doc** in `docs/` — this folder is the design
   record.

## Pre-commit checklist

For any change touching mechanism / training / inference code:

- [ ] Tests pass locally: `python -m pytest tests/ -q`
- [ ] `modal run train_modal.py::sanity_check` prints max logit diff ≈ 0
  and passes the assert (verifies bit-exact identity at step 0)
- [ ] If you added a config field: update [config.md](config.md)
- [ ] If you added a metric: update [observability.md](observability.md)
- [ ] If you changed a wiring rule: verify the LoRA target regex is
  still consistent with the TTT layer set

For any change to the README or docs:

- [ ] All copy-pasteable commands actually work
- [ ] Config values referenced match `ttt_config.py`
- [ ] Cross-references between docs are still valid

## What to do vs not to do

**Do:**
- Keep `inplace_ttt.py` and `ttt_wiring.py` free of Modal imports.
- Route new config through `TTTConfig` or `TrainConfig` in
  `ttt_config.py`, not module-level module constants.
- Add tests for any new pure function.
- Prefer editing existing files over creating new ones.

**Don't:**
- Move the Python modules into a subpackage or `src/` directory —
  Modal ships them by name from the working directory.
- Add fast-weight state that isn't reset by
  `reset_fast_weights` / `reset_session_state`. Silent leaks between
  runs are the worst class of bug in this codebase.
- Add LoRA targeting to `down_proj` on TTT layers. The regex enforces
  this, but any change to the regex needs to preserve the invariant.
- Change init of `W_target` or `target_conv` without also updating
  `sanity_check` — bit-exact identity is a load-bearing property for
  debugging.

## Common change patterns

### Adding a new mechanism knob (e.g., a new gate variant)

1. Add the field to `TTTConfig` in `ttt_config.py` with a docstring
   and sensible default. Document in [config.md](config.md).
2. Read it in `inplace_ttt.py` where relevant. Default value should
   reproduce current behavior (backward-compat).
3. Add a test in `test_mechanism.py` that exercises the new path.
4. Add a metric to `observability.py` if the new path has its own
   failure mode.
5. Add an entry to the [failure-modes.md](failure-modes.md) list.

### Adding a new training-loop knob

1. Add the field to `TrainConfig` in `ttt_config.py`.
2. Consume it in `train_modal.py`. Default value should reproduce
   current behavior.
3. If it's user-facing at CLI level, add a flag to `train()` and to
   `_apply_cli_overrides`.
4. Document in [config.md](config.md) + [training.md](training.md).

### Adding a new inference entrypoint

1. Add the `@modal.method()` in `TTTInference` if it needs GPU state.
2. Add a `@app.local_entrypoint()` at the bottom of `infer_modal.py`
   that calls into the class.
3. Document in [inference.md](inference.md).
4. If it's a variant of `holdout_eval` / `single_paper_eval` and
   benefits from the three-way (BASE / LORA-ONLY / FULL) comparison,
   use `_three_way_eval` helper.

### Renaming a config field

Search-and-replace across the whole repo. This will catch:
- `ttt_config.py` (the definition)
- `train_modal.py` and `infer_modal.py` (consumers)
- Tests referencing the field
- `docs/` (documentation)

The `run_name` field is the ckpt directory name — renaming it will
orphan existing checkpoints unless you also rename the directory on
the Modal volume.

## Testing new code

**Unit test what can be unit-tested.** Pure functions belong in the
`tests/` suite. Guidelines:

- Use the fixtures in `conftest.py` for tiny TTT modules.
- Keep tests under 1 second each (the whole suite runs in ~3s).
- Use `torch.manual_seed(42)` for anything stochastic.

**GPU-required behavior goes in `sanity_check`.** The
[`train_modal.py::sanity_check`](../train_modal.py) function runs on
Modal and verifies mechanism identity properties on the actual model.
If you added a new invariant that only manifests on the real model,
add a check there.

## Reviewing PRs (checklist)

- [ ] Tests pass in CI (or locally if you don't have CI yet)
- [ ] Modal-side change: `sanity_check` output attached to the PR
- [ ] Config change: `docs/config.md` updated with the new field
- [ ] Behavior change: appropriate doc updated
- [ ] Comments explain WHY, not WHAT (the code already says what)
- [ ] No new module-level state that could leak between sessions
- [ ] No new module-level imports that break the Modal-free property
  of `inplace_ttt.py` / `ttt_wiring.py`

## Design principles (project-specific)

1. **Fail loudly at load time, not silently at inference.**
   Vocab-size mismatch on the loss-mask reference, ckpt tensor count
   mismatch — these raise `RuntimeError` at load, not during a run.
2. **One canonical assembly.** `build_model()` in
   [`model_setup.py`](../model_setup.py) is the single way to build
   the model. Train and inference must never diverge.
3. **Session state is per-stream.** Batch size > 1 is rejected. If
   you're tempted to add a multi-stream batch, first read
   [mechanism.md](mechanism.md) — the fast weights are inherently
   per-stream.
4. **Sanity check is a load-bearing property.** Zero-W_target
   + LoRA-B=0 → bit-exact base Qwen3 is what distinguishes "TTT off"
   from "TTT never patched." Don't break this without a very good
   reason.

## Related docs

- [architecture.md](architecture.md) — module boundaries
- [testing.md](testing.md) — what each test covers
- [config.md](config.md) — where config lives
