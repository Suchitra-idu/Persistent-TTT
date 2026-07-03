# Chat REPL

Interactive chat with fast-weight memory as the sole cross-turn
channel. Split into a Modal-hosted `TTTInference` class + a local
Python REPL client.

## Why two processes

`modal run` doesn't forward stdin to `local_entrypoint` subprocesses,
so the interactive loop can't live inside `infer_modal.py`.
[`chat_client.py`](../chat_client.py) is a regular local Python process
that calls into the deployed class via
`modal.Cls.from_name("inplace-ttt-infer", "TTTInference")`.

## Deploy + connect

```
modal deploy infer_modal.py
python chat_client.py --ckpt step_400
```

The class stays warm on Modal until the scaledown window
(`scaledown_window=300` seconds). Between conversations, the fast-
weight state persists inside the container. When it scales down and
a new container spins up, the fast weights start fresh (unless resumed
from a saved snapshot).

## Client flags

```
python chat_client.py [OPTIONS]

--ckpt CKPT              Checkpoint (empty for base Qwen3)
--evolve / --no-evolve   Fast-weight evolution (default: on)
--enable-thinking        Qwen3 thinking mode (default: off — see below)
--system TEXT            Static system prompt (kept, not conversation)
--from-snapshot NAME     Resume from a saved fast-weight snapshot
--max-new-tokens N       Per-turn generation cap (default: 512)
--temperature FLOAT      Default 0.6 thinking / 0.7 non-thinking
--top-p FLOAT            Default 0.95 thinking / 0.8 non-thinking
--top-k INT              Default 20
--debug                  Print raw sampled tokens + stop reason + thinking trace
```

Sampling defaults follow the Qwen3-8B model card. Do NOT pass
`--temperature 0` — the Qwen team explicitly warns against greedy
decoding (endless repetition).

## REPL commands

| command | effect |
|---|---|
| `/m` | Multiline/paste mode; ends on a blank line. Joins pasted lines as one message. |
| `/save NAME` | Persist current fast weights to `<run_name>/sessions/NAME.pt`. |
| `/reset` | Drop fast-weight state and start over with the same model. |
| `/quit` | Exit. |

### The `/m` command exists because…

Pasting multi-line text into a plain `input()` loop causes each line
to fire a separate `input()` call → each becomes its own turn (usually
fragmentary → nonsense responses). `/m` collects lines until a blank
one, then sends the joined block as a single turn.

## Cross-turn memory invariant

Each turn the model sees ONLY:
- Optional static system prompt (an instruction, not conversation)
- The current turn's user message, formatted via Qwen3's chat template

Prior user/assistant turns are **NOT** re-fed as context.
`past_key_values` from earlier turns is dropped at the turn boundary.
The within-turn KV cache is used for O(N) sampling and discarded.

The TTT fast-weight state (`state.delta` + the buffered partial chunk)
is the **sole cross-turn memory channel**. Re-feeding conversation
history would silently turn this into a test of context-window memory,
not the mechanism.

## What happens per turn (`chat_turn` internals)

1. **Mode setup** (idempotent per turn — NO fast-weight reset):
   ```
   set_ttt_stateful(model, True)
   set_ttt_evolve(model, evolve)
   ```

2. **Format the prompt** via Qwen3 chat template. Includes optional
   system message + current user message. No prior turns.

3. **Prefill** with `past_key_values=None` (attention must not see any
   prior turn):
   ```
   out = model(input_ids=new_ids, use_cache=True)
   past_kv = out.past_key_values      # within-turn cache only
   next_logits = out.logits[:, -1, :]
   ```

4. **Sample loop** for up to `max_new_tokens`:
   ```
   next_id = sample_top_p(next_logits, T, top_p, top_k)
   if next_id in chat_stop_ids: break
   out = model(input_ids=[[next_id]], past_key_values=past_kv, use_cache=True)
   past_kv = out.past_key_values
   ```
   Each `model()` call flows through `_stream_forward` on TTT layers,
   which:
   - Applies current `state.delta` to the token's Z
   - Buffers into `pending_z`, `pending_v`
   - When `pending_tokens == chunk_size`, calls `_commit_chunk()`
     which updates `state.delta`

5. **Turn boundary cleanup:**
   ```
   reset_v_left_context(model)   # NOT reset_fast_weights!
   ```
   Clears the tap's rolling buffer (so next turn's causal conv doesn't
   see prior-turn embeddings) but keeps `state.delta` and
   `pending_*` (the actual cross-turn memory).

6. **Decode + strip specials + split thinking**:
   ```
   raw = tokenizer.decode(generated, skip_special_tokens=False)
   cleaned = strip_chat_specials(raw)
   thinking_text, answer = split_thinking(cleaned)
   ```

7. **Diagnostic snapshot:**
   ```
   norms = state_norms(model, source="stream")
   state_ratio = mean_state_ratio(norms)
   pending, chunk = stream_pending_progress(model)
   ```

## What the REPL prints per turn

```
bot> <answer text>
  [state_ratio=X.XXe-XX  pending=NN/CHUNK]
```

- `state_ratio` grows turn-to-turn if fast weights are accumulating.
- `pending` climbs toward `chunk_size` between commits; each commit
  drops it back to 0 and increases `state.delta`.

Both zero after N turns of `--evolve` → carry is not engaging. Either
`evolve=False` or a bug.

## Thinking mode

Qwen3 supports a `<think>...</think>` mode where the model emits an
internal thinking trace before its answer. Default is **OFF** here
because our LoRA/TTT was trained on raw arxiv papers, never on
`<think>` traces — thinking tokens would be out-of-distribution for
both the LoRA and the carry.

Pass `--enable-thinking` to opt in, at the cost of OOD-ness. When on,
the client shows the thinking trace in `--debug` mode.

## Chat is fundamentally OOD

LoRA + TTT trained on **raw paper text**, not on question-answer or
conversation. Consequences:

- Responses tend to be **paper-flavored** (essay style, discursive,
  academic register) rather than chatbot-style.
- Model may **hallucinate chat scaffolding** ("you>" / "bot>" tokens
  appearing in output text) because chat template markup is out of
  the training distribution.
- Fast-weight carry may cause the model to **wander into whatever
  narrative the carry encodes** rather than answer the new question.

This isn't a bug in the mechanism; it's a training-data mismatch.
Fix would be a second-stage chat-format fine-tune, either raw
supervised or with a retrieval-shaped objective (see
[failure-modes.md](failure-modes.md) if this becomes a priority).

## Snapshot lifecycle

**Save:**
```
you> /save mychat
[saved fast weights -> /ckpt/ttt-v1.1/sessions/mychat.pt]
```

`export_fast_weights(model)` writes a dict `{layer_idx: fp32 delta
tensor on CPU}` for each TTT module with non-None `state.delta`.

**Resume in a later session:**
```
python chat_client.py --ckpt step_400 --from-snapshot mychat
```

`chat_reset(from_snapshot_name="mychat")`:
1. `reset_fast_weights(model)` — clear existing state.
2. Load snapshot from `<CKPT_MOUNT>/<run_name>/sessions/mychat.pt`.
3. `import_fast_weights(model, snapshot)` — populate `state.delta`
   from snapshot values, cast to device.

**Important:** a snapshot is only valid for the exact slow weights
it was created under. Loading a snapshot after further LoRA/wdown
training silently applies a delta against a `W0` that no longer
exists — behavior will be silently wrong.

## Related docs

- [inference.md](inference.md) — `TTTInference` class methods
- [mechanism.md](mechanism.md) — `_stream_forward` details
- [checkpoints.md](checkpoints.md) — snapshot storage
- [failure-modes.md](failure-modes.md) — chat wandering, OOD text
