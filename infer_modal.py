"""
Modal inference app for the trained In-Place TTT model.
The key knob everywhere is `evolve`: True updates fast weights chunk-by-chunk; False freezes them (eta=0 ablation).
"""

import os

import modal

from ttt_config import (
    CKPT_MOUNT, CKPT_VOLUME_NAME, DATASET_SPEC, HF_CACHE_MOUNT,
    HF_CACHE_VOLUME_NAME, TRAIN_CFG,
)

app = modal.App("inplace-ttt-infer")

# Forward TTT_* env vars from the local shell into the container so
# that `TTT_DATASET=... modal run` picks the same spec at train and
# eval time. See train_modal.py for the full rationale.
_FORWARD_ENV_KEYS = ("TTT_DATASET", "TTT_MODEL_SIZE", "TTT_LAYER_STRIDE",
                     "TTT_LAYER_START", "TTT_BASE_MODEL")
_FORWARDED_ENV = {k: os.environ[k]
                  for k in _FORWARD_ENV_KEYS if k in os.environ}
if _FORWARDED_ENV:
    print(f"[modal] forwarding env to container: "
          f"{ {k: v for k, v in _FORWARDED_ENV.items()} }")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "transformers>=4.51",
        "peft>=0.18.0",
        "accelerate>=1.0",
        "datasets>=3.0",
    )
    .env({"HF_HOME": HF_CACHE_MOUNT, **_FORWARDED_ENV})
    .add_local_python_source("ttt_config", "inplace_ttt", "ttt_wiring",
                             "model_setup", "data_utils", "chat_utils",
                             "train_utils")
)

ckpt_vol = modal.Volume.from_name(CKPT_VOLUME_NAME, create_if_missing=True)
hf_vol = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
VOLUMES = {CKPT_MOUNT: ckpt_vol, HF_CACHE_MOUNT: hf_vol}

SECRETS = [modal.Secret.from_name("wandb"), modal.Secret.from_name("huggingface")]


def _ckpt_paths(ckpt: str):
    """Checkpoint forms: 'step_600' (under TRAIN_CFG.run_name) or 'other_run/step_600' (explicit run). Empty string runs the untrained patched model."""
    if not ckpt:
        return None, None, None
    if "/" in ckpt:
        base = os.path.join(CKPT_MOUNT, ckpt)
    else:
        base = os.path.join(CKPT_MOUNT, TRAIN_CFG.run_name, ckpt)
    return (os.path.join(base, "adapter"),
            os.path.join(base, "ttt_params.pt"),
            os.path.join(base, "per_source_carries.pt"))


@app.cls(image=image,gpu=["H100", "A100-80GB"], volumes=VOLUMES, timeout=60 * 60,
         scaledown_window=300)
class TTTInference:
    ckpt: str = modal.parameter(default="")

    load_ttt: bool = modal.parameter(default=True)

    @modal.enter()
    def load(self):
        from model_setup import build_model
        from ttt_wiring import load_per_source_carries

        adapter, ttt_ckpt, per_source_pc = _ckpt_paths(self.ckpt)
        if not self.load_ttt:
            ttt_ckpt = None
        self.model, self.tokenizer = build_model(
            adapter_path=adapter, ttt_ckpt_path=ttt_ckpt, trainable=False,
            attn_impl="sdpa",
        )
        self.model.eval()
        self.model.config.use_cache = True

        # Everlasting-carry: load the per-source persistent carriers if the
        # checkpoint has them. `session_perplexity(use_everlasting_carry=True)`
        # installs `per_source_carries[doc.source]` before each doc's forward.
        # Empty dict when the file is absent -- backward compatible with
        # pre-everlasting checkpoints.
        self.per_source_carries = {}
        self.per_source_meta = {}
        if per_source_pc and self.load_ttt:
            loaded, meta = load_per_source_carries(per_source_pc)
            import torch as _torch
            for src, per_layer in loaded.items():
                self.per_source_carries[src] = {
                    int(k): v.to(device="cuda", dtype=_torch.float32)
                    for k, v in per_layer.items()
                }
            self.per_source_meta = meta
            if self.per_source_carries:
                n_updates = meta.get("n_updates", {})
                summary = ", ".join(
                    f"{s}:{n_updates.get(s, '?')}"
                    for s in sorted(self.per_source_carries)
                )
                print(f"loaded per-source carriers "
                      f"({len(self.per_source_carries)} sources, "
                      f"n_updates={{{summary}}})")

    def _set_mode(self, evolve: bool, stateful: bool, fresh: bool = True):
        from inplace_ttt import iter_ttt_modules, reset_fast_weights

        if fresh:
            reset_fast_weights(self.model)
        self.model._ttt_tap.stateful = stateful
        for m in iter_ttt_modules(self.model):
            m.stateful = stateful
            m.ttt_evolve = evolve

    @modal.method()
    def perplexity(self, text: str, evolve: bool = True) -> float:
        import math

        import torch

        self._set_mode(evolve=evolve, stateful=False)
        ids = self.tokenizer(text, return_tensors="pt",
                             truncation=True,
                             max_length=TRAIN_CFG.max_seq_len
                             ).input_ids.cuda()
        with torch.no_grad():
            loss = self.model(input_ids=ids, labels=ids).loss
        return math.exp(loss.item())

    @modal.method()
    def session_perplexity(self, texts: list, evolve: bool = True,
                           slice_papers: bool = True,
                           slice_seed: int = 0,
                           equal_n_slices: int = 0,
                           reset_between_items: bool = False,
                           reset_between_docs: bool = True,
                           use_everlasting_carry: bool = False,
                           force_source: str = "",
                           sources: list | None = None) -> list:
        """Per-item perplexity with the fast weight persisting across items
        by default. Reset behavior:
          - `reset_between_docs=True` (default): fast weight resets when
            `item.doc_idx` changes. Matches training-side semantics
            (each doc starts with a fresh state) and is what you want for
            most eval flows -- otherwise paper 1's state leaks into
            paper 2 and the reported state_ratio grows unboundedly across
            the multi-paper eval.
          - `reset_between_items=True`: fast weight resets before every
            item, including slices within a single doc. Combined with
            `evolve=True`, this is the "carry-off" mode -- isolates
            within-item chunk adaptation from cross-item persistence.
        Both flags can coexist; `reset_between_items` implies stronger
        resetting than `reset_between_docs`.

        Everlasting-carry:
          - `use_everlasting_carry=True` (only meaningful when the
            checkpoint carried `per_source_carries.pt`): after the reset
            at each doc boundary, install `per_source_carries[source]` as
            the fast-weight seed. Cold sources (not in the dict) still
            start from zero, matching training.
          - `force_source="RedPajamaC4"`: install THAT source's carrier
            regardless of the doc's actual source. Swap test: a source-
            specific benefit should degrade under a mismatched install.
          - `sources`: per-doc source labels, aligned with `texts`. When
            None but everlasting-carry is requested, all docs are treated
            as unlabeled (cold-start each doc unless `force_source` is set).
        """
        import math

        import numpy as np
        import torch

        from inplace_ttt import (
            advance_session_state, install_carried_delta, iter_ttt_modules,
            mean_state_ratio, reset_session_state, state_norms,
        )
        from train_utils import (
            SessionItem, equal_token_slices, make_slice_sessions,
        )

        self._set_mode(evolve=evolve, stateful=False)
        for m in iter_ttt_modules(self.model):
            m.session_mode = True
        reset_session_state(self.model)

        paper_token_ids = [
            self.tokenizer(
                t, return_tensors="pt", truncation=True,
                max_length=TRAIN_CFG.max_seq_len,
            ).input_ids[0].tolist()
            for t in texts
        ]
        doc_lengths = [len(ids) for ids in paper_token_ids]

        if equal_n_slices > 0:
            items = [
                SessionItem(paper_idx, s, e)
                for paper_idx, L in enumerate(doc_lengths)
                for s, e in equal_token_slices(L, equal_n_slices)
            ]
        elif slice_papers:
            rng = np.random.default_rng(slice_seed)
            items = make_slice_sessions(
                len(texts), doc_lengths, rng,
                session_papers=(len(texts), len(texts)),
                slice_prob=TRAIN_CFG.slice_prob,
                slice_range=(TRAIN_CFG.slice_min, TRAIN_CFG.slice_max),
                min_slice_tokens=TRAIN_CFG.slice_min_tokens,
                shuffle=False,
            )[0]
        else:
            items = [SessionItem(i, 0, doc_lengths[i])
                     for i in range(len(texts))]

        # Resolve which per-source carry to install for each doc.
        # `force_source` overrides the doc's own label (swap test).
        installable_carries = (getattr(self, "per_source_carries", {})
                               if use_everlasting_carry else {})
        forced_key = force_source if force_source else None

        def _install_for_doc(doc_idx: int) -> str:
            """Install this doc's per-source carrier (or `force_source`'s).
            Returns the label actually installed, or "" if nothing installed."""
            if not use_everlasting_carry:
                return ""
            src = (forced_key if forced_key is not None
                   else (sources[doc_idx] if sources
                         and doc_idx < len(sources) else ""))
            if src and src in installable_carries:
                install_carried_delta(self.model, installable_carries[src])
                return src
            return ""

        slice_in_paper = [0] * len(texts)
        prev_doc_idx = None
        out = []
        try:
            # First doc: install its carrier from the start.
            if items:
                _install_for_doc(items[0].doc_idx)
            for pos, item in enumerate(items):
                if (reset_between_docs and prev_doc_idx is not None
                        and item.doc_idx != prev_doc_idx):
                    reset_session_state(self.model)
                    _install_for_doc(item.doc_idx)
                prev_doc_idx = item.doc_idx
                ids = torch.tensor(
                    [paper_token_ids[item.doc_idx][item.start:item.end]],
                    device="cuda",
                )
                with torch.no_grad():
                    loss = self.model(input_ids=ids, labels=ids).loss
                advance_session_state(self.model)
                norms = state_norms(self.model, source="session")
                state_ratio = mean_state_ratio(norms)
                if reset_between_items:
                    reset_session_state(self.model)
                    # Reinstall so within-item chunk adaptation still
                    # starts from the per-source seed on the next item of
                    # the same doc.
                    _install_for_doc(item.doc_idx)
                out.append({
                    "paper_idx": int(item.doc_idx),
                    "slice_in_paper": slice_in_paper[item.doc_idx],
                    "session_pos": pos,
                    "start": int(item.start),
                    "end": int(item.end),
                    "n_tokens": int(item.end - item.start),
                    "ppl": math.exp(loss.item()),
                    "state_ratio_mean": state_ratio,
                })
                slice_in_paper[item.doc_idx] += 1
        finally:
            for m in iter_ttt_modules(self.model):
                m.session_mode = False
            reset_session_state(self.model)

        return out

    @modal.method()
    def generate(self, prompt: str, evolve: bool = True,
                 max_new_tokens: int = 512,
                 fast_weight_snapshot: dict | None = None,
                 temperature: float = 0.7, top_p: float = 0.9,
                 seed: int | None = None,
                 do_sample: bool = True) -> dict:
        """Streaming generation. Returns text plus a fast-weights snapshot for cross-session persistence."""
        import torch

        from inplace_ttt import export_fast_weights, import_fast_weights

        self._set_mode(evolve=evolve, stateful=True)
        if fast_weight_snapshot:
            import_fast_weights(self.model, fast_weight_snapshot)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=do_sample,
                temperature=temperature, top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0, ids.shape[1]:],
                                     skip_special_tokens=True)
        snapshot = export_fast_weights(self.model)
        return {"text": text, "fast_weights": snapshot}

    @modal.method()
    def fetch_holdout_texts(self, n_papers: int, seed: int = 0,
                            n_papers_per_source: int = 0,
                            min_tokens_est: int = 0) -> list:
        """Sample papers from the contamination-free holdout (newest
        `spec.holdout_last_n` rows excluded from training).

        - `n_papers_per_source > 0` and the dataset has a source column
          -> pick exactly that many per source (total =
          n_sources * n_papers_per_source). Ensures every domain
          contributes to the eval table.
        - Otherwise: sample `n_papers` uniformly.

        `min_tokens_est > 0` filters to docs whose `tokens_est` (or
        char-length proxy) meets the threshold, so eval doesn't pick
        tiny StackExchange posts.

        Returns `[{"text": str, "source": str}, ...]`.
        """
        import random

        from data_utils import apply_source_filter, open_dataset, split_holdout

        spec = DATASET_SPEC
        _, holdout = split_holdout(open_dataset(spec), spec)
        holdout = apply_source_filter(holdout, spec)

        if min_tokens_est > 0:
            tokens_est_col = spec.tokens_est_column
            if tokens_est_col and tokens_est_col in holdout.column_names:
                holdout = holdout.filter(
                    lambda ex: ex[tokens_est_col] >= min_tokens_est,
                )
            else:
                char_threshold = 4 * min_tokens_est
                text_col_local = spec.text_column
                holdout = holdout.filter(
                    lambda ex: len(ex[text_col_local]) >= char_threshold,
                )

        if len(holdout) == 0:
            return []

        rng = random.Random(seed)
        has_source = "source" in holdout.column_names
        text_col = spec.text_column

        if has_source and n_papers_per_source > 0:
            from collections import defaultdict
            by_src = defaultdict(list)
            for i, s in enumerate(holdout["source"]):
                by_src[s].append(i)
            indices = []
            for src in sorted(by_src):
                pool = by_src[src]
                rng.shuffle(pool)
                indices.extend(pool[:n_papers_per_source])
        else:
            n = min(n_papers, len(holdout))
            indices = rng.sample(range(len(holdout)), n)

        return [
            {"text": holdout[i][text_col],
             "source": (holdout[i]["source"] if has_source else "")}
            for i in indices
        ]

    @modal.method()
    def save_session(self, name: str):
        import torch

        from inplace_ttt import export_fast_weights

        path = os.path.join(CKPT_MOUNT, TRAIN_CFG.run_name,
                            "sessions", f"{name}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(export_fast_weights(self.model), path)
        ckpt_vol.commit()
        return path

    # Chat tests TTT as a memory mechanism: the model is given STRICTLY no in-context
    # access to prior turns. Each turn sees only "[<system>\n\n]User: ... \nAssistant: ".
    # Per-turn KV cache is used within the turn and discarded; the embedding tap's
    # rolling buffer is reset at the turn boundary. TTT fast-weight state
    # (state.delta + pending partial chunk) PERSISTS -- it is the only carrier of
    # cross-turn conversation memory. Re-feeding earlier turns breaks the test.
    @modal.method()
    def chat_reset(self, evolve: bool = True,
                   from_snapshot_name: str = "") -> dict:
        """Reset TTT fast weights and (optionally) load a snapshot."""
        import torch

        from inplace_ttt import import_fast_weights

        self._set_mode(evolve=evolve, stateful=True, fresh=True)
        seeded = False
        if from_snapshot_name:
            path = os.path.join(CKPT_MOUNT, TRAIN_CFG.run_name,
                                "sessions", f"{from_snapshot_name}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"snapshot not found: {path}")
            snapshot = torch.load(path, map_location="cuda")
            import_fast_weights(self.model, snapshot)
            seeded = True

        return {"ready": True, "evolve": evolve,
                "seeded_from_snapshot": seeded}

    @modal.method()
    def chat_turn(self, user_message: str,
                  system_prompt: str = "",
                  enable_thinking: bool = True,
                  evolve: bool = True,
                  max_new_tokens: int = 512,
                  temperature: float = 0.6, top_p: float = 0.95,
                  top_k: int = 20) -> dict:
        """One conversation turn. Defaults follow Qwen3 thinking-mode recommendation; greedy decoding is explicitly warned against (endless repetition)."""
        import torch

        from chat_utils import (
            chat_stop_token_ids, sample_top_p, split_thinking,
            strip_chat_specials,
        )
        from inplace_ttt import (
            iter_ttt_modules, mean_state_ratio, state_norms,
            stream_pending_progress,
        )

        # No fast-weight reset here; carry persists turn-to-turn within a container.
        self.model._ttt_tap.stateful = True
        for m in iter_ttt_modules(self.model):
            m.stateful = True
            m.ttt_evolve = evolve

        if not hasattr(self, "_chat_stop_ids"):
            self._chat_stop_ids = chat_stop_token_ids(self.tokenizer)

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        new_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        new_ids = self.tokenizer(
            new_text, return_tensors="pt"
        ).input_ids.cuda()

        generated = []
        past_kv = None     # within-turn cache only; never carried over
        stop_reason = "max_tokens"
        stop_token_id = None
        with torch.no_grad():
            # Prefill this turn only. No past_key_values from prior turns --
            # attention must NOT see any earlier conversation.
            out = self.model(input_ids=new_ids, use_cache=True)
            past_kv = out.past_key_values
            next_logits = out.logits[:, -1, :]

            for _ in range(max_new_tokens):
                next_id = sample_top_p(next_logits, temperature, top_p,
                                       top_k=top_k)
                if next_id in self._chat_stop_ids:
                    stop_reason = "stop_token"
                    stop_token_id = next_id
                    break
                generated.append(next_id)
                tok = torch.tensor([[next_id]], device="cuda")
                out = self.model(input_ids=tok, past_key_values=past_kv,
                                 use_cache=True)
                past_kv = out.past_key_values
                next_logits = out.logits[:, -1, :]

        # Turn boundary: drop within-turn KV cache and reset the conv left-context
        # buffer so the next turn's causal conv starts with zero left context.
        # TTT state.delta + pending chunk buffer PERSIST -- the only cross-turn memory.
        from inplace_ttt import reset_v_left_context
        reset_v_left_context(self.model)

        raw_with_specials = self.tokenizer.decode(
            generated, skip_special_tokens=False,
        )
        cleaned = strip_chat_specials(raw_with_specials)
        thinking_text, answer = split_thinking(cleaned)

        # state_ratio==0.0 has two meanings: evolve was off (no delta ever staged),
        # OR chunk_size tokens not yet accumulated. pending_tokens / chunk_size
        # disambiguates.
        norms = state_norms(self.model, source="stream")
        state_ratio = mean_state_ratio(norms)
        pending_tokens, chunk_size = stream_pending_progress(self.model)

        return {
            "text": answer.strip(),
            "thinking_text": thinking_text,
            "raw": raw_with_specials,
            "token_ids": [int(t) for t in generated],
            "stop_reason": stop_reason,
            "stop_token_id": (int(stop_token_id)
                              if stop_token_id is not None else None),
            "state_ratio_mean": float(state_ratio),
            "pending_tokens": int(pending_tokens),
            "chunk_size": int(chunk_size),
        }


@app.local_entrypoint()
def compare_ppl(text_path: str, ckpt: str = ""):
    text = open(text_path).read()
    engine = TTTInference(ckpt=ckpt)
    on = engine.perplexity.remote(text, evolve=True)
    off = engine.perplexity.remote(text, evolve=False)
    print(f"ppl  TTT on  {on:.3f}")
    print(f"ppl  TTT off {off:.3f}")
    print(f"gap          {off - on:+.3f}  (positive = TTT helping)")


def _print_paper_preview(texts: list, labels: list | None = None,
                         max_chars: int = 400):
    if labels is None:
        labels = [str(i + 1) for i in range(len(texts))]
    print("selected papers:")
    for label, text in zip(labels, texts):
        snippet = " ".join(text.strip().splitlines())
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        print(f"- {label}: {snippet}")
    print()


def _print_session_results(carry: list, carry_off: list, fresh: list,
                           paper_labels: list = None,
                           paper_sources: list | None = None):
    """Per-item table + token-weighted per-paper summary. Three ppl columns:
      carry     -- TTT on, fast weight carries across items
      carry-off -- TTT on, fast weight reset between items
                   (isolates within-item chunk adaptation)
      fresh     -- TTT off (fast weight = 0 throughout)
    Two gap columns:
      Δwithin  = fresh - carry-off  (within-item chunk-scan benefit)
      Δbetween = carry-off - carry  (cross-item session-carry benefit)
    Sum = Δtotal = fresh - carry."""
    import math

    print(f"{'pos':>4}  {'p.s':<6} {'n_tok':>6}  "
          f"{'carry':>9}  {'carry-off':>9}  {'fresh':>9}  "
          f"{'Δwithin':>9}  {'Δbetween':>9}  "
          f"{'state c':>9}  {'state co':>9}")
    for c, co, f in zip(carry, carry_off, fresh):
        label = f"{c['paper_idx'] + 1}.{c['slice_in_paper'] + 1}"
        d_within = f['ppl'] - co['ppl']
        d_between = co['ppl'] - c['ppl']
        state_c = c.get('state_ratio_mean', 0.0)
        state_co = co.get('state_ratio_mean', 0.0)
        print(f"{c['session_pos']:>4}  {label:<6} {c['n_tokens']:>6}  "
              f"{c['ppl']:>9.3f}  {co['ppl']:>9.3f}  {f['ppl']:>9.3f}  "
              f"{d_within:>+9.3f}  {d_between:>+9.3f}  "
              f"{state_c:>9.2e}  {state_co:>9.2e}")

    if not carry:
        return
    n_papers = max(c['paper_idx'] for c in carry) + 1
    label_width = max(
        (len(str(paper_labels[p])) for p in range(n_papers))
        if paper_labels else (len(str(p + 1)) for p in range(n_papers)),
        default=5,
    )
    label_width = max(label_width, len("paper"))

    print()
    print("per-paper (token-weighted):")
    print(f"{'paper':<{label_width}}  {'n_tok':>8}  "
          f"{'carry':>9}  {'carry-off':>9}  {'fresh':>9}  "
          f"{'Δwithin':>9}  {'Δbetween':>9}")
    for p in range(n_papers):
        c_log_tok = sum(math.log(c['ppl']) * c['n_tokens']
                        for c in carry if c['paper_idx'] == p)
        co_log_tok = sum(math.log(co['ppl']) * co['n_tokens']
                         for co in carry_off if co['paper_idx'] == p)
        f_log_tok = sum(math.log(f['ppl']) * f['n_tokens']
                        for f in fresh if f['paper_idx'] == p)
        n_tok = sum(c['n_tokens'] for c in carry if c['paper_idx'] == p)
        if not n_tok:
            continue
        c_ppl = math.exp(c_log_tok / n_tok)
        co_ppl = math.exp(co_log_tok / n_tok)
        f_ppl = math.exp(f_log_tok / n_tok)
        label = str(paper_labels[p]) if paper_labels else str(p + 1)
        d_within = f_ppl - co_ppl
        d_between = co_ppl - c_ppl
        print(f"{label:<{label_width}}  {n_tok:>8}  "
              f"{c_ppl:>9.3f}  {co_ppl:>9.3f}  {f_ppl:>9.3f}  "
              f"{d_within:>+9.3f}  {d_between:>+9.3f}")

    if paper_sources and any(paper_sources):
        _print_per_source_summary(carry, carry_off, fresh, paper_sources)


def _print_per_source_summary(carry: list, carry_off: list, fresh: list,
                              paper_sources: list):
    """Token-weighted PPL per source label (e.g. RedPajamaC4). Same three
    modes and two gaps as `_print_session_results`. Shows the core "which
    domain benefits most" table for a mixed-source dataset."""
    import math
    from collections import defaultdict

    by_src_carry = defaultdict(list)
    by_src_carry_off = defaultdict(list)
    by_src_fresh = defaultdict(list)
    for c in carry:
        src = paper_sources[c['paper_idx']] if c['paper_idx'] < len(paper_sources) else ""
        if src:
            by_src_carry[src].append(c)
    for co in carry_off:
        src = paper_sources[co['paper_idx']] if co['paper_idx'] < len(paper_sources) else ""
        if src:
            by_src_carry_off[src].append(co)
    for f in fresh:
        src = paper_sources[f['paper_idx']] if f['paper_idx'] < len(paper_sources) else ""
        if src:
            by_src_fresh[src].append(f)

    if not by_src_carry:
        return
    print()
    print("per-source (token-weighted):")
    label_w = max(len(s) for s in by_src_carry)
    label_w = max(label_w, len("source"))
    print(f"{'source':<{label_w}}  {'n_papers':>9}  {'n_tok':>10}  "
          f"{'carry':>9}  {'carry-off':>9}  {'fresh':>9}  "
          f"{'Δwithin':>9}  {'Δbetween':>9}")
    for src in sorted(by_src_carry):
        c_items = by_src_carry[src]
        co_items = by_src_carry_off.get(src, [])
        f_items = by_src_fresh.get(src, [])
        n_tok = sum(c['n_tokens'] for c in c_items)
        if not n_tok:
            continue
        c_log = sum(math.log(c['ppl']) * c['n_tokens'] for c in c_items)
        co_log = sum(math.log(co['ppl']) * co['n_tokens'] for co in co_items)
        f_log = sum(math.log(f['ppl']) * f['n_tokens'] for f in f_items)
        co_tok = sum(co['n_tokens'] for co in co_items) or n_tok
        f_tok = sum(f['n_tokens'] for f in f_items) or n_tok
        c_ppl = math.exp(c_log / n_tok)
        co_ppl = math.exp(co_log / co_tok)
        f_ppl = math.exp(f_log / f_tok)
        n_papers = len({c['paper_idx'] for c in c_items})
        print(f"{src:<{label_w}}  {n_papers:>9d}  {n_tok:>10d}  "
              f"{c_ppl:>9.3f}  {co_ppl:>9.3f}  {f_ppl:>9.3f}  "
              f"{f_ppl - co_ppl:>+9.3f}  {co_ppl - c_ppl:>+9.3f}")


def _three_way_eval(base_engine, ckpt: str, texts: list,
                    session_kwargs: dict,
                    paper_sources: list | None = None,
                    use_everlasting_carry: bool = False,
                    force_source: str = ""):
    """Print BASE / LORA-ONLY / FULL tables on the same texts. The three
    configs share input so any per-slice number is directly comparable.
    LORA-ONLY and FULL are skipped when ckpt is empty. `paper_sources`
    forwards a per-paper source label to the per-source summary.

    `use_everlasting_carry` and `force_source` only affect FULL (the only
    config that has the trained per-source carriers)."""
    def _run(engine, label, everlasting: bool):
        print(f"=== {label} ===")
        ec_kwargs = (
            {"use_everlasting_carry": True,
             "force_source": force_source,
             "sources": paper_sources or []}
            if everlasting else {}
        )
        try:
            carry = engine.session_perplexity.remote(
                texts, evolve=True, **session_kwargs, **ec_kwargs,
            )
            carry_off = engine.session_perplexity.remote(
                texts, evolve=True, reset_between_items=True,
                **session_kwargs, **ec_kwargs,
            )
            fresh = engine.session_perplexity.remote(
                texts, evolve=False, **session_kwargs, **ec_kwargs,
            )
            _print_session_results(carry, carry_off, fresh,
                                   paper_sources=paper_sources)
        except Exception as e:
            print(f"[failed: {e}]")
        print()

    _run(base_engine, "BASE  (Qwen3, no LoRA, no TTT)", everlasting=False)
    if ckpt:
        _run(TTTInference(ckpt=ckpt, load_ttt=False),
             f"LORA-ONLY  (ckpt={ckpt}, TTT silent)",
             everlasting=False)
        full_tag = ""
        if use_everlasting_carry:
            full_tag = " + everlasting-carry"
            if force_source:
                full_tag += f" (force_source={force_source!r})"
        _run(TTTInference(ckpt=ckpt, load_ttt=True),
             f"FULL  (ckpt={ckpt}, LoRA + TTT{full_tag})",
             everlasting=use_everlasting_carry)


@app.local_entrypoint()
def holdout_eval(n_papers: int = 5, seed: int = 0, ckpt: str = "",
                 slice_papers: bool = True,
                 equal_n_slices: int = 4,
                 n_papers_per_source: int = 2,
                 min_tokens_est: int = 0,
                 use_everlasting_carry: bool = False,
                 force_source: str = ""):
    """Three-way comparison in one run:
      1. BASE       -- pure Qwen3 (no LoRA, no TTT). The pretraining baseline.
      2. LORA-ONLY  -- trained LoRA on base, TTT silent (W_target=0).
      3. FULL       -- trained LoRA + trained TTT. The full model.
    Requires --ckpt for (2) and (3); with no --ckpt only BASE is shown.

    Each of (1)-(3) runs the same texts three times: carry (TTT + cross-item
    carry), carry-off (TTT within item only, reset between items), fresh
    (TTT silent). Δwithin = fresh - carry-off; Δbetween = carry-off - carry.

    Pass `--equal-n-slices 4` to force every paper into 4 equal slices
    (session_pos labels become 1.1, 1.2, 1.3, 1.4, 2.1, ...). Without it,
    each paper is one item.

    For a multi-source dataset (SlimPajama), pass
    `--n-papers-per-source 2` for a per-source table; also pass
    `--min-tokens-est 2048` to skip tiny StackExchange posts.

    Everlasting-carry (requires a checkpoint trained with --mode everlasting):
      --use-everlasting-carry              Install per-source carriers as
                                            the fast-weight seed on FULL.
      --force-source RedPajamaC4           Install THAT source's carrier
                                            for every doc regardless of its
                                            actual source (swap test)."""
    base_engine = TTTInference(ckpt="")
    rows = base_engine.fetch_holdout_texts.remote(
        n_papers=n_papers, seed=seed,
        n_papers_per_source=n_papers_per_source,
        min_tokens_est=min_tokens_est,
    )
    if not rows:
        print("no holdout papers available")
        return
    texts = [r["text"] for r in rows]
    sources = [r["source"] for r in rows]
    labels = [f"holdout {i+1}" + (f" [{s}]" if s else "")
              for i, s in enumerate(sources)]
    _print_paper_preview(texts, labels)
    _three_way_eval(
        base_engine, ckpt, texts,
        session_kwargs={"slice_papers": slice_papers, "slice_seed": seed,
                        "equal_n_slices": equal_n_slices},
        paper_sources=sources,
        use_everlasting_carry=use_everlasting_carry,
        force_source=force_source,
    )


@app.local_entrypoint()
def session_eval(papers_dir: str, ckpt: str = "",
                 slice_papers: bool = True, slice_seed: int = 0):
    """Feed every .txt in papers_dir (sorted) as one session, twice."""
    import glob

    paths = sorted(glob.glob(os.path.join(papers_dir, "*.txt")))
    texts = [open(p).read() for p in paths]
    labels = [os.path.basename(p) for p in paths]
    _print_paper_preview(texts, labels)
    engine = TTTInference(ckpt=ckpt)
    carry = engine.session_perplexity.remote(
        texts, evolve=True, slice_papers=slice_papers, slice_seed=slice_seed,
    )
    carry_off = engine.session_perplexity.remote(
        texts, evolve=True, reset_between_items=True,
        slice_papers=slice_papers, slice_seed=slice_seed,
    )
    fresh = engine.session_perplexity.remote(
        texts, evolve=False, slice_papers=slice_papers, slice_seed=slice_seed,
    )
    _print_session_results(
        carry, carry_off, fresh,
        paper_labels=labels,
    )


@app.local_entrypoint()
def generate_cli(prompt: str, ckpt: str = "", evolve: bool = True,
                 max_new_tokens: int = 512):
    engine = TTTInference(ckpt=ckpt)
    out = engine.generate.remote(prompt, evolve=evolve,
                                 max_new_tokens=max_new_tokens)
    print(out["text"])
    n = len(out["fast_weights"])
    print(f"\n[{n} TTT layers accumulated fast weight state]")


@app.local_entrypoint()
def holdout_generate(n_papers: int = 1, prefix_chars: int = 1200,
                     max_new_tokens: int = 120, seed: int = 0,
                     ckpt: str = "", temperature: float = 0.7,
                     top_p: float = 0.9, greedy: bool = False):
    """Side-by-side carry-on vs carry-off generation on held-out papers. Pass --greedy so the only differing factor is the carry."""
    engine = TTTInference(ckpt=ckpt)
    rows = engine.fetch_holdout_texts.remote(n_papers=n_papers, seed=seed)
    if not rows:
        print("no holdout papers available")
        return
    texts = [r["text"] for r in rows]

    do_sample = not greedy
    for i, text in enumerate(texts):
        prompt = text[:prefix_chars]
        print("=" * 80)
        print(f"paper {i+1}  (prefix {len(prompt)} chars, "
              f"greedy={greedy}, T={temperature}, top_p={top_p}, seed={seed})")
        print("-" * 80)
        print("PROMPT (tail):")
        print(prompt[-400:] if len(prompt) > 400 else prompt)
        print("-" * 80)
        carry = engine.generate.remote(
            prompt, evolve=True, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, seed=seed,
            do_sample=do_sample,
        )
        fresh = engine.generate.remote(
            prompt, evolve=False, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, seed=seed,
            do_sample=do_sample,
        )
        print("CARRY ON (evolve=True):")
        print(carry["text"])
        print("-" * 80)
        print("CARRY OFF (evolve=False):")
        print(fresh["text"])
        print()


@app.local_entrypoint()
def single_paper_eval(n_slices: int = 8, ckpt: str = "", seed: int = 0):
    """One held-out paper sliced into n equal-token parts, run as a single
    session. Prints the BASE / LORA-ONLY / FULL three-way comparison (see
    holdout_eval docstring). --ckpt='' shows only BASE."""
    base_engine = TTTInference(ckpt="")
    rows = base_engine.fetch_holdout_texts.remote(n_papers=1, seed=seed)
    if not rows:
        print("no holdout papers available")
        return
    texts = [r["text"] for r in rows]
    sources = [r["source"] for r in rows]
    label = "selected paper" + (f" [{sources[0]}]" if sources[0] else "")
    _print_paper_preview(texts, [label])
    _three_way_eval(
        base_engine, ckpt, texts,
        session_kwargs={"equal_n_slices": n_slices},
        paper_sources=sources,
    )


# Interactive chat lives in chat_client.py -- `modal run` doesn't forward stdin
# to local_entrypoint subprocesses, so the REPL runs as a regular python process
# and calls into the deployed class via modal.Cls.from_name(...).
