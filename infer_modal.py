"""
Modal inference app for the trained In-Place TTT model.
The key knob everywhere is `evolve`: True updates fast weights chunk-by-chunk; False freezes them (eta=0 ablation).
"""

import os

import modal

from ttt_config import (
    CKPT_MOUNT, CKPT_VOLUME_NAME, HF_CACHE_MOUNT, HF_CACHE_VOLUME_NAME,
    TEXT_COLUMN, TRAIN_CFG,
)

app = modal.App("inplace-ttt-infer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "transformers>=4.51",
        "peft>=0.18.0",
        "accelerate>=1.0",
        "datasets>=3.0",
    )
    .env({"HF_HOME": HF_CACHE_MOUNT})
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
        return None, None
    if "/" in ckpt:
        base = os.path.join(CKPT_MOUNT, ckpt)
    else:
        base = os.path.join(CKPT_MOUNT, TRAIN_CFG.run_name, ckpt)
    return os.path.join(base, "adapter"), os.path.join(base, "ttt_params.pt")


@app.cls(image=image,gpu=["H100", "A100-80GB"], volumes=VOLUMES, timeout=60 * 60,
         scaledown_window=300)
class TTTInference:
    ckpt: str = modal.parameter(default="")

    @modal.enter()
    def load(self):
        from model_setup import build_model

        adapter, ttt_ckpt = _ckpt_paths(self.ckpt)
        self.model, self.tokenizer = build_model(
            adapter_path=adapter, ttt_ckpt_path=ttt_ckpt, trainable=False,
            attn_impl="sdpa",
        )
        self.model.eval()
        self.model.config.use_cache = True

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
                           equal_n_slices: int = 0) -> list:
        """Per-slice perplexity with fast weights persisting across the session."""
        import math

        import numpy as np
        import torch

        from inplace_ttt import (
            advance_session_state, iter_ttt_modules, mean_state_ratio,
            reset_session_state, state_norms,
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

        slice_in_paper = [0] * len(texts)
        out = []
        try:
            for pos, item in enumerate(items):
                ids = torch.tensor(
                    [paper_token_ids[item.doc_idx][item.start:item.end]],
                    device="cuda",
                )
                with torch.no_grad():
                    loss = self.model(input_ids=ids, labels=ids).loss
                advance_session_state(self.model)
                norms = state_norms(self.model, source="session")
                state_ratio = mean_state_ratio(norms)
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
    def fetch_holdout_texts(self, n_papers: int, seed: int = 0) -> list:
        """Sample n papers from the contamination-free holdout (newest HOLDOUT_LAST_N rows excluded from training)."""
        import random

        from data_utils import open_dataset, split_holdout

        _, holdout = split_holdout(open_dataset())
        rng = random.Random(seed)
        idx = rng.sample(range(len(holdout)), min(n_papers, len(holdout)))
        return [holdout[i][TEXT_COLUMN] for i in idx]

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


def _print_session_results(carry: list, fresh: list,
                           paper_labels: list = None):
    """Per-item table + token-weighted per-paper summary. 'state' col is ||eta*carried_delta||_F / ||W_down||_F averaged across TTT layers."""
    import math

    print(f"{'pos':>4}  {'p.s':<6} {'n_tok':>6}  "
          f"{'ppl carry':>10}  {'ppl fresh':>10}  {'gap':>8}  "
          f"{'state':>10}")
    for c, f in zip(carry, fresh):
        label = f"{c['paper_idx'] + 1}.{c['slice_in_paper'] + 1}"
        gap = f['ppl'] - c['ppl']
        state = c.get('state_ratio_mean', 0.0)
        print(f"{c['session_pos']:>4}  {label:<6} {c['n_tokens']:>6}  "
              f"{c['ppl']:>10.3f}  {f['ppl']:>10.3f}  {gap:>+8.3f}  "
              f"{state:>10.2e}")

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
          f"{'ppl carry':>10}  {'ppl fresh':>10}  {'gap':>8}")
    for p in range(n_papers):
        c_log_tok = sum(math.log(c['ppl']) * c['n_tokens']
                        for c in carry if c['paper_idx'] == p)
        f_log_tok = sum(math.log(f['ppl']) * f['n_tokens']
                        for f in fresh if f['paper_idx'] == p)
        n_tok = sum(c['n_tokens'] for c in carry if c['paper_idx'] == p)
        if not n_tok:
            continue
        c_ppl = math.exp(c_log_tok / n_tok)
        f_ppl = math.exp(f_log_tok / n_tok)
        label = str(paper_labels[p]) if paper_labels else str(p + 1)
        gap = f_ppl - c_ppl
        print(f"{label:<{label_width}}  {n_tok:>8}  "
              f"{c_ppl:>10.3f}  {f_ppl:>10.3f}  {gap:>+8.3f}")


@app.local_entrypoint()
def holdout_eval(n_papers: int = 5, seed: int = 0, ckpt: str = "",
                 slice_papers: bool = True):
    engine = TTTInference(ckpt=ckpt)
    texts = engine.fetch_holdout_texts.remote(n_papers, seed)
    _print_paper_preview(texts, [f"holdout {i+1}" for i in range(len(texts))])
    carry = engine.session_perplexity.remote(
        texts, evolve=True, slice_papers=slice_papers, slice_seed=seed,
    )
    fresh = engine.session_perplexity.remote(
        texts, evolve=False, slice_papers=slice_papers, slice_seed=seed,
    )
    _print_session_results(carry, fresh)


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
    fresh = engine.session_perplexity.remote(
        texts, evolve=False, slice_papers=slice_papers, slice_seed=slice_seed,
    )
    _print_session_results(
        carry, fresh,
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
    texts = engine.fetch_holdout_texts.remote(n_papers, seed)
    if not texts:
        print("no holdout papers available")
        return

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
    """One held-out paper sliced into n equal-token parts, run as a single session."""
    engine = TTTInference(ckpt=ckpt)
    texts = engine.fetch_holdout_texts.remote(1, seed)
    if not texts:
        print("no holdout papers available")
        return
    _print_paper_preview(texts, ["selected paper"])
    carry = engine.session_perplexity.remote(
        texts, evolve=True, equal_n_slices=n_slices,
    )
    fresh = engine.session_perplexity.remote(
        texts, evolve=False, equal_n_slices=n_slices,
    )
    _print_session_results(carry, fresh)


# Interactive chat lives in chat_client.py -- `modal run` doesn't forward stdin
# to local_entrypoint subprocesses, so the REPL runs as a regular python process
# and calls into the deployed class via modal.Cls.from_name(...).
