"""
Modal inference app for the trained In-Place TTT model.

The key knob everywhere is `evolve`:
    evolve=True   fast weights update chunk-by-chunk as text streams in
    evolve=False  fast weight evolution is frozen; any previously
                  accumulated (or imported) state is still APPLIED,
                  and with no state loaded the model behaves as plain
                  Qwen3-8B + LoRA. This is your eta-ablation in one flag.

Usage
-----
    # perplexity of a text file under both settings (the mechanism signal)
    modal run infer_modal.py::compare_ppl --text-path paper.txt \
        --ckpt step_600

    # generation, with TTT evolving over the prompt
    modal run infer_modal.py::generate_cli --prompt "..." --ckpt step_600

    # generation with evolution off
    modal run infer_modal.py::generate_cli --prompt "..." --ckpt step_600 \
        --no-evolve

    # interactive multi-turn chat; /save <name> persists the evolved
    # fast weights, /quit exits. Resume later with --from-snapshot <name>.
    modal run infer_modal.py::chat --ckpt step_600
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
    .add_local_python_source("ttt_config", "inplace_ttt", "model_setup",
                             "data_utils", "chat_utils")
)

ckpt_vol = modal.Volume.from_name(CKPT_VOLUME_NAME, create_if_missing=True)
hf_vol = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
VOLUMES = {CKPT_MOUNT: ckpt_vol, HF_CACHE_MOUNT: hf_vol}

# If the dataset repo is private, create the secret once
#   modal secret create huggingface HF_TOKEN=hf_...
# and add `secrets=SECRETS` to the @app.cls decorator below.
# SECRETS = [modal.Secret.from_name("huggingface")]

SECRETS = [modal.Secret.from_name("wandb"), modal.Secret.from_name("huggingface")]


def _ckpt_paths(ckpt: str):
    """ckpt like 'step_600' under the run dir from TRAIN_CFG.run_name.
    Pass an empty string to run the untrained patched model."""
    if not ckpt:
        return None, None
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
            adapter_path=adapter, ttt_ckpt_path=ttt_ckpt, trainable=False
        )
        self.model.eval()
        self.model.config.use_cache = True

    # ------------------------------------------------------------------
    def _set_mode(self, evolve: bool, stateful: bool, fresh: bool = True):
        """One place that flips all TTT switches (DRY)."""
        from inplace_ttt import (
            reset_fast_weights, set_ttt_evolve, set_ttt_stateful,
        )

        if fresh:
            reset_fast_weights(self.model)
        set_ttt_stateful(self.model, stateful)
        set_ttt_evolve(self.model, evolve)

    # ------------------------------------------------------------------
    @modal.method()
    def perplexity(self, text: str, evolve: bool = True) -> float:
        """Whole-sequence perplexity via the stateless scan path.
        evolve=False is the eta=0 ablation; the gap between the two
        numbers on held-out long papers is your mechanism signal."""
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

    # ------------------------------------------------------------------
    @modal.method()
    def session_perplexity(self, texts: list, evolve: bool = True) -> list:
        """Per-paper perplexity with fast weights persisting ACROSS the
        papers, mirroring session training. The headline persistence
        signal is later papers getting cheaper with evolve=True than
        with evolve=False, i.e. memory of earlier papers carrying over
        through weights, with no shared context window."""
        import math

        import torch

        from inplace_ttt import (
            advance_session_state, reset_session_state, set_session_mode,
        )

        self._set_mode(evolve=evolve, stateful=False)
        set_session_mode(self.model, True)
        reset_session_state(self.model)
        ppls = []
        try:
            for text in texts:
                ids = self.tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=TRAIN_CFG.max_seq_len,
                ).input_ids.cuda()
                with torch.no_grad():
                    loss = self.model(input_ids=ids, labels=ids).loss
                advance_session_state(self.model)
                ppls.append(math.exp(loss.item()))
        finally:
            set_session_mode(self.model, False)
            reset_session_state(self.model)
        return ppls

    # ------------------------------------------------------------------
    @modal.method()
    def generate(self, prompt: str, evolve: bool = True,
                 max_new_tokens: int = 512,
                 fast_weight_snapshot: dict | None = None) -> dict:
        """Streaming generation. Fast weights evolve over the prompt and
        the generated tokens (chunk by chunk) when evolve=True. Returns
        the text plus a snapshot of the accumulated fast weights so a
        later session can resume from them (cross-session persistence)."""
        import torch

        from inplace_ttt import export_fast_weights, import_fast_weights

        self._set_mode(evolve=evolve, stateful=True)
        if fast_weight_snapshot:
            import_fast_weights(self.model, fast_weight_snapshot)

        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0, ids.shape[1]:],
                                     skip_special_tokens=True)
        snapshot = export_fast_weights(self.model)
        return {"text": text, "fast_weights": snapshot}

    # ------------------------------------------------------------------
    @modal.method()
    def fetch_holdout_texts(self, n_papers: int, seed: int = 0) -> list:
        """Sample n papers from the contamination-free holdout (the
        newest HOLDOUT_LAST_N rows, which split_holdout excludes from
        training). Deterministic given the seed."""
        import random

        from data_utils import open_dataset, split_holdout

        _, holdout = split_holdout(open_dataset())
        rng = random.Random(seed)
        idx = rng.sample(range(len(holdout)), min(n_papers, len(holdout)))
        return [holdout[i][TEXT_COLUMN] for i in idx]

    # ------------------------------------------------------------------
    @modal.method()
    def save_session(self, name: str):
        """Persist the current fast weight state to the checkpoint
        volume, the unit of your session-boundary research."""
        import torch

        from inplace_ttt import export_fast_weights

        path = os.path.join(CKPT_MOUNT, TRAIN_CFG.run_name,
                            "sessions", f"{name}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(export_fast_weights(self.model), path)
        ckpt_vol.commit()
        return path

    # ------------------------------------------------------------------
    # Chat. The whole point of this loop is to test TTT as a memory
    # mechanism, so the model is given STRICTLY no in-context access to
    # the CONVERSATION (prior user/assistant turns). Each turn:
    #   * the model sees only "[<system>\n\n]User: ... \nAssistant: " --
    #     a static system prompt is allowed (it's an instruction, not
    #     conversation), but prior turns are not re-fed
    #   * the per-turn KV cache is used WITHIN the turn for O(N)
    #     sampling and then discarded
    #   * the embedding tap's rolling buffer is reset at the turn
    #     boundary so the causal conv's left context doesn't bleed
    #     prior-turn embeddings into the new turn
    #   * the TTT fast-weight state (state.delta + the pending partial
    #     chunk) PERSISTS -- it is the only carrier of cross-turn
    #     conversation memory.
    # If you ever start re-feeding earlier user/assistant turns into
    # the prompt, you are no longer testing the mechanism; you are
    # testing context-window memory.
    # ------------------------------------------------------------------
    @modal.method()
    def chat_reset(self, system_prompt: str = "", evolve: bool = True,
                   from_snapshot_name: str = "") -> dict:
        """Start a fresh chat. Resets TTT fast weights and the embedding
        rolling buffer; optionally loads a saved fast-weight snapshot.
        The system prompt is stored and prepended to every turn -- it is
        a static instruction, not in-context conversation memory."""
        import torch

        from chat_utils import chat_stop_token_ids
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

        self._chat_ready = True
        self._chat_system_prompt = system_prompt
        self._chat_stop_ids = chat_stop_token_ids(self.tokenizer)
        return {"ready": True, "evolve": evolve, "seeded_from_snapshot": seeded,
                "has_system_prompt": bool(system_prompt.strip())}

    @modal.method()
    def chat_turn(self, user_message: str, max_new_tokens: int = 512,
                  temperature: float = 0.7, top_p: float = 0.9) -> str:
        """One conversation turn. The model sees ONLY this turn's prompt
        (optional system prompt + the current user message) -- no past
        KV cache from prior turns, no re-fed conversation history.
        Within-turn KV cache is used for O(N) sampling and discarded at
        turn end; the embedding rolling buffer is reset. TTT fast
        weights persist across turns and are the SOLE channel for
        conversation memory."""
        import torch

        from chat_utils import format_user_turn, sample_top_p

        if not getattr(self, "_chat_ready", False):
            raise RuntimeError("call chat_reset() before chat_turn()")

        new_text = format_user_turn(user_message, self._chat_system_prompt)
        new_ids = self.tokenizer(
            new_text, return_tensors="pt"
        ).input_ids.cuda()

        generated = []
        past_kv = None     # within-turn cache only; never carried over
        with torch.no_grad():
            # Prefill on this turn only. No past_key_values from prior
            # turns -- attention must NOT see any earlier conversation.
            out = self.model(input_ids=new_ids, use_cache=True)
            past_kv = out.past_key_values
            next_logits = out.logits[:, -1, :]

            for _ in range(max_new_tokens):
                next_id = sample_top_p(next_logits, temperature, top_p)
                if next_id in self._chat_stop_ids:
                    break
                generated.append(next_id)
                tok = torch.tensor([[next_id]], device="cuda")
                out = self.model(input_ids=tok, past_key_values=past_kv,
                                 use_cache=True)
                past_kv = out.past_key_values
                next_logits = out.logits[:, -1, :]

        # Turn boundary: drop the within-turn KV cache (just go out of
        # scope) and reset the embedding rolling buffer so the next
        # turn's causal conv starts with zero left context. TTT
        # state.delta + pending chunk buffer PERSIST -- they are the
        # only memory carried across turns by design.
        self.model._ttt_tap.reset_stream()

        return self.tokenizer.decode(generated,
                                     skip_special_tokens=True).rstrip()


# ---------------------------------------------------------------------------
# CLI entrypoints
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def compare_ppl(text_path: str, ckpt: str = ""):
    """The single most informative eval. Run on held-out papers longer
    than ~4k tokens. No gap at 10k training papers => scale data, per
    the staged plan."""
    text = open(text_path).read()
    engine = TTTInference(ckpt=ckpt)
    on = engine.perplexity.remote(text, evolve=True)
    off = engine.perplexity.remote(text, evolve=False)
    print(f"ppl  TTT on  {on:.3f}")
    print(f"ppl  TTT off {off:.3f}")
    print(f"gap          {off - on:+.3f}  (positive = TTT helping)")


@app.local_entrypoint()
def holdout_eval(n_papers: int = 5, seed: int = 0, ckpt: str = ""):
    """One command, zero local files. Samples n held-out papers (never
    seen by the slow weights), runs them as one session with fast weight
    carry and once without. The carry-vs-fresh gap on papers 2..n is the
    cross-session memory signal."""
    engine = TTTInference(ckpt=ckpt)
    texts = engine.fetch_holdout_texts.remote(n_papers, seed)
    with_mem = engine.session_perplexity.remote(texts, evolve=True)
    without = engine.session_perplexity.remote(texts, evolve=False)
    print(f"{'paper #':<8} {'ppl carry':>10} {'ppl fresh':>10} {'gap':>8}")
    for k, (a, b) in enumerate(zip(with_mem, without), 1):
        print(f"{k:<8} {a:>10.3f} {b:>10.3f} {b - a:>+8.3f}")


@app.local_entrypoint()
def session_eval(papers_dir: str, ckpt: str = ""):
    """Feed every .txt in papers_dir (sorted) as one session, twice.
    Per-paper ppl with carry vs without isolates cross-paper memory."""
    import glob

    paths = sorted(glob.glob(os.path.join(papers_dir, "*.txt")))
    texts = [open(p).read() for p in paths]
    engine = TTTInference(ckpt=ckpt)
    with_mem = engine.session_perplexity.remote(texts, evolve=True)
    without = engine.session_perplexity.remote(texts, evolve=False)
    print(f"{'paper':<40} {'ppl carry':>10} {'ppl fresh':>10} {'gap':>8}")
    for p, a, b in zip(paths, with_mem, without):
        print(f"{os.path.basename(p):<40} {a:>10.3f} {b:>10.3f} "
              f"{b - a:>+8.3f}")


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
def chat(ckpt: str = "", evolve: bool = True, system: str = "",
         from_snapshot: str = "", max_new_tokens: int = 512,
         temperature: float = 0.7, top_p: float = 0.9):
    """Interactive REPL against the trained model. Fast weights evolve
    over the conversation when evolve=True; if the run is interesting,
    /save <name> persists the accumulated state under <run>/sessions/
    so a later chat can pick up where this one left off with
    --from-snapshot <name>.

    REPL commands:
        /save <name>   persist current fast weights as a snapshot
        /reset         start over with the same model (drops chat state)
        /quit          exit"""
    engine = TTTInference(ckpt=ckpt)
    info = engine.chat_reset.remote(
        system_prompt=system, evolve=evolve,
        from_snapshot_name=from_snapshot,
    )
    print(f"chat ready  (evolve={info['evolve']}, "
          f"seeded_from_snapshot={info['seeded_from_snapshot']}, "
          f"system_prompt={info['has_system_prompt']})")
    print("commands: /save <name>, /reset, /quit")
    print()
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            engine.chat_reset.remote(system_prompt=system, evolve=evolve)
            print("[reset]\n")
            continue
        if user.startswith("/save"):
            parts = user.split(maxsplit=1)
            if len(parts) != 2 or not parts[1].strip():
                print("[usage: /save <name>]\n")
                continue
            try:
                path = engine.save_session.remote(name=parts[1].strip())
                print(f"[saved fast weights -> {path}]\n")
            except Exception as e:
                print(f"[save error: {e}]\n")
            continue
        try:
            reply = engine.chat_turn.remote(
                user_message=user,
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            )
        except Exception as e:
            print(f"[error: {e}]\n")
            continue
        print(f"bot> {reply}\n")
