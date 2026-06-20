"""
Modal app for In-Place TTT continual pretraining on Qwen3-8B.

Usage
-----
    # 0. one-time wiring check, must print max |logit diff| ~ 0
    modal run train_modal.py::sanity_check

    # 1. overfit smoke test on 100 papers, loss must fall fast
    modal run --detach train_modal.py::train --limit-docs 100 --num-epochs 5

    # 2. the real run
    modal run --detach train_modal.py::train

Checkpoints land in the 'ttt-checkpoints' volume as
    /ckpt/<run_name>/step_<n>/adapter/        (PEFT LoRA adapter)
    /ckpt/<run_name>/step_<n>/ttt_params.pt   (W_down, W_target, Conv1D)
"""

import dataclasses
import math
import os
import time

import modal

from ttt_config import (
    CKPT_MOUNT, CKPT_VOLUME_NAME, HF_CACHE_MOUNT, HF_CACHE_VOLUME_NAME,
    TEXT_COLUMN, TOKENS_EST_COLUMN, TRAIN_CFG, TTT_CFG,
)

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------
app = modal.App("inplace-ttt-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "transformers>=4.51",
        "peft>=0.18.0",
        "datasets>=3.0",
        "bitsandbytes>=0.45",
        "accelerate>=1.0",
        "wandb>=0.19",
    )
    # Prebuilt wheel matches torch 2.8 + cu12 + cp311 + cxx11abiTRUE.
    # Avoids needing nvcc in the base image (debian_slim has no CUDA toolkit).
    .pip_install(
        "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
    )
    .env({"HF_HOME": HF_CACHE_MOUNT})
    .add_local_python_source("ttt_config", "inplace_ttt", "model_setup",
                             "data_utils", "observability", "train_utils")
)

ckpt_vol = modal.Volume.from_name(CKPT_VOLUME_NAME, create_if_missing=True)
hf_vol = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

VOLUMES = {CKPT_MOUNT: ckpt_vol, HF_CACHE_MOUNT: hf_vol}
GPU = "H100"          # A100-80GB also works, ["H100", "A100-80GB"] for fallback

# Telemetry needs a wandb API key, create it once with
#   modal secret create wandb WANDB_API_KEY=...
# If the dataset repo is private, also
#   modal secret create huggingface HF_TOKEN=hf_...
# and append modal.Secret.from_name("huggingface") to SECRETS.
SECRETS = [modal.Secret.from_name("wandb"), modal.Secret.from_name("huggingface")]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_token_dataset(tokenizer, limit_docs: int | None):
    """One paper per sequence, truncated to max_seq_len, short docs
    dropped (a doc must span multiple chunks for TTT to do anything).
    No packing; document-boundary semantics are handled by the session
    machinery, not by sequence construction."""
    from data_utils import open_dataset, split_holdout

    cfg = TRAIN_CFG
    ds, _ = split_holdout(open_dataset())

    # Cheap pre-filter on the precomputed estimate, then an exact filter
    # after tokenization. Saves tokenizing docs that would be dropped.
    if TOKENS_EST_COLUMN in ds.column_names:
        ds = ds.filter(
            lambda ex: ex[TOKENS_EST_COLUMN] >= cfg.min_doc_tokens,
            desc="pre-filter by tokens_est",
        )
    if limit_docs:
        ds = ds.select(range(min(limit_docs, len(ds))))

    def tokenize(batch):
        out = tokenizer(
            batch[TEXT_COLUMN], truncation=True, max_length=cfg.max_seq_len,
            return_attention_mask=False,
        )
        return {"input_ids": out["input_ids"]}

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names,
                desc="tokenizing")
    ds = ds.filter(lambda ex: len(ex["input_ids"]) >= cfg.min_doc_tokens,
                   desc="dropping short docs (exact)")
    print(f"dataset ready, {len(ds)} documents")
    return ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS,
              timeout=60 * 60 * 24)
def train(limit_docs: int = 0, num_epochs: int = 0,
          grad_accum: int = 0, session: int = -1,
          single_paper: int = -1):
    import numpy as np
    import torch
    from transformers import get_cosine_schedule_with_warmup

    from inplace_ttt import (
        advance_session_state, build_param_groups, iter_ttt_modules,
        reset_session_state, session_state_norms, set_session_mode,
    )
    from model_setup import build_model
    from observability import (
        Telemetry, gpu_stats, param_health, session_metrics,
        snapshot_wdown,
    )
    from train_utils import (
        apply_loss_mask, apply_protect_passes, build_common_token_mask,
        build_session_items, common_mask_from_counts,
        expected_items_per_doc, grad_norms, load_reference_counts,
        make_session_schedule, make_single_paper_sessions,
    )

    overrides = {}
    if num_epochs:
        overrides["num_epochs"] = num_epochs
    if grad_accum:
        overrides["grad_accum_steps"] = grad_accum
    if session in (0, 1):
        overrides["session_training"] = bool(session)
    if single_paper in (0, 1):
        overrides["single_paper_sessions"] = bool(single_paper)
    cfg = dataclasses.replace(TRAIN_CFG, **overrides) if overrides else TRAIN_CFG
    epochs = cfg.num_epochs
    torch.manual_seed(cfg.seed)

    # ---- model -----------------------------------------------------------
    model, tokenizer = build_model(trainable=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # Required so checkpointed segments get gradients when inputs come
    # from frozen (no-grad) embeddings. Classic PEFT + checkpointing trap.
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    set_session_mode(model, cfg.session_training)

    optim_groups, named_groups = build_param_groups(
        model, TTT_CFG,
        lr_lora=cfg.lr_lora, lr_wdown=cfg.lr_wdown, lr_new=cfg.lr_new_modules,
        wd_full=cfg.weight_decay_full, wd_lora=cfg.weight_decay_lora,
    )
    n_train = sum(p.numel() for g in optim_groups for p in g["params"])
    print(f"trainable params {n_train/1e6:.1f}M "
          f"(lora {sum(p.numel() for p in named_groups['lora'])/1e6:.1f}M, "
          f"wdown {sum(p.numel() for p in named_groups['wdown'])/1e6:.1f}M, "
          f"new {sum(p.numel() for p in named_groups['new'])/1e6:.1f}M)")

    # ---- observability -----------------------------------------------------
    from dataclasses import asdict
    telemetry = Telemetry(
        enabled=cfg.wandb_enabled, project=cfg.wandb_project,
        run_name=cfg.run_name, job_type="train",
        config={**asdict(cfg), **asdict(TTT_CFG),
                "trainable_params_M": n_train / 1e6},
    )
    ttt_modules = list(iter_ttt_modules(model))
    wdown_init = snapshot_wdown(named_groups["wdown"])  # drift baseline

    # 8-bit Adam keeps optimizer state for the ~400M fully trained params
    # small. The model itself stays bf16; NF4 quantization is incompatible
    # with fully training W_down, so do not add QLoRA here.
    import bitsandbytes as bnb
    optimizer = bnb.optim.PagedAdamW8bit(optim_groups, betas=(0.9, 0.95))

    # ---- data ------------------------------------------------------------
    ds = load_token_dataset(tokenizer, limit_docs or None)
    hf_vol.commit()   # persist dataset + model downloads for next runs
    # One pass to memoize per-doc lengths so the session scheduler can
    # slice token ranges without re-reading each row. HF datasets are
    # memory-mapped, so iterating is cheap and the resulting list is
    # small (one int per doc).
    doc_lengths = [len(ex["input_ids"]) for ex in ds]

    # Content-token loss mask. Built once over either a reference
    # general-English corpus (preferred, structural fix for "domain
    # glue looks common") or the actual training corpus (fallback).
    # Lives on GPU for cheap indexing in the hot loop; ~150KB at
    # Qwen3's vocab size, negligible.
    common_mask = None
    if cfg.loss_mask_enabled:
        # Prefer external reference; fall back to in-corpus if missing.
        ref_counts, ref_meta = load_reference_counts(
            cfg.loss_mask_reference_counts_path, model.config.vocab_size,
        )
        if ref_counts is not None:
            print(f"loss mask: using external reference from "
                  f"{ref_meta.get('dataset_id', '?')}/"
                  f"{ref_meta.get('dataset_config', '?')} "
                  f"({ref_meta.get('n_tokens', 0):,} tokens)")
            common_mask = common_mask_from_counts(
                ref_counts, keep_fraction=cfg.loss_mask_keep_fraction,
            )
        else:
            if cfg.loss_mask_reference_counts_path:
                print(f"loss mask: reference path "
                      f"{cfg.loss_mask_reference_counts_path!r} not "
                      f"found, falling back to in-corpus frequency. "
                      f"Run build_reference_counts to populate.")
            common_mask = build_common_token_mask(
                (ex["input_ids"] for ex in ds),
                vocab_size=model.config.vocab_size,
                keep_fraction=cfg.loss_mask_keep_fraction,
            )
        n_before = int(common_mask.sum())
        n_freed_terms, n_freed_numeric = apply_protect_passes(
            common_mask, tokenizer,
            cfg.loss_mask_protect_terms,
            cfg.loss_mask_protect_numeric,
        )
        common_mask = common_mask.cuda()
        n_final = int(common_mask.sum())
        print(f"loss mask: {n_final}/{model.config.vocab_size} token ids "
              f"masked (kf={cfg.loss_mask_keep_fraction:.2f}, "
              f"protect-terms freed {n_freed_terms}, "
              f"protect-numeric freed {n_freed_numeric}, "
              f"initial {n_before}); "
              f"first_tokens={cfg.loss_mask_first_tokens} at paper-start items")
    # Slicing inflates the number of forward/backward passes per epoch.
    # Use the expected items/doc to size the cosine schedule so warmup
    # and anneal land roughly where they would for a non-sliced run.
    if cfg.single_paper_sessions:
        # Each session = one paper sliced into k ~ U[lo, hi] pieces, so
        # items per doc is the mean of that uniform.
        items_per_doc = 0.5 * (
            cfg.single_paper_slices_min + cfg.single_paper_slices_max
        )
    else:
        items_per_doc = expected_items_per_doc(
            cfg.slice_prob, cfg.slice_min, cfg.slice_max
        )
    items_per_epoch = math.ceil(len(ds) * items_per_doc)
    steps_per_epoch = math.ceil(items_per_epoch / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(cfg.warmup_min_steps, int(cfg.warmup_ratio * total_steps)),
        num_training_steps=total_steps,
    )
    if cfg.single_paper_sessions:
        mode_line = (f"single_paper_sessions=True "
                     f"k in [{cfg.single_paper_slices_min}, "
                     f"{cfg.single_paper_slices_max}] "
                     f"min_tok={cfg.slice_min_tokens}")
    else:
        mode_line = (f"n_papers in [{cfg.session_papers_min}, "
                     f"{cfg.session_papers_max}]; "
                     f"slice_prob={cfg.slice_prob} "
                     f"k in [{cfg.slice_min}, {cfg.slice_max}] "
                     f"min_tok={cfg.slice_min_tokens}")
    print(f"{total_steps} optimizer steps "
          f"({len(ds)} docs x {epochs} epochs / accum {cfg.grad_accum_steps}); "
          f"session_training={cfg.session_training}; {mode_line}")

    # ---- loop ------------------------------------------------------------
    # One paper = one forward/backward, exactly as before. Sessions only
    # control the fast weight carry. Note an optimizer step landing mid
    # session makes the carried state slightly stale w.r.t. the freshly
    # updated slow weights; this is standard TBPTT behavior and benign.
    run_dir = os.path.join(CKPT_MOUNT, cfg.run_name)
    rng = np.random.default_rng(cfg.seed)
    step, micro, t0 = 0, 0, time.time()
    running, step_loss = 0.0, 0.0
    window_tokens, total_tokens, sessions_done, nonfinite = 0, 0, 0, 0

    for epoch in range(epochs):
        if cfg.single_paper_sessions:
            sessions = make_single_paper_sessions(
                len(ds), doc_lengths,
                cfg.single_paper_slices_min, cfg.single_paper_slices_max,
                cfg.slice_min_tokens, rng,
            )
        else:
            sessions = make_session_schedule(
                len(ds), cfg.session_papers_min, cfg.session_papers_max, rng,
            )
            sessions = build_session_items(
                sessions, doc_lengths,
                slice_prob=cfg.slice_prob, slice_min=cfg.slice_min,
                slice_max=cfg.slice_max, min_slice_tokens=cfg.slice_min_tokens,
                rng=rng,
            )
        for session in sessions:
            reset_session_state(model)
            for pos, item in enumerate(session):
                # SessionItem may be a whole paper (start=0, end=len) or
                # one slice of one. From the model's perspective both
                # look like a sequence; the fast-weight carry threads
                # all of them.
                full_ids = ds[item.doc_idx]["input_ids"]
                ids = torch.tensor(
                    [full_ids[item.start:item.end]], device="cuda"
                )
                # Leading-token mask applies ONLY at paper-start items;
                # mid-paper slices (start > 0) are real content, not
                # shared boilerplate.
                first_n = (
                    cfg.loss_mask_first_tokens if item.start == 0 else 0
                )
                labels = apply_loss_mask(ids, common_mask,
                                         first_tokens=first_n)
                loss = model(input_ids=ids, labels=labels).loss

                # Anomaly guard: a nonfinite loss must not poison the
                # accumulated gradients. Skip backward, count, alert.
                if not torch.isfinite(loss):
                    nonfinite += 1
                    telemetry.log({"anomaly/nonfinite_count": nonfinite,
                                   "micro/step": micro})
                    if nonfinite in (1, 10, 100):
                        telemetry.alert(
                            "Nonfinite loss",
                            f"{nonfinite} nonfinite losses, last at micro "
                            f"{micro} (doc {item.doc_idx} "
                            f"[{item.start}:{item.end}], epoch {epoch})",
                        )
                    advance_session_state(model)  # keep carry semantics
                    micro += 1
                    continue

                (loss / cfg.grad_accum_steps).backward()
                advance_session_state(model)   # carry fast weights forward
                step_loss += loss.item() / cfg.grad_accum_steps
                n_tok = ids.numel()
                window_tokens += n_tok
                total_tokens += n_tok
                state_norms = session_state_norms(model)
                state_ratio_mean = (
                    sum(state_norms.values()) / len(state_norms)
                    if state_norms else 0.0
                )
                micro_log = {
                    "micro/step": micro,
                    "micro/paper_loss": loss.item(),
                    "micro/paper_tokens": n_tok,
                    "micro/session_pos": pos,
                    "micro/session_n": len(session),
                    "micro/state_ratio_mean": state_ratio_mean,
                }
                if common_mask is not None:
                    # Fraction of positions that contributed to the loss
                    # this step -- the runtime check that masking is doing
                    # roughly what loss_mask_keep_fraction promised.
                    micro_log["micro/unmasked_token_frac"] = (
                        float((labels != -100).float().mean())
                    )
                telemetry.log(micro_log)
                micro += 1

                if micro % cfg.grad_accum_steps:
                    continue

                norms = grad_norms(named_groups)   # before clipping
                total_norm = torch.nn.utils.clip_grad_norm_(
                    (p for g in optim_groups for p in g["params"]),
                    cfg.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                # ---- per-step telemetry (cheap scalars) ----------------
                lrs = scheduler.get_last_lr()
                dt = time.time() - t0
                metrics = {
                    "train/step": step,
                    "train/loss": step_loss,
                    "train/lr_lora": lrs[0],
                    "train/lr_wdown": lrs[1],
                    "train/lr_new": lrs[2],
                    "train/grad_clip_ratio": float(
                        min(1.0, cfg.max_grad_norm / (float(total_norm)
                                                      + 1e-12))
                    ),
                    "grad/lora": norms["lora"],
                    "grad/wdown": norms["wdown"],
                    "grad/new": norms["new"],
                    "perf/tokens_per_s": window_tokens / max(dt, 1e-9),
                    "perf/sec_per_step": dt,
                    "perf/total_tokens": total_tokens,
                    "session/sessions_done": sessions_done,
                    **session_metrics(session_state_norms(model)),
                    **gpu_stats(),
                }
                if step % cfg.param_log_every == 0:
                    metrics.update(
                        param_health(named_groups, wdown_init, ttt_modules)
                    )
                telemetry.log(metrics)
                running += step_loss
                step_loss = 0.0
                t0, window_tokens = time.time(), 0

                if step % cfg.log_every == 0:
                    print(
                        f"epoch {epoch} step {step}/{total_steps} "
                        f"loss {running/cfg.log_every:.4f} "
                        f"lr {lrs[0]:.2e} "
                        f"|g|lora {norms['lora']:.3f} "
                        f"|g|wdown {norms['wdown']:.3f} "
                        f"|g|new {norms['new']:.3f} "
                        f"state/W0 {metrics.get('session/state_ratio_mean', 0):.2e}"
                    )
                    # If grad/new stays ~0 while grad/lora is healthy, the
                    # X0 tap or target computation is broken. Stop and debug.
                    # If session/state_ratio_* climbs steadily past ~1e-1,
                    # unbounded fast weight growth; add forgetting.
                    running = 0.0

                if step % cfg.save_every == 0 or step == total_steps:
                    save_checkpoint(model, run_dir, step)
                    ckpt_vol.commit()
            sessions_done += 1

    save_checkpoint(model, run_dir, step)
    ckpt_vol.commit()
    telemetry.finish()
    print("done")


def save_checkpoint(model, run_dir: str, step: int):
    from inplace_ttt import save_ttt_state_dict

    path = os.path.join(run_dir, f"step_{step}")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(os.path.join(path, "adapter"))           # LoRA
    save_ttt_state_dict(model, os.path.join(path, "ttt_params.pt"), TTT_CFG)
    print(f"saved checkpoint -> {path}")


# ---------------------------------------------------------------------------
# One-time job: precompute reference unigram counts for the loss mask.
#
#     modal run train_modal.py::build_reference_counts
#     modal run train_modal.py::build_reference_counts --dataset-id wikipedia \
#         --dataset-config 20220301.en --split train --out-name reference_wiki.pt
#
# Outputs land at /ckpt/loss_mask/<out_name>; the default matches
# TRAIN_CFG.loss_mask_reference_counts_path so subsequent training picks
# it up automatically.
# ---------------------------------------------------------------------------
@app.function(image=image, volumes=VOLUMES, secrets=SECRETS,
              timeout=60 * 60)
def build_reference_counts(
    dataset_id: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "train",
    text_column: str = "text",
    out_name: str = "reference_wikitext103.pt",
    limit_docs: int = 0,
):
    """Build a [vocab_size] unigram count tensor from a general-English
    reference corpus, tokenized with Qwen3's BPE so token ids align
    with the training run. Saved to /ckpt/loss_mask/<out_name> as a
    pickled dict containing the counts plus metadata (tokenizer name,
    dataset id, n_tokens, ...).

    Run once per tokenizer change; the resulting counts are reusable
    across all training runs. Cost: ~10-20 min on CPU for wikitext-103.

    The fallback path in train() means a missing file does not block
    training -- you can ship the in-corpus baseline first and add the
    reference later when convenient.
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from ttt_config import BASE_MODEL
    from train_utils import count_unigrams

    print(f"loading {dataset_id} / {dataset_config} / {split} from HF Hub")
    ds = load_dataset(dataset_id, dataset_config, split=split)
    if limit_docs:
        ds = ds.select(range(min(limit_docs, len(ds))))
    print(f"loaded {len(ds)} rows; text column = {text_column!r}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    vocab_size = max(tokenizer.vocab_size, len(tokenizer))
    print(f"tokenizer vocab_size = {vocab_size}")

    # Batched tokenization. Wikitext rows are short paragraph chunks,
    # some are empty header lines; the tokenizer handles those fine.
    # add_special_tokens=False so BOS/EOS don't pollute the unigram
    # tally with artifacts that never appear at training time.
    def tokenize(batch):
        out = tokenizer(batch[text_column],
                        add_special_tokens=False, truncation=False)
        return {"input_ids": out["input_ids"]}

    print("tokenizing (batched)...")
    tokens_ds = ds.map(tokenize, batched=True, batch_size=1000,
                       remove_columns=ds.column_names, desc="tokenize")

    print("counting unigrams (one pass)...")
    counts = count_unigrams(
        (ex["input_ids"] for ex in tokens_ds), vocab_size=vocab_size,
    )
    n_tokens = int(counts.sum().item())
    n_unique = int((counts > 0).sum().item())
    print(f"counted {n_tokens:,} tokens across {len(tokens_ds)} rows; "
          f"{n_unique} unique token ids observed")

    out_dir = os.path.join(CKPT_MOUNT, "loss_mask")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    torch.save({
        "counts": counts,
        "tokenizer_name": BASE_MODEL,
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "split": split,
        "n_docs": len(tokens_ds),
        "n_tokens": n_tokens,
        "n_unique": n_unique,
        "vocab_size": vocab_size,
    }, out_path)
    ckpt_vol.commit()
    print(f"saved -> {out_path}")
    print(f"set TRAIN_CFG.loss_mask_reference_counts_path = {out_path!r} "
          f"to use this in training (default already points here).")


# ---------------------------------------------------------------------------
# Wiring check. Run this FIRST, before any training.
# ---------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=60 * 30)
def sanity_check():
    """Zero-init TTT + fresh LoRA (B=0) must reproduce base Qwen3-8B
    exactly. A non-tiny diff means the wiring is broken; do not train."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from model_setup import build_model
    from ttt_config import BASE_MODEL

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    ids = tokenizer(
        "Test-time training updates a subset of weights during inference. "
        * 40,
        return_tensors="pt",
    ).input_ids.cuda()

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    with torch.no_grad():
        ref = base(ids).logits
    del base
    torch.cuda.empty_cache()

    patched, _ = build_model(trainable=False)
    patched.eval()
    with torch.no_grad():
        got = patched(ids).logits

    diff = (ref - got).abs().max().item()
    print(f"max |logit diff| = {diff:.6f}")
    assert diff < 1e-2, "identity check FAILED, wiring is broken"
    print("identity check passed")


# ---------------------------------------------------------------------------
# Loss-mask diagnostic. CPU-only; does not start training.
#
#     modal run train_modal.py::diagnose_loss_mask
#     modal run train_modal.py::diagnose_loss_mask --limit-docs 400
#     modal run train_modal.py::diagnose_loss_mask --keep-fraction 0.5
#
# Use this BEFORE a real run to verify that loss_mask_keep_fraction is
# catching function words / punctuation rather than domain content words
# you care about (e.g. "model", "training", "diffusion"). The chronological
# prefix sweep surfaces date-driven drift -- a token that only enters the
# mask once newer arxiv years are included is exactly the kind of failure
# mode raw frequency masking has on a domain corpus.
# ---------------------------------------------------------------------------
@app.function(image=image, volumes=VOLUMES, secrets=SECRETS,
              timeout=60 * 30)
def diagnose_loss_mask(limit_docs: int = 0, top_k: int = 80,
                       keep_fraction: float = 0.0,
                       use_reference: bool = False):
    """Report what the content-token loss mask catches.

    Two modes:

      use_reference=False (default): build the mask from the
        date-sorted training corpus, broken into cumulative
        chronological prefixes (first 25%, 50%, 75%, 100%). This is
        the diagnostic for the in-corpus fallback path.

      use_reference=True: build the mask from the external reference
        unigram distribution at
        TRAIN_CFG.loss_mask_reference_counts_path. The reference is
        fixed, so chronological slicing collapses to a single row;
        the spot-check then reflects what training will actually use.

    Sections (both modes):
      1. Mask statistics (mask size, fraction of positions actually
         kept). The kept fraction should land near keep_fraction;
         deviation reveals discretization slack.
      2. Top-K masked tokens, decoded back to strings. If this is all
         function words / punctuation, the mask is doing its job.
      3. Spot-check table: a curated list of (function | ML-glue |
         domain-content) words, showing MASKED vs kept. Lets you see
         at a glance which categories the mask eats.

    Args:
        limit_docs: cap on training docs (in-corpus mode only).
        top_k: how many top-frequency masked tokens to decode/print.
        keep_fraction: override TRAIN_CFG.loss_mask_keep_fraction
                       (0.0 = use the value in config).
        use_reference: see above.
    """
    import torch
    from transformers import AutoTokenizer

    from ttt_config import BASE_MODEL
    from train_utils import (
        apply_protect_passes, common_mask_from_counts,
        load_reference_counts,
    )

    kf = keep_fraction if keep_fraction > 0 else TRAIN_CFG.loss_mask_keep_fraction
    print(f"loss_mask diagnostic: keep_fraction={kf:.3f}, "
          f"mode={'external-reference' if use_reference else 'in-corpus'}\n")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # tokenizer.vocab_size excludes added/special tokens; len(tokenizer)
    # is the full table the model indexes into.
    vocab_size = max(tokenizer.vocab_size, len(tokenizer))

    if use_reference:
        # External reference path: load the precomputed counts and
        # synthesize a single "snapshot" so the rest of the function
        # (section 1/2/3) runs unchanged. Per-slice chronological
        # bookkeeping is meaningless when the baseline is fixed, so
        # we collapse to one row labeled with the reference's source.
        ref_counts, ref_meta = load_reference_counts(
            TRAIN_CFG.loss_mask_reference_counts_path, vocab_size,
        )
        if ref_counts is None:
            raise RuntimeError(
                f"use_reference=True but "
                f"{TRAIN_CFG.loss_mask_reference_counts_path!r} not "
                f"found; run build_reference_counts first"
            )
        label = (f"ref:{ref_meta.get('dataset_id', '?')}/"
                 f"{ref_meta.get('dataset_config', '?')}")
        print(f"reference: {label}, "
              f"{ref_meta.get('n_tokens', 0):,} tokens across "
              f"{ref_meta.get('n_docs', 0)} rows")
        snapshots = [(label, ref_meta.get("n_docs", 0), ref_counts)]
    else:
        ds = load_token_dataset(tokenizer, limit_docs or None)
        hf_vol.commit()
        n = len(ds)
        if n < 4:
            raise RuntimeError(
                f"need at least 4 docs for quartile slicing, got {n}"
            )

        # Cumulative date-sorted prefixes. ds is the train split
        # returned by split_holdout, which excludes the newest
        # HOLDOUT_LAST_N rows; remaining rows are in arxiv-id order,
        # i.e. oldest -> newest.
        cuts = [
            ("first 25%", n // 4),
            ("first 50%", n // 2),
            ("first 75%", 3 * n // 4),
            ("full set",  n),
        ]
        cut_lookup = {k: label for label, k in cuts}

        # Stream once; snapshot running counts at each cumulative
        # boundary. Avoids re-tallying the prefix four times.
        print(f"counting tokens over {n} docs (one pass)...")
        snapshots = []   # list of (label, k_docs, counts_tensor)
        running = torch.zeros(vocab_size, dtype=torch.int64)
        for i in range(n):
            ids_list = ds[i]["input_ids"]
            t = torch.as_tensor(ids_list, dtype=torch.int64)
            if t.numel():
                if int(t.max()) >= vocab_size or int(t.min()) < 0:
                    raise ValueError(
                        f"OOR token at doc {i}: min={int(t.min())}, "
                        f"max={int(t.max())}, vocab_size={vocab_size}"
                    )
                running += torch.bincount(t, minlength=vocab_size)
            if (i + 1) in cut_lookup:
                snapshots.append(
                    (cut_lookup[i + 1], i + 1, running.clone())
                )

    # Build the mask at each snapshot and apply BOTH protect passes so
    # the diagnostic reflects what the training loop will actually use.
    slices = []   # list of dicts
    for label, k, counts in snapshots:
        mask = common_mask_from_counts(counts, keep_fraction=kf)
        n_pre_protect = int(mask.sum())
        freed_terms, freed_numeric = apply_protect_passes(
            mask, tokenizer,
            TRAIN_CFG.loss_mask_protect_terms,
            TRAIN_CFG.loss_mask_protect_numeric,
        )
        total = int(counts.sum().item())
        n_masked = int(mask.sum())
        kept_share = (
            float((counts * (~mask).long()).sum().item() / total)
            if total else 0.0
        )
        slices.append({
            "label": label, "k": k, "tokens": total,
            "mask": mask, "counts": counts,
            "n_pre_protect": n_pre_protect,
            "freed_terms": freed_terms,
            "freed_numeric": freed_numeric,
            "n_masked": n_masked, "kept_share": kept_share,
        })

    # ------------------------------------------------- section 1: stats --
    header = ("Mask statistics (external reference)"
              if use_reference else
              "Per-slice mask statistics (post-protect passes)")
    print(f"\n=== {header} ===")
    label_w = max(20, max(len(s["label"]) for s in slices) + 1)
    print(f"  {'slice':<{label_w}} {'docs':>6} {'tokens':>14} "
          f"{'pre_protect':>11} {'free_t':>7} {'free_n':>7} "
          f"{'masked_ids':>11} {'pos_kept':>9}")
    for s in slices:
        print(f"  {s['label']:<{label_w}} {s['k']:>6} {s['tokens']:>14,} "
              f"{s['n_pre_protect']:>11} {s['freed_terms']:>7} "
              f"{s['freed_numeric']:>7} "
              f"{s['n_masked']:>11} {s['kept_share']:>8.1%}")
    print("  free_t = freed by protect-terms list, "
          "free_n = freed by protect-numeric predicate.")

    # ----------------------------------- section 2: top-K masked tokens --
    where = "external ref" if use_reference else "full set"
    print(f"\n=== Top-{top_k} masked tokens ({where}, descending freq) ===")
    print(f"  {'rank':>4}  {'id':>6}  {'count':>14}  token")
    full = slices[-1]
    masked_counts = full["counts"].clone()
    masked_counts[~full["mask"]] = 0
    top_vals, top_ids = torch.topk(masked_counts, k=min(top_k, vocab_size))
    for rank, (val, tid) in enumerate(
        zip(top_vals.tolist(), top_ids.tolist()), 1
    ):
        if val == 0:
            break
        s = tokenizer.decode([tid])
        print(f"  {rank:>4}  {tid:>6}  {val:>14,}  {s!r}")

    # ------------------------------------------- section 3: spot check --
    # Three groups so the user can see (a) the mask catches obvious glue,
    # (b) whether it overcatches ML-glue, (c) whether it spares domain
    # content. The leading-space form (' word') is the BPE convention for
    # mid-sequence words; the bare form occasionally differs in id.
    spot_terms = [
        ("function-words",
         ["the", "of", "and", "is", "we", "in", "to", "a", "for", "with"]),
        ("ML-glue (the question)",
         ["model", "training", "data", "loss", "gradient", "layer",
          "network", "learning", "weights", "function"]),
        ("domain content (should stay kept)",
         ["transformer", "diffusion", "convolution", "attention",
          "embedding", "tokenizer", "Bayesian", "Markov", "kernel",
          "VAE", "GAN", "policy", "reward"]),
    ]
    print("\n=== Spot check: mask status across cumulative slices ===")
    print("  Row 1: first-piece status per slice (legacy summary).")
    print("  Row 2: ACTUAL pieces -- decoded string and full-set mask "
          "status of each.")
    print("  A multi-piece row whose first piece is MASKED does NOT "
          "mean the full term was masked; it means BPE split the term "
          "and one piece happened to be a common id. Read row 2.")
    col_w = max(10, max(len(s["label"]) for s in slices))
    header_cells = "  ".join(f"{s['label']:>{col_w}s}" for s in slices)
    print(f"\n  {'term':<24}  {header_cells}")
    print(f"  {'-' * 24}  " + "  ".join("-" * col_w for _ in slices))
    full_mask = slices[-1]["mask"]
    for group_name, terms in spot_terms:
        print(f"  [{group_name}]")
        for term in terms:
            ids = tokenizer.encode(" " + term, add_special_tokens=False)
            if not ids:
                continue
            # Row 1: first-piece status across slices.
            cells = []
            for s in slices:
                cells.append("MASKED" if bool(s["mask"][ids[0]])
                             else "kept")
            cell_str = "  ".join(f"{c:>{col_w}s}" for c in cells)
            shown = f"' {term}' (1st id {ids[0]})"
            print(f"  {shown:<24}  {cell_str}")
            # Row 2: per-piece breakdown in the full-set mask.
            piece_parts = []
            for tid in ids:
                tid = int(tid)
                mark = "M" if bool(full_mask[tid]) else "k"
                decoded = tokenizer.decode([tid])
                piece_parts.append(f"[{tid}:{decoded!r}={mark}]")
            piece_str = " + ".join(piece_parts)
            tag = " <-- single piece" if len(ids) == 1 else \
                  f" <-- {len(ids)}-piece BPE split"
            print(f"  {'':<24}  pieces: {piece_str}{tag}")

    print("\ndone.")
    print("Reading the spot check:")
    print("  - Row 1 is the legacy 'first-piece' summary; Row 2 is the "
          "truth.")
    print("  - Single-piece rows: 'MASKED' there = the whole word's "
          "loss is dropped. If those are domain content, the "
          "protect-list should have caught them; otherwise raise "
          "keep_fraction or add the term.")
    print("  - Multi-piece rows with M+k...: BPE split the term, the "
          "first piece is a common id (e.g. ' V', ' G'). protect_token_ids "
          "deliberately skips these to avoid leak-unmasking every "
          "capital-V mid-sentence word; the trailing pieces still "
          "receive loss, so the model still trains on the rare onset "
          "+ the full tail.")


@app.local_entrypoint()
def main(limit_docs: int = 0, num_epochs: int = 0,
         grad_accum: int = 0, session: int = -1,
         single_paper: int = -1):
    train.remote(limit_docs=limit_docs, num_epochs=num_epochs,
                 grad_accum=grad_accum, session=session,
                 single_paper=single_paper)
