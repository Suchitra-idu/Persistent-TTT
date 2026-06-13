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
          grad_accum: int = 0, session: int = -1):
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
    from train_utils import grad_norms, make_session_schedule

    overrides = {}
    if num_epochs:
        overrides["num_epochs"] = num_epochs
    if grad_accum:
        overrides["grad_accum_steps"] = grad_accum
    if session in (0, 1):
        overrides["session_training"] = bool(session)
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
    steps_per_epoch = math.ceil(len(ds) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(cfg.warmup_min_steps, int(cfg.warmup_ratio * total_steps)),
        num_training_steps=total_steps,
    )
    print(f"{total_steps} optimizer steps "
          f"({len(ds)} docs x {epochs} epochs / accum {cfg.grad_accum_steps}); "
          f"session_training={cfg.session_training} "
          f"n in [{cfg.session_papers_min}, {cfg.session_papers_max}]")

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
        sessions = make_session_schedule(
            len(ds), cfg.session_papers_min, cfg.session_papers_max, rng
        )
        for session in sessions:
            reset_session_state(model)
            for pos, doc_idx in enumerate(session):
                ids = torch.tensor(
                    [ds[int(doc_idx)]["input_ids"]], device="cuda"
                )
                loss = model(input_ids=ids, labels=ids).loss

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
                            f"{micro} (doc {int(doc_idx)}, epoch {epoch})",
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
                telemetry.log({
                    "micro/step": micro,
                    "micro/paper_loss": loss.item(),
                    "micro/paper_tokens": n_tok,
                    "micro/session_pos": pos,
                    "micro/session_n": len(session),
                    "micro/state_ratio_mean": state_ratio_mean,
                })
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


@app.local_entrypoint()
def main(limit_docs: int = 0, num_epochs: int = 0,
         grad_accum: int = 0, session: int = -1):
    train.remote(limit_docs=limit_docs, num_epochs=num_epochs,
                 grad_accum=grad_accum, session=session)
