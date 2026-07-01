"""
Modal app for In-Place TTT continual pretraining on Qwen3.

Usage: modal run --detach train_modal.py::train
"""

import dataclasses
import math
import os
import time

import modal

from ttt_config import (
    BASE_MODEL, CKPT_MOUNT, CKPT_VOLUME_NAME, HF_CACHE_MOUNT,
    HF_CACHE_VOLUME_NAME, TEXT_COLUMN, TOKENS_EST_COLUMN, TRAIN_CFG, TTT_CFG,
)

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
    # flash-attn wheel matches torch 2.8 + cu12 + cp311 + cxx11abiTRUE -- avoids nvcc in the base image.
    .pip_install(
        "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
    )
    .env({"HF_HOME": HF_CACHE_MOUNT})
    .add_local_python_source("ttt_config", "inplace_ttt", "ttt_wiring",
                             "model_setup", "data_utils", "observability",
                             "train_utils")
)

ckpt_vol = modal.Volume.from_name(CKPT_VOLUME_NAME, create_if_missing=True)
hf_vol = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

VOLUMES = {CKPT_MOUNT: ckpt_vol, HF_CACHE_MOUNT: hf_vol}
GPU = "H100"

SECRETS = [modal.Secret.from_name("wandb"), modal.Secret.from_name("huggingface")]


def load_token_dataset(tokenizer, limit_docs: int | None):
    from data_utils import open_dataset, split_holdout

    cfg = TRAIN_CFG
    ds, _ = split_holdout(open_dataset())

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


def _apply_cli_overrides(num_epochs, grad_accum, session, single_paper):
    overrides = {}
    if num_epochs:
        overrides["num_epochs"] = num_epochs
    if grad_accum:
        overrides["grad_accum_steps"] = grad_accum
    if session in (0, 1):
        overrides["session_training"] = bool(session)
    if single_paper in (0, 1):
        overrides["single_paper_sessions"] = bool(single_paper)
    return dataclasses.replace(TRAIN_CFG, **overrides) if overrides else TRAIN_CFG


def _resolve_resume(resume_from: str, run_name: str):
    """resume_from accepts 'step_<n>' (same-run) or '<other_run>/step_<n>' (cross-run).
    Returns (adapter_path, ttt_ckpt_path) or (None, None) if resume_from is empty.
    Optimizer momentum is NOT preserved across resume."""
    if not resume_from:
        return None, None
    if "/" in resume_from:
        resume_dir = os.path.join(CKPT_MOUNT, resume_from)
    else:
        resume_dir = os.path.join(CKPT_MOUNT, run_name, resume_from)
    adapter_path = os.path.join(resume_dir, "adapter")
    ttt_ckpt_path = os.path.join(resume_dir, "ttt_params.pt")
    if not (os.path.exists(adapter_path) and os.path.exists(ttt_ckpt_path)):
        raise FileNotFoundError(
            f"resume checkpoint incomplete at {resume_dir}: "
            f"need both adapter/ and ttt_params.pt"
        )
    print(f"resuming from {resume_dir}")
    return adapter_path, ttt_ckpt_path


def _setup_loss_mask(cfg, ds, tokenizer, vocab_size):
    """Build the content-token loss mask and move it to GPU. Returns None if disabled."""
    from train_utils import (
        apply_protect_passes, common_mask_from_counts, count_unigrams,
        load_reference_counts,
    )

    if not cfg.loss_mask_enabled:
        return None

    ref_counts, ref_meta = load_reference_counts(
        cfg.loss_mask_reference_counts_path, vocab_size,
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
        counts = count_unigrams(
            (ex["input_ids"] for ex in ds), vocab_size=vocab_size,
        )
        common_mask = common_mask_from_counts(
            counts, keep_fraction=cfg.loss_mask_keep_fraction,
        )
    n_before = int(common_mask.sum())
    freed = apply_protect_passes(
        common_mask, tokenizer,
        cfg.loss_mask_protect_terms,
        cfg.loss_mask_protect_numeric,
        cfg.loss_mask_protect_symbols,
    )
    common_mask = common_mask.cuda()
    print(f"loss mask: {int(common_mask.sum())}/{vocab_size} token ids "
          f"masked (kf={cfg.loss_mask_keep_fraction:.2f}, "
          f"protect-terms freed {freed[0]}, "
          f"protect-numeric freed {freed[1]}, "
          f"protect-symbols freed {freed[2]}, "
          f"initial {n_before}); "
          f"first_tokens={cfg.loss_mask_first_tokens} at paper-start items")
    return common_mask


def _make_epoch_sessions(cfg, num_docs, doc_lengths, rng):
    """Build the session schedule for one epoch. Dispatches on
    cfg.single_paper_sessions."""
    from train_utils import make_slice_sessions

    if cfg.single_paper_sessions:
        return make_slice_sessions(
            num_docs, doc_lengths, rng,
            session_papers=(1, 1), slice_prob=1.0,
            slice_range=(cfg.single_paper_slices_min,
                         cfg.single_paper_slices_max),
            min_slice_tokens=cfg.slice_min_tokens,
        )
    return make_slice_sessions(
        num_docs, doc_lengths, rng,
        session_papers=(cfg.session_papers_min, cfg.session_papers_max),
        slice_prob=cfg.slice_prob,
        slice_range=(cfg.slice_min, cfg.slice_max),
        min_slice_tokens=cfg.slice_min_tokens,
    )


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS,
              timeout=60 * 60 * 24)
def train(limit_docs: int = 0, num_epochs: int = 0,
          grad_accum: int = 0, session: int = -1,
          single_paper: int = -1, resume_from: str = ""):
    import numpy as np
    import torch
    from transformers import get_cosine_schedule_with_warmup

    from inplace_ttt import (
        advance_session_state, gate_reg_term, iter_ttt_modules,
        mean_state_ratio, reset_session_state, state_norms,
    )
    from ttt_wiring import build_param_groups
    from model_setup import build_model
    from observability import Telemetry, gpu_stats, param_health
    from train_utils import apply_loss_mask, expected_items_per_doc

    cfg = _apply_cli_overrides(num_epochs, grad_accum, session, single_paper)
    epochs = cfg.num_epochs
    torch.manual_seed(cfg.seed)

    adapter_path, ttt_ckpt_path = _resolve_resume(resume_from, cfg.run_name)
    model, tokenizer = build_model(
        adapter_path=adapter_path, ttt_ckpt_path=ttt_ckpt_path,
        trainable=True,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # enable_input_require_grads required so checkpointed segments get gradients
    # from frozen embeddings -- classic PEFT + checkpointing trap.
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    for m in iter_ttt_modules(model):
        m.session_mode = cfg.session_training

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

    from dataclasses import asdict
    telemetry = Telemetry(
        enabled=cfg.wandb_enabled, project=cfg.wandb_project,
        run_name=cfg.run_name, job_type="train",
        config={**asdict(cfg), **asdict(TTT_CFG),
                "base_model": BASE_MODEL,
                "num_layers": model.config.num_hidden_layers,
                "trainable_params_M": n_train / 1e6},
    )
    ttt_modules = list(iter_ttt_modules(model))
    # Snapshot W_down for drift tracking; bf16 clone, ~600MB on GPU.
    wdown_init = [p.detach().clone() for p in named_groups["wdown"]]

    # NF4 quantization is incompatible with fully training W_down -- do not add QLoRA here.
    import bitsandbytes as bnb
    optimizer = bnb.optim.PagedAdamW8bit(optim_groups, betas=(0.9, 0.95))

    ds = load_token_dataset(tokenizer, limit_docs or None)
    hf_vol.commit()

    eval_papers = []
    if cfg.eval_every > 0:
        eval_papers = fetch_holdout_papers_ids(
            tokenizer, cfg.eval_n_papers, cfg.eval_holdout_seed
        )
        if not eval_papers:
            print("no holdout papers available, in-loop eval disabled")
        else:
            lens = ", ".join(str(len(p)) for p in eval_papers)
            print(f"in-loop eval ON, {len(eval_papers)} papers ({lens} tokens), "
                  f"{cfg.eval_n_slices} slices each, every {cfg.eval_every} steps")
    doc_lengths = [len(ex["input_ids"]) for ex in ds]

    common_mask = _setup_loss_mask(cfg, ds, tokenizer, model.config.vocab_size)
    if cfg.single_paper_sessions:
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

    run_dir = os.path.join(CKPT_MOUNT, cfg.run_name)
    rng = np.random.default_rng(cfg.seed)
    step, micro, t0 = 0, 0, time.time()
    running, step_loss = 0.0, 0.0
    window_tokens, total_tokens, sessions_done, nonfinite = 0, 0, 0, 0

    for epoch in range(epochs):
        for session in _make_epoch_sessions(cfg, len(ds), doc_lengths, rng):
            reset_session_state(model)
            for pos, item in enumerate(session):
                full_ids = ds[item.doc_idx]["input_ids"]
                ids = torch.tensor(
                    [full_ids[item.start:item.end]], device="cuda"
                )
                # Leading-token mask applies ONLY at paper-start items; mid-paper
                # slices (start > 0) are real content, not shared boilerplate.
                first_n = (
                    cfg.loss_mask_first_tokens if item.start == 0 else 0
                )
                labels = apply_loss_mask(ids, common_mask,
                                         first_tokens=first_n)
                loss = model(input_ids=ids, labels=labels).loss
                if TTT_CFG.output_gate and TTT_CFG.gate_reg_weight > 0:
                    loss = loss + TTT_CFG.gate_reg_weight * gate_reg_term(model)

                # Anomaly guard: nonfinite loss must not poison accum gradients -- skip backward.
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
                    advance_session_state(model)
                    micro += 1
                    continue

                (loss / cfg.grad_accum_steps).backward()
                advance_session_state(model)
                step_loss += loss.item() / cfg.grad_accum_steps
                n_tok = ids.numel()
                window_tokens += n_tok
                total_tokens += n_tok
                sn = state_norms(model, source="session")
                state_ratio_mean = mean_state_ratio(sn)
                micro_log = {
                    "micro/step": micro,
                    "micro/paper_loss": loss.item(),
                    "micro/paper_tokens": n_tok,
                    "micro/session_pos": pos,
                    "micro/session_n": len(session),
                    "micro/state_ratio_mean": state_ratio_mean,
                }
                if common_mask is not None:
                    micro_log["micro/unmasked_token_frac"] = (
                        float((labels != -100).float().mean())
                    )
                telemetry.log(micro_log)
                micro += 1

                if micro % cfg.grad_accum_steps:
                    continue

                # L2 grad norms per param group, before clipping.
                norms = {
                    name: float(torch.as_tensor(sum(
                        p.grad.detach().float().pow(2).sum()
                        for p in params if p.grad is not None
                    )).sqrt())
                    for name, params in named_groups.items()
                }
                total_norm = torch.nn.utils.clip_grad_norm_(
                    (p for g in optim_groups for p in g["params"]),
                    cfg.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                lrs = scheduler.get_last_lr()
                dt = time.time() - t0
                sn = state_norms(model, source="session")
                session_out = {f"session/state_ratio_L{i}": v
                               for i, v in sn.items()}
                if sn:
                    session_out["session/state_ratio_mean"] = mean_state_ratio(sn)
                    session_out["session/state_ratio_max"] = max(sn.values())
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
                    **session_out,
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
                    running = 0.0

                if step % cfg.save_every == 0 or step == total_steps:
                    save_checkpoint(model, run_dir, step)
                    ckpt_vol.commit()

                if eval_papers and step % cfg.eval_every == 0:
                    eval_metrics = run_holdout_eval(
                        model, eval_papers, cfg.eval_n_slices,
                        train_session_mode=cfg.session_training,
                    )
                    telemetry.log({"train/step": step, **eval_metrics})
                    print(
                        f"  [eval, {len(eval_papers)} papers] "
                        f"carry_ppl {eval_metrics['eval/carry_ppl']:.2f} "
                        f"fresh_ppl {eval_metrics['eval/fresh_ppl']:.2f} "
                        f"gap {eval_metrics['eval/gap']:+.2f} "
                        f"state/W0 {eval_metrics['eval/state_ratio_final']:.2e}"
                    )
            sessions_done += 1

    save_checkpoint(model, run_dir, step)
    ckpt_vol.commit()
    telemetry.finish()
    print("done")


def save_checkpoint(model, run_dir: str, step: int):
    from ttt_wiring import save_ttt_state_dict

    path = os.path.join(run_dir, f"step_{step}")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(os.path.join(path, "adapter"))
    save_ttt_state_dict(model, os.path.join(path, "ttt_params.pt"), TTT_CFG)
    print(f"saved checkpoint -> {path}")


def fetch_holdout_papers_ids(tokenizer, n_papers: int, seed: int):
    import random

    from data_utils import open_dataset, split_holdout

    _, holdout = split_holdout(open_dataset())
    if len(holdout) == 0:
        return []
    rng = random.Random(seed)
    n = min(n_papers, len(holdout))
    indices = rng.sample(range(len(holdout)), n)
    return [
        tokenizer(holdout[i][TEXT_COLUMN], truncation=True,
                  max_length=TRAIN_CFG.max_seq_len).input_ids
        for i in indices
    ]


def _eval_paper(model, paper_ids, n_slices: int, evolve: bool) -> list:
    """Run one held-out paper as a single-paper session; return per-slice
    (n_tokens, ppl, state_ratio) rows."""
    import math

    import torch

    from inplace_ttt import (
        advance_session_state, iter_ttt_modules, mean_state_ratio,
        reset_session_state, state_norms,
    )
    from train_utils import equal_token_slices

    for m in iter_ttt_modules(model):
        m.ttt_evolve = evolve
        m.session_mode = True
    reset_session_state(model)
    rows = []
    for s, e in equal_token_slices(len(paper_ids), n_slices):
        ids = torch.tensor([paper_ids[s:e]], device="cuda")
        with torch.no_grad():
            loss = model(input_ids=ids, labels=ids).loss
        advance_session_state(model)
        state_ratio = mean_state_ratio(state_norms(model, source="session"))
        rows.append((e - s, math.exp(loss.item()), state_ratio))
    return rows


def _token_weighted_ppl(rows) -> float:
    import math

    total_log = sum(math.log(p) * n for n, p, _ in rows)
    total_n = sum(n for n, _, _ in rows)
    return math.exp(total_log / total_n) if total_n else float("nan")


def run_holdout_eval(model, holdout_papers, n_slices: int,
                    train_session_mode: bool) -> dict:
    """Multi-paper carry-vs-fresh perplexity eval. Snapshots and restores
    TTT module state so this is a no-op against the training loop."""
    import math

    from inplace_ttt import iter_ttt_modules

    modules = list(iter_ttt_modules(model))
    was_training = model.training
    snap = [
        (m.carried_delta.clone() if m.carried_delta is not None else None,
         m._next_carried.clone() if m._next_carried is not None else None)
        for m in modules
    ]

    per_paper = []
    model.eval()
    try:
        for paper_ids in holdout_papers:
            carry_rows = _eval_paper(model, paper_ids, n_slices, evolve=True)
            fresh_rows = _eval_paper(model, paper_ids, n_slices, evolve=False)
            per_paper.append({
                "carry_ppl": _token_weighted_ppl(carry_rows),
                "fresh_ppl": _token_weighted_ppl(fresh_rows),
                "state_ratio_final": (carry_rows[-1][2]
                                      if carry_rows else 0.0),
                "carry_rows": carry_rows,
                "fresh_rows": fresh_rows,
            })
    finally:
        if was_training:
            model.train()
        for m in modules:
            m.ttt_evolve = True
            m.session_mode = train_session_mode
        for m, (cd, nc) in zip(modules, snap):
            m.carried_delta, m._next_carried = cd, nc

    n = len(per_paper)
    carry_mean = math.exp(sum(math.log(p["carry_ppl"]) for p in per_paper) / n)
    fresh_mean = math.exp(sum(math.log(p["fresh_ppl"]) for p in per_paper) / n)
    state_mean = sum(p["state_ratio_final"] for p in per_paper) / n

    metrics = {
        "eval/carry_ppl": carry_mean,
        "eval/fresh_ppl": fresh_mean,
        "eval/gap": fresh_mean - carry_mean,        # positive => carry helps
        "eval/state_ratio_final": state_mean,
    }
    for i, p in enumerate(per_paper):
        metrics[f"eval/paper_{i}/carry_ppl"] = p["carry_ppl"]
        metrics[f"eval/paper_{i}/fresh_ppl"] = p["fresh_ppl"]
        metrics[f"eval/paper_{i}/gap"] = p["fresh_ppl"] - p["carry_ppl"]
        metrics[f"eval/paper_{i}/state_ratio_final"] = p["state_ratio_final"]
    for s_idx in range(n_slices):
        carry_logs, fresh_logs, state_vals = [], [], []
        for p in per_paper:
            if s_idx < len(p["carry_rows"]):
                carry_logs.append(math.log(p["carry_rows"][s_idx][1]))
                state_vals.append(p["carry_rows"][s_idx][2])
            if s_idx < len(p["fresh_rows"]):
                fresh_logs.append(math.log(p["fresh_rows"][s_idx][1]))
        if carry_logs:
            metrics[f"eval/carry_ppl_slice_{s_idx}"] = math.exp(
                sum(carry_logs) / len(carry_logs)
            )
            metrics[f"eval/state_ratio_slice_{s_idx}"] = (
                sum(state_vals) / len(state_vals)
            )
        if fresh_logs:
            metrics[f"eval/fresh_ppl_slice_{s_idx}"] = math.exp(
                sum(fresh_logs) / len(fresh_logs)
            )
    return metrics


@app.function(image=image, volumes=VOLUMES, secrets=SECRETS,
              timeout=60 * 60)
def build_reference_counts(
    dataset_id: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "train",
    text_column: str = "text",
    out_name: str = "reference_wikitext103.pt",
    limit_docs: int = 0,
):
    """Build a [vocab_size] unigram count tensor from a reference corpus,
    saved to /ckpt/loss_mask/<out_name>."""
    import torch
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    from train_utils import count_unigrams

    print(f"loading {dataset_id} / {dataset_config} / {split} from HF Hub")
    ds = load_dataset(dataset_id, dataset_config, split=split)
    if limit_docs:
        ds = ds.select(range(min(limit_docs, len(ds))))
    print(f"loaded {len(ds)} rows; text column = {text_column!r}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Align on model.config.vocab_size (padded) rather than tokenizer.vocab_size
    # so saved counts match what train() validates against.
    vocab_size = AutoConfig.from_pretrained(BASE_MODEL).vocab_size
    print(f"tokenizer vocab_size = {tokenizer.vocab_size}, "
          f"model.config.vocab_size = {vocab_size} (using padded)")

    # add_special_tokens=False so BOS/EOS don't pollute the unigram tally.
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


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=60 * 10)
def sanity_check():
    """Verify TTT wiring: with W_target zeroed, TTT path must contribute zero."""
    import torch
    from transformers import AutoTokenizer

    from inplace_ttt import iter_ttt_modules
    from model_setup import build_model

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Need N > chunk_size so the scan path fires (early-return at N <= C would mask bugs).
    ids = tokenizer(
        "Test-time training updates a subset of weights during inference. "
        * 20,
        return_tensors="pt",
    ).input_ids.cuda()
    print(f"input tokens: {ids.shape[1]} (chunk_size={TTT_CFG.chunk_size})")

    model, _ = build_model(trainable=False)
    model.eval()
    modules = list(iter_ttt_modules(model))
    for m in modules:
        m.stateful = False

    for m in modules:
        m.ttt_evolve = True
    with torch.no_grad():
        logits_on_real = model(ids).logits
    for m in modules:
        m.ttt_evolve = False
    with torch.no_grad():
        logits_off = model(ids).logits
    diff_real = (logits_on_real - logits_off).abs().max().item()
    print(f"max |logit diff| at REAL init (small-randn W_target) = "
          f"{diff_real:.4f}")

    for m in modules:
        m.w_target.data.zero_()
        m.ttt_evolve = True
    with torch.no_grad():
        logits_on_zero = model(ids).logits

    diff_zero = (logits_on_zero - logits_off).abs().max().item()
    print(f"max |logit diff| at ZEROED W_target = {diff_zero:.6f}")
    assert diff_zero < 1e-3, "TTT path not exact-zero at W_target=0, wiring broken"
    print("identity check passed")


@app.function(image=image, volumes=VOLUMES, secrets=SECRETS,
              timeout=60 * 30)
def diagnose_loss_mask(limit_docs: int = 0, top_k: int = 80,
                       keep_fraction: float = 0.0,
                       use_reference: bool = False):
    """Report what the content-token loss mask catches (in-corpus or reference mode)."""
    import torch
    from transformers import AutoTokenizer

    from train_utils import (
        apply_protect_passes, common_mask_from_counts,
        load_reference_counts,
    )

    kf = keep_fraction if keep_fraction > 0 else TRAIN_CFG.loss_mask_keep_fraction
    print(f"loss_mask diagnostic: keep_fraction={kf:.3f}, "
          f"mode={'external-reference' if use_reference else 'in-corpus'}\n")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # len(tokenizer) is the full table the model indexes into.
    vocab_size = max(tokenizer.vocab_size, len(tokenizer))

    if use_reference:
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

        # ds is oldest -> newest (arxiv-id order) after split_holdout.
        cuts = [
            ("first 25%", n // 4),
            ("first 50%", n // 2),
            ("first 75%", 3 * n // 4),
            ("full set",  n),
        ]
        cut_lookup = {k: label for label, k in cuts}

        print(f"counting tokens over {n} docs (one pass)...")
        snapshots = []
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

    slices = []
    for label, k, counts in snapshots:
        mask = common_mask_from_counts(counts, keep_fraction=kf)
        n_pre_protect = int(mask.sum())
        freed_terms, freed_numeric, freed_symbols = apply_protect_passes(
            mask, tokenizer,
            TRAIN_CFG.loss_mask_protect_terms,
            TRAIN_CFG.loss_mask_protect_numeric,
            TRAIN_CFG.loss_mask_protect_symbols,
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
            "freed_symbols": freed_symbols,
            "n_masked": n_masked, "kept_share": kept_share,
        })

    header = ("Mask statistics (external reference)"
              if use_reference else
              "Per-slice mask statistics (post-protect passes)")
    print(f"\n=== {header} ===")
    label_w = max(20, max(len(s["label"]) for s in slices) + 1)
    print(f"  {'slice':<{label_w}} {'docs':>6} {'tokens':>14} "
          f"{'pre_protect':>11} {'free_t':>7} {'free_n':>7} {'free_s':>7} "
          f"{'masked_ids':>11} {'pos_kept':>9}")
    for s in slices:
        print(f"  {s['label']:<{label_w}} {s['k']:>6} {s['tokens']:>14,} "
              f"{s['n_pre_protect']:>11} {s['freed_terms']:>7} "
              f"{s['freed_numeric']:>7} {s['freed_symbols']:>7} "
              f"{s['n_masked']:>11} {s['kept_share']:>8.1%}")
    print("  free_t = protect-terms, free_n = protect-numeric, "
          "free_s = protect-symbols.")

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
            cells = []
            for s in slices:
                cells.append("MASKED" if bool(s["mask"][ids[0]])
                             else "kept")
            cell_str = "  ".join(f"{c:>{col_w}s}" for c in cells)
            shown = f"' {term}' (1st id {ids[0]})"
            print(f"  {shown:<24}  {cell_str}")
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
