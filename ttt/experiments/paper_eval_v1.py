"""Every measurement the paper rests on, for one checkpoint, into one folder.

    modal run ttt/experiments/paper_eval_v1.py --checkpoint ttt-v1.1/step_600
    modal run ttt/experiments/paper_eval_v1.py --checkpoint lora-only/step_600 \\
        --suites regimes --flags "<the flags it was trained with>"

Pass the `--flags` the checkpoint was trained with; they are recorded in the
manifest. A checkpoint with no TTT layers runs the regimes suite only. The bundle
lands in `results/<run>/<step>/<stamp>/`, its raw rows also on the checkpoint
volume under `evidence/`. Rebuild the tables from a bundle's raw rows, no GPU:

    python -m ttt.experiments.paper_eval_v1 results/ttt-v1.1/step_600/<stamp>
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import modal

from ttt import cli
from ttt.adapters import modal_runtime
from ttt.adapters.local_storage import LocalStorage
from ttt.adapters.numpy_rng import NumpyRng
from ttt.app import (
    boundary_eval,
    context_eval,
    data_pipeline,
    dynamic_eval,
    relatedness_eval,
    eval_loop,
    evidence,
    repeat_eval,
    token_gain,
)
from ttt.core import report
from ttt.core.metrics import ALL_SOURCES
from ttt.core.stats import Estimate, Interval
from ttt.core.types import EvalRow, SliceRow
from ttt.experiments import _runtime
from ttt.experiments._runtime import Engine

app = modal.App("ttt-paper-eval-v1")
image = modal_runtime.build_image()

REGIMES = "regimes"
BOUNDARY = "boundary"
REPEAT = "repeat"
CONTEXT = "context"
RELATEDNESS = "relatedness"
DYNEVAL = "dyneval"
TOKEN_GAIN = "token_gain"
SUITES = (REGIMES, BOUNDARY, REPEAT)
# Opt-in: context costs a second, longer forward per slice; relatedness and dyneval download
# Wikipedia, and dyneval takes gradient steps, so it runs on models without TTT layers too.
KNOWN_SUITES = SUITES + (CONTEXT, RELATEDNESS, DYNEVAL, TOKEN_GAIN)
TTT_SUITES = (BOUNDARY, REPEAT, CONTEXT, RELATEDNESS, TOKEN_GAIN)

RESULTS_DIR = "results"
WIKIPEDIA = "RedPajamaWikipedia"
EVIDENCE_DIR = "evidence"
RAW_FILES = {
    "regimes": EvalRow,
    "regime_slices": SliceRow,
    "boundary": boundary_eval.ArmRow,
    "repeat": repeat_eval.RepeatItem,
    "context": context_eval.ContextRow,
    "relatedness": relatedness_eval.RelatedRow,
    "dyneval_self": relatedness_eval.RelatedRow,
    "dyneval_related": relatedness_eval.RelatedRow,
    "dyneval_tuning": dynamic_eval.TuneRow,
    "token_gain": token_gain.TokenRow,
}
DEFAULT_N_BOOT = 1000
BOOT_SEED = 0


@dataclass(frozen=True)
class Plan:
    suites: tuple[str, ...] = SUITES
    n_docs: int = 20
    n_slices: int = 8
    n_chain: int = 5
    n_repeats: int = 10
    n_related: int = 200
    n_tune: int = 40
    n_wiki_shards: int = 2
    # Fresh pass (plan/PREREGISTRATION.md); the defaults reproduce the development cases.
    wiki_first_shard: int = 0
    wiki_case_seed: int = BOOT_SEED
    unique_related: bool = False
    n_generic: int = 0
    in_context: bool = False
    chain_lengths: str = ""
    exclude_holdout_seed: int = -1
    overlap_seeds: str = ""
    cases_file: str = ""
    target_tokens: int = 0
    # Sharding dyneval so a preemption costs one shard: a settings subset to tune,
    # fixed per-objective settings to measure instead of tuning, a slice of cases.
    dyneval_grid: str = ""
    dyneval_fixed: str = ""
    case_slice: str = ""

    def __post_init__(self) -> None:
        unknown = set(self.suites) - set(KNOWN_SUITES)
        if unknown:
            raise ValueError(f"unknown suites {sorted(unknown)}; expected some of {list(KNOWN_SUITES)}")
        object.__setattr__(self, "suites", tuple(self.suites))


@dataclass(frozen=True)
class Analysis:
    regimes: tuple[Estimate, ...] = ()
    boundary: tuple[Estimate, ...] = ()
    repeat_curve: tuple[evidence.RepeatPoint, ...] = ()
    repeat_trend: tuple[Estimate, ...] = ()
    context: tuple[Estimate, ...] = ()
    relatedness: tuple[Estimate, ...] = ()
    dyneval: tuple[Estimate, ...] = ()
    dyneval_settings: Mapping[str, str] = field(default_factory=dict)
    carry_share: Interval | None = None


def run(
    resolved: cli.Resolved,
    *,
    engine: Engine,
    source,
    plan: Plan,
    announce: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Raw per-document rows per suite, plus what the holdout actually held."""
    resolved = _with_docs(resolved, max(plan.n_docs, plan.n_chain))
    by_source = _holdout_by_source(resolved, engine, source, plan, announce)
    eval_docs = tuple(d for s in sorted(by_source) for d in by_source[s][: plan.n_docs])
    announce(f"{len(eval_docs)} documents across {len(by_source)} sources")

    rows: dict[str, list] = {}
    ports = dict(compute=engine.compute, fast_weights=engine.fast_weights)
    suites = plan.suites
    layers = engine.resolved.ttt.layer_indices
    if layers == () and set(suites) & set(TTT_SUITES):
        announce(f"no TTT layers in this checkpoint; skipping {list(TTT_SUITES)}")
        suites = tuple(s for s in suites if s not in TTT_SUITES)

    if REGIMES in suites:
        announce("regimes ...")
        measured = eval_loop.evaluate(
            docs=eval_docs, n_slices=plan.n_slices,
            **ports,
        )
        rows["regimes"] = evidence.records(measured.rows)
        rows["regime_slices"] = evidence.records(measured.slices)
    if BOUNDARY in suites:
        announce("boundary ...")
        chains = {s: docs[: plan.n_chain] for s, docs in by_source.items()}
        rows["boundary"] = evidence.records(
            boundary_eval.measure(chains=chains, n_slices=plan.n_slices, **ports)
        )
    if REPEAT in suites:
        announce("repeat ...")
        rows["repeat"] = evidence.records(
            repeat_eval.measure(docs=eval_docs, n_repeats=plan.n_repeats, **ports)
        )
    if CONTEXT in suites:
        announce("context ...")
        rows["context"] = evidence.records(
            context_eval.measure(docs=eval_docs, n_slices=plan.n_slices, **ports)
        )
    if {RELATEDNESS, DYNEVAL, TOKEN_GAIN} & set(suites):
        announce("wikipedia cases ...")
        from ttt.experiments import _wiki_relatedness

        n_tune = plan.n_tune if DYNEVAL in suites else 0
        chain_lengths = tuple(int(k) for k in plan.chain_lengths.split(",") if k.strip())
        cases, generic_blocks = _cached_cases(
            engine, plan, announce,
            lambda: _wiki_relatedness.build_cases(
                n_targets=plan.n_related + n_tune,
                n_shards=plan.n_wiki_shards,
                tokenizer=engine.tokenizer,
                other_docs=[d for s, docs in sorted(by_source.items()) if s != WIKIPEDIA for d in docs],
                rng=engine.rng(plan.wiki_case_seed),
                announce=announce,
                first_shard=plan.wiki_first_shard,
                unique_related=plan.unique_related,
                n_generic=plan.n_generic,
                n_chain=max(chain_lengths, default=0),
                seen=_training_shingles(resolved, engine, source, plan, announce),
            ),
        )
        # Selection is sequential, so the first n_related are the relatedness suite's cases.
        reported, tuning = cases[: plan.n_related], cases[plan.n_related :]
        if plan.target_tokens:
            reported = [dataclasses.replace(c, target=c.target[: plan.target_tokens]) for c in reported]
        generic = None
        if generic_blocks and {RELATEDNESS, TOKEN_GAIN} & set(suites):
            announce(f"generic carry from {len(generic_blocks)} articles ...")
            generic = _carry_mean(generic_blocks, engine)
        if RELATEDNESS in suites:
            announce("relatedness ...")
            rows["relatedness"] = evidence.records(
                relatedness_eval.measure(
                    cases=reported, generic=generic, in_context=plan.in_context,
                    chain_lengths=chain_lengths, **ports,
                )
            )
        if TOKEN_GAIN in suites:
            announce("token gain ...")
            rows["token_gain"] = evidence.records(token_gain.measure(
                cases=reported, generic=generic, in_context=plan.in_context, **ports
            ))
        if DYNEVAL in suites:
            from ttt.experiments._dynamic_adapter import LoraAdapter

            adapter = LoraAdapter(engine.model)
            fixed = dynamic_eval.parse_objectives(plan.dyneval_fixed)
            if not fixed:
                settings = dynamic_eval.parse_settings(plan.dyneval_grid) or dynamic_eval.grid()
                announce(f"dyneval: tuning {len(settings)} settings on {len(tuning)} cases, "
                         f"{adapter.n_parameters:,d} LoRA parameters")
                tune_rows = dynamic_eval.tune(
                    cases=tuning, compute=engine.compute, adapter=adapter, settings=settings
                )
                rows["dyneval_tuning"] = evidence.records(tune_rows)
                if not plan.dyneval_grid:
                    fixed = {o: dynamic_eval.best(tune_rows, o) for o in dynamic_eval.OBJECTIVES}
            start, _, stop = plan.case_slice.partition(":")
            measured = reported[int(start or 0) : int(stop) if stop else None]
            for objective, setting in fixed.items():
                announce(f"dyneval: {objective} with {setting.label} on {len(measured)} cases")
                rows[f"dyneval_{objective}"] = evidence.records(
                    dynamic_eval.measure(cases=measured, compute=engine.compute, adapter=adapter, setting=setting)
                )
    return {
        "rows": rows,
        "suites": list(suites),
        "docs_per_source": {s: min(len(d), plan.n_docs) for s, d in sorted(by_source.items())},
        "ttt_layers": None if layers is None else list(layers),
        "describe": dict(resolved.describe()),
    }


def _holdout_by_source(resolved, engine, source, plan, announce) -> dict[str, list]:
    """Per-source holdout documents; with `exclude_holdout_seed`, documents that
    seed would pick are removed, so a fresh seed never repeats them."""
    def draw(seed: int, n_per_source: int):
        cfg = dataclasses.replace(resolved.train, eval_n_docs_per_source=n_per_source)
        held = data_pipeline.holdout(
            source=source, spec=resolved.spec, cfg=cfg, rng=engine.rng(seed),
            tokenizer=engine.tokenizer,
        )
        for shortfall in held.shortfalls:
            announce(f"  {shortfall.source}: {shortfall.got} of {shortfall.wanted} held out")
        return held.docs

    wanted = resolved.train.eval_n_docs_per_source
    if plan.exclude_holdout_seed < 0:
        docs = draw(resolved.train.eval_holdout_seed, wanted)
    else:
        old = {d.token_ids for d in draw(plan.exclude_holdout_seed, wanted)}
        fresh = [d for d in draw(resolved.train.eval_holdout_seed, 2 * wanted) if d.token_ids not in old]
        announce(f"  excluding {len(old)} documents of holdout seed {plan.exclude_holdout_seed}")
        docs = fresh
    by_source: dict[str, list] = {}
    for doc in docs:
        if len(by_source.setdefault(doc.source, [])) < wanted:
            by_source[doc.source].append(doc)
    return by_source


def _training_shingles(resolved, engine, source, plan, announce) -> set[str]:
    """Shingles of the Wikipedia training documents each listed seed selects."""
    seeds = [int(x) for x in plan.overlap_seeds.split(",") if x.strip()]
    if not seeds:
        return set()
    if not resolved.limit_docs:
        raise ValueError("overlap_seeds needs the training flags, including limit_docs")
    seen: set[str] = set()
    for seed in seeds:
        docs = data_pipeline.documents(
            data_pipeline.load(
                source=source, spec=resolved.spec, cfg=resolved.train, rng=engine.rng(seed),
                tokenizer=engine.tokenizer, limit_docs=resolved.limit_docs,
                weights=resolved.weights,
            )
        )
        wiki = [d for d in docs if d.source == WIKIPEDIA]
        for doc in wiki:
            seen |= relatedness_eval.shingles(engine.tokenizer.decode(list(doc.token_ids[:700])))
        announce(f"  training overlap: seed {seed}, {len(wiki)} Wikipedia documents")
    return seen


def _cached_cases(engine, plan, announce, build):
    """Built once and saved, so every model is scored on the identical cases."""
    storage = engine.storage
    path = f"{EVIDENCE_DIR}/cases/{plan.cases_file}.json" if plan.cases_file else ""
    if path and storage is not None and storage.exists(path):
        saved = evidence.load_json(storage, path)
        _check_case_plan(saved.get("plan", {}), plan, path)
        announce(f"  cases loaded from {path}")
        return [_case_from(c) for c in saved["cases"]], [tuple(b) for b in saved["generic"]]
    cases, _, pools = build()
    if path and storage is not None:
        evidence.save_json(storage, path, {
            "cases": [dataclasses.asdict(c) for c in cases],
            "generic": [list(b) for b in pools.generic],
            "excluded_for_overlap": pools.excluded,
            "plan": dataclasses.asdict(plan),
        })
        storage.commit()
        announce(f"  cases saved to {path}")
    return cases, list(pools.generic)


CASE_SETTINGS = ("n_related", "n_tune", "n_wiki_shards", "wiki_first_shard", "wiki_case_seed",
                 "unique_related", "n_generic", "overlap_seeds", "exclude_holdout_seed")


def _check_case_plan(saved: Mapping[str, Any], plan: Plan, path: str) -> None:
    """The settings that decide which articles become cases must match the saved file."""
    current = dataclasses.asdict(plan)
    changed = [k for k in CASE_SETTINGS if k in saved and saved[k] != current[k]]
    if changed:
        raise ValueError(
            f"{path} was built with different case settings {changed}; "
            "use another cases_file or the saved settings"
        )


def _case_from(saved: Mapping[str, Any]) -> relatedness_eval.Case:
    return relatedness_eval.Case(
        index=saved["index"],
        target_title=saved["target_title"],
        target=tuple(saved["target"]),
        primes={k: tuple(v) for k, v in saved["primes"].items()},
        prime_titles=dict(saved["prime_titles"]),
        chain=tuple(tuple(b) for b in saved.get("chain", ())),
    )


def _carry_mean(blocks, engine):
    fast_weights = engine.fast_weights
    fast_weights.set_mode(evolve=True, session=True)
    states = []
    for block in blocks:
        fast_weights.reset_carry()
        engine.compute.eval_loss(block)
        fast_weights.advance_carry()
        states.append(fast_weights.snapshot(to_cpu=True))
    fast_weights.reset_carry()
    return relatedness_eval.generic_carry(states)


def _with_docs(resolved: cli.Resolved, n_docs: int) -> cli.Resolved:
    if resolved.train.eval_n_docs_per_source >= n_docs:
        return resolved
    return dataclasses.replace(
        resolved, train=dataclasses.replace(resolved.train, eval_n_docs_per_source=n_docs)
    )


def analyse(rows: Mapping[str, list], *, n_boot: int = DEFAULT_N_BOOT, seed: int = BOOT_SEED) -> Analysis:
    rng = NumpyRng(seed)
    typed = {
        name: evidence.rows_from(RAW_FILES[name], items) for name, items in rows.items() if name in RAW_FILES
    }
    kw = dict(rng=rng, n_boot=n_boot)
    regimes = evidence.regime_estimates(typed.get("regimes", ()), **kw)
    boundary = evidence.boundary_estimates(typed.get("boundary", ()), **kw)
    return Analysis(
        regimes=regimes,
        boundary=boundary,
        repeat_curve=evidence.repeat_curve(typed.get("repeat", ()), **kw),
        repeat_trend=evidence.repeat_trend(typed.get("repeat", ()), **kw),
        context=context_eval.estimates(typed.get("context", ()), **kw),
        relatedness=relatedness_eval.estimates(typed.get("relatedness", ()), **kw),
        dyneval=tuple(
            dataclasses.replace(e, measure=f"dyneval_{objective}_{e.measure}")
            for objective in dynamic_eval.OBJECTIVES
            for e in relatedness_eval.estimates(typed.get(f"dyneval_{objective}", ()), **kw)
        ),
        dyneval_settings={
            objective: dynamic_eval.best(typed["dyneval_tuning"], objective).label
            for objective in dynamic_eval.OBJECTIVES
            if typed.get("dyneval_tuning")
        },
        carry_share=(
            context_eval.carry_share(typed["context"], **kw) if typed.get("context") else None
        ),
    )


def write_bundle(
    folder: Path,
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    announce: Callable[[str], None] = print,
) -> Analysis:
    storage = LocalStorage(folder)
    for name, items in payload["rows"].items():
        evidence.save_json(storage, f"raw/{name}.json", items)
    evidence.save_json(storage, "manifest.json", {**manifest, "n_boot": n_boot, "boot_seed": BOOT_SEED})
    return render_bundle(folder, payload["rows"], n_boot=n_boot, announce=announce)


def render_bundle(
    folder: Path,
    rows: Mapping[str, list],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    announce: Callable[[str], None] = print,
) -> Analysis:
    """Tables from raw rows; overwrites what a previous render wrote."""
    analysis = analyse(rows, n_boot=n_boot)
    storage = LocalStorage(folder)
    tables = {
        "regime_gaps": analysis.regimes,
        "boundary_gaps": analysis.boundary,
        "repeat_trend": analysis.repeat_trend,
        "context_gaps": analysis.context,
        "relatedness": analysis.relatedness,
        "dyneval": analysis.dyneval,
    }
    for name, estimates in tables.items():
        if estimates:
            write_estimates(storage, f"tables/{name}", estimates)
    if analysis.repeat_curve:
        storage.write_bytes("tables/repeat_curve.csv", _repeat_csv(analysis.repeat_curve).encode())
    storage.write_bytes("SUMMARY.md", summary(analysis).encode())
    announce(f"bundle -> {folder}")
    return analysis


def write_estimates(storage, stem: str, estimates) -> None:
    storage.write_bytes(f"{stem}.md", (report.render(report.estimate_table(estimates)) + "\n").encode())
    storage.write_bytes(f"{stem}.csv", estimates_csv(estimates).encode())


def estimates_csv(estimates) -> str:
    lines = ["measure,source,n_docs,delta_ppl,ci_low,ci_high,excludes_zero,reference_ppl,relative,win_rate"]
    lines += [
        f"{e.measure},{e.source},{e.n_docs},{e.interval.point:.6g},{e.interval.low:.6g},"
        f"{e.interval.high:.6g},{int(e.interval.excludes_zero)},{e.reference:.6g},"
        f"{e.relative:.6g},{e.win_rate:.4f}"
        for e in estimates
    ]
    return "\n".join(lines) + "\n"


def _repeat_csv(points) -> str:
    lines = [
        "source,repeat,n_docs,delta_ppl,ci_low,ci_high,"
        "state_ratio_q25,state_ratio_median,state_ratio_q75,gate_q25,gate_median,gate_q75"
    ]
    lines += [
        f"{p.source},{p.repeat},{p.n_docs},{p.gap.point:.6g},{p.gap.low:.6g},{p.gap.high:.6g},"
        f"{p.state_ratio.low:.6g},{p.state_ratio.median:.6g},{p.state_ratio.high:.6g},"
        f"{p.gate_mean.low:.6g},{p.gate_mean.median:.6g},{p.gate_mean.high:.6g}"
        for p in points
    ]
    return "\n".join(lines) + "\n"


def summary(analysis: Analysis) -> str:
    """The pooled headline per measure; per-source detail lives in tables/."""
    pooled = [
        e
        for e in analysis.regimes + analysis.boundary + analysis.repeat_trend + analysis.context + analysis.relatedness + analysis.dyneval
        if e.source == ALL_SOURCES
    ]
    lines = ["# Evidence bundle", "", "## Pooled estimates (95% paired bootstrap over documents)", ""]
    lines += ["```", report.render(report.estimate_table(pooled)), "```", ""]
    if analysis.carry_share is not None:
        share = analysis.carry_share
        lines += [
            "## Carry against full context",
            "",
            f"The carried state recovers {100 * share.point:.1f}% "
            f"[{100 * share.low:.1f}%, {100 * share.high:.1f}%] of the gain from placing the "
            "previous slice in attention.",
            "",
        ]
    if analysis.dyneval_settings:
        lines += ["## Dynamic evaluation", "", "Settings chosen on the tuning cases:", ""]
        lines += [f"- tuned for {o}: {label}" for o, label in analysis.dyneval_settings.items()]
        lines.append("")
    return "\n".join(lines) + "\n"


def bundle_parts(checkpoint: str, run_name: str) -> tuple[str, str]:
    """(run, step) for the folder name; a bare `step_N` belongs to `run_name`."""
    path = cli.resume_path(checkpoint, run_name)
    run, _, step = path.rpartition("/")
    return run or run_name, step


def git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


@app.function(
    image=image,
    gpu=modal_runtime.GPU,
    volumes={
        modal_runtime.CKPT_MOUNT: modal_runtime.checkpoint_volume(),
        modal_runtime.HF_CACHE_MOUNT: modal_runtime.cache_volume(),
    },
    secrets=modal_runtime.secrets(),
    timeout=4 * 60 * 60,
)
@modal_runtime.caching
def paper_eval(checkpoint: str, bundle: str, plan: dict, **flags) -> dict[str, Any]:
    from ttt.adapters.hf_data_source import HfDataSource
    from ttt.adapters.torch_compute import EvalCompute

    resolved = cli.from_flags(**{**cli.env_defaults(), **flags, "resume_from": checkpoint})
    storage = modal_runtime.checkpoint_storage()
    engine = _runtime.build(
        resolved, storage=storage, root=modal_runtime.CKPT_MOUNT, trainable=False
    )
    engine.compute = EvalCompute(engine.model)
    payload = run(resolved, engine=engine, source=HfDataSource(), plan=Plan(**plan))
    for name, items in payload["rows"].items():
        evidence.save_json(storage, f"{EVIDENCE_DIR}/{bundle}/raw/{name}.json", items)
    storage.commit()
    return payload


@app.local_entrypoint()
def main(
    checkpoint: str = "",
    suites: str = ",".join(SUITES),
    n_docs: int = Plan.n_docs,
    n_slices: int = Plan.n_slices,
    n_chain: int = Plan.n_chain,
    n_repeats: int = Plan.n_repeats,
    n_related: int = Plan.n_related,
    n_tune: int = Plan.n_tune,
    dyneval_grid: str = "",
    dyneval_fixed: str = "",
    case_slice: str = "",
    tag: str = "",
    n_wiki_shards: int = Plan.n_wiki_shards,
    wiki_first_shard: int = Plan.wiki_first_shard,
    wiki_case_seed: int = Plan.wiki_case_seed,
    unique_related: bool = Plan.unique_related,
    n_generic: int = Plan.n_generic,
    in_context: bool = Plan.in_context,
    chain_lengths: str = Plan.chain_lengths,
    exclude_holdout_seed: int = Plan.exclude_holdout_seed,
    overlap_seeds: str = Plan.overlap_seeds,
    cases_file: str = Plan.cases_file,
    target_tokens: int = Plan.target_tokens,
    n_boot: int = DEFAULT_N_BOOT,
    out_dir: str = RESULTS_DIR,
    flags: str = "",
):
    given = dict(locals())
    if not checkpoint:
        raise SystemExit("--checkpoint is required, e.g. --checkpoint ttt-v1.1/step_600")
    parsed = cli.parse_flags(flags)
    resolved = cli.from_flags(**{**cli.env_defaults(), **parsed})
    # Modal builds the CLI from the signature, so every Plan field is a parameter above.
    plan = Plan(**{
        **{f.name: given[f.name] for f in dataclasses.fields(Plan) if f.name != "suites"},
        "suites": tuple(s.strip() for s in suites.split(",") if s.strip()),
    })
    run_part, step = bundle_parts(checkpoint, resolved.train.run_name)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bundle = f"{run_part}/{step}/{stamp}{'-' + tag if tag else ''}"

    payload = paper_eval.remote(
        checkpoint=checkpoint, bundle=bundle, plan=dataclasses.asdict(plan), **parsed
    )
    manifest = {
        "checkpoint": checkpoint,
        "bundle": bundle,
        "invocation": cli.invocation(),
        "flags": flags,
        "plan": dataclasses.asdict(plan),
        "git": git_state(),
        **{k: v for k, v in payload.items() if k != "rows"},
    }
    write_bundle(Path(out_dir) / bundle, payload, manifest, n_boot=n_boot)


def rerender(folder: str, n_boot: int | None = None) -> Analysis:
    """`n_boot` defaults to the bundle's own, so a re-render reproduces its intervals."""
    root = Path(folder)
    storage = LocalStorage(root)
    if n_boot is None:
        n_boot = evidence.load_json(storage, "manifest.json").get("n_boot") or DEFAULT_N_BOOT
    rows = {
        Path(path).stem: evidence.load_json(storage, path)
        for path in storage.listdir("raw")
        if path.endswith(".json")
    }
    return render_bundle(root, rows, n_boot=n_boot)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m ttt.experiments.paper_eval_v1 <bundle folder> [n_boot]")
    rerender(sys.argv[1], *(int(a) for a in sys.argv[2:3]))
