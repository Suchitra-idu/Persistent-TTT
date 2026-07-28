# Ring 1 map

The science that varies. Plugins behind registries; a new plugin is a new file
plus one line in the package `__init__`. May import `torch`, never
`transformers` / `peft` / `datasets` / `modal` / `wandb`.

## Strategies

A strategy turns doc lengths into one epoch of `Session`s.

| File | Is |
|---|---|
| [_registry.py](../ttt/extensions/strategies/_registry.py) | The `Strategy` protocol, `STRATEGIES`, `register`, `get` |
| [hybrid.py](../ttt/extensions/strategies/hybrid.py) | Training on SlimPajama |
| [everlasting.py](../ttt/extensions/strategies/everlasting.py) | Producing the per-source seeds |

| | hybrid | everlasting |
|---|---|---|
| session | one doc | one doc |
| items per doc | 1 if short, else 2–6 slices | always 1, the whole doc |
| `carry_scope` | `"session"` — dies at the doc boundary | `"source"` — outlives the epoch |
| what it teaches | using a non-zero `S_0` | what a good `S_0` *is* |

Hybrid's length gate is the point: docs under `carry_min_tokens` (2100) become
single-item sessions, so the model keeps seeing the `S_0 = 0` case instead of
only ever starting warm.

Four methods, all pure:

| Method | Returns |
|---|---|
| `build(doc_lengths, rng)` | the epoch's sessions; same seed, same schedule |
| `count(doc_lengths)` | total work items, without drawing cuts — this sizes the LR schedule |
| `compose(doc_lengths, sources)` | `CompositionRow`s for the boot log |
| `describe()` | one line |

`count` has to agree with `build` without running it, because the LR schedule
is sized before the first session exists. The contract suite asserts that.

## Datasets

| File | Is |
|---|---|
| [_registry.py](../ttt/extensions/datasets/_registry.py) | `DATASETS`, `register`, `get`, `DEFAULT` |
| [slimpajama_6b.py](../ttt/extensions/datasets/slimpajama_6b.py) | The corpus. 6 sources, CommonCrawl excluded |
| [fixture.py](../ttt/extensions/datasets/fixture.py) | 3 synthetic sources. Tests never download |

The spec type is core's `DatasetSpec`, not a separate protocol — pure data
needs no seam (D5). The registry's job is the name and the D9 guarantee: every
row carries a `source`, checked as a contract instead of at container start.

## Mechanism

[inplace_ttt.py](../ttt/extensions/mechanism/inplace_ttt.py) — `InPlaceTTTMLP`,
the gated-MLP replacement. It holds no math: every kernel comes from
[`core.ttt_math`](../ttt/core/ttt_math.py) and [`core.carry`](../ttt/core/carry.py).
Patching it onto a real Qwen3 is Ring 3's job.

Two forward paths over two state families:

| | `_scan_forward` (training) | `_stream_forward` (generation) |
|---|---|---|
| sees | the whole item at once | tokens as they arrive |
| state | `carried`, advanced per item | `stream.delta`, committed per chunk |
| causality | exclusive cumsum | apply, *then* buffer |
| accumulation | EMA at `carried_decay` | pure sum, bounded by the clip |

They agree numerically — `tests/extensions/test_inplace_ttt.py` asserts the
stream reproduces the scan, including across call boundaries, which is what
keeps the conv's left-context stitching honest.

Three resets, three state families:

| Call | Clears | Keeps |
|---|---|---|
| `reset_stream()` | stream delta, pending chunk, conv context | the carry |
| `reset_v_context()` | conv context only | stream delta, the carry |
| `reset_carry()` | the carry | everything streaming |

## Testing

One contract suite per registry, parametrized over `REGISTRY.values()`, so a
new plugin is tested the moment it registers. Enforced by
`tests/architecture/test_suite_completeness.py`. See [testing.md](testing.md).
