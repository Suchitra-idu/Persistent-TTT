"""A tiny synthetic multi-source spec, so smoke and reproducibility tests never
download anything. Same shape as the real corpus: `meta` struct, three sources.
"""

from __future__ import annotations

from ttt.core.config.dataset import DatasetSpec
from ttt.extensions.datasets._registry import register

SOURCES = ("FixtureProse", "FixtureCode", "FixtureMath")

FIXTURE = register(
    DatasetSpec(
        name="fixture",
        source="_fixture",
        text_column="text",
        source_meta_column="meta",
        source_meta_key="redpajama_set_name",
        include_sources=SOURCES,
        holdout_last_n=4,
    )
)
