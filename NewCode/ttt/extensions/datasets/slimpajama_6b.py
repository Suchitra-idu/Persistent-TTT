"""DKYoon/SlimPajama-6B — the corpus. The default spec (D9).

Excluding CommonCrawl drops ~54% of rows and leaves ~2.5M documents across six
domains, which is what makes a per-domain TTT gap measurable at 0.6B.
"""

from __future__ import annotations

from ttt.core.config.dataset import DatasetSpec
from ttt.extensions.datasets._registry import register

SOURCES = (
    "RedPajamaC4",
    "RedPajamaGithub",
    "RedPajamaBook",
    "RedPajamaArXiv",
    "RedPajamaWikipedia",
    "RedPajamaStackExchange",
)

SLIMPAJAMA_6B = register(
    DatasetSpec(
        name="slimpajama-6b",
        source="DKYoon/SlimPajama-6B",
        text_column="text",
        source_meta_column="meta",
        source_meta_key="redpajama_set_name",
        include_sources=SOURCES,
        # Without it, --limit-docs takes the raw ~77%-C4 head and the gap
        # signal is web text. "none" opts out.
        default_source_preset="slim-research",
        # Wide enough that per-source eval sampling still finds Books/ArXiv
        # rows, which are ~0.1% and ~0.9% of this subsample.
        holdout_last_n=5000,
    )
)
