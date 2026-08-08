"""The language-transfer eval's language lists — targets `hybrid`; rationale
is in docs/experiments-map.md. TRAIN_LANGUAGES and EVAL_LANGUAGES are two
names, not one, so a future transfer run can point EVAL_LANGUAGES elsewhere."""

from __future__ import annotations

WIKI_SNAPSHOT = "20231101"

# The one place this is set — extensions/datasets/lang_transfer.py sizes
# TRAIN_LANGS.holdout_last_n off it, so the two can't drift apart the way a
# hardcoded holdout size and an independently-changed prep default would.
DEFAULT_TARGET_ROWS = 500

# (Hub language config, display name). Worst ten base-model ppl readings
# from a real `lang_scan_v1` run.
TRAIN_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("fur", "Friulian"),
    ("vec", "Venetian"),
    ("sc", "Sardinian"),
    ("szl", "Silesian"),
    ("lij", "Ligurian"),
    ("ht", "Haitian Creole"),
    ("li", "Limburgish"),
    ("scn", "Sicilian"),
    ("ilo", "Iloko"),
    ("pap", "Papiamento"),
)

EVAL_LANGUAGES: tuple[tuple[str, str], ...] = TRAIN_LANGUAGES

OVERLAP_LANGUAGES: tuple[str, ...] = tuple(
    sorted(set(code for code, _ in TRAIN_LANGUAGES) & set(code for code, _ in EVAL_LANGUAGES))
)

# `lang_scan_v1`'s scouting pool — what a future transfer or expansion run
# draws from (exclusions explained in docs/experiments-map.md).
CANDIDATE_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("awa", "Awadhi"),
    ("mai", "Maithili"),
    ("bh", "Bhojpuri"),
    ("ast", "Asturian"),
    ("oc", "Occitan"),
    ("ba", "Bashkir"),
    ("yi", "Eastern Yiddish"),
    ("tpi", "Tok Pisin"),
)

# 10% of the raw (pre-length-filter) combined train pool: enough that
# every language has real odds of appearing in train_v1's periodic holdout
# (combine() shuffles), without giving up more of the training pool than that.
TRAIN_HOLDOUT_FRACTION = 0.1
TRAIN_HOLDOUT_LAST_N = round(
    TRAIN_HOLDOUT_FRACTION * DEFAULT_TARGET_ROWS * len(TRAIN_LANGUAGES)
)

# Storage-relative; extensions/datasets/lang_transfer.py makes these absolute.
CORPUS_ROOT = "lang_corpus"
COMBINED_ROOT = "lang_corpus_combined"


def _checked_role(role: str) -> str:
    if role not in ("train", "eval"):
        raise ValueError(f"role must be 'train' or 'eval', got {role!r}")
    return role


def corpus_dir(role: str) -> str:
    """Per-language fetch cache — never a DatasetSpec.source (docs/experiments-map.md)."""
    return f"{CORPUS_ROOT}/{_checked_role(role)}"


def combined_dir(role: str) -> str:
    """The shuffled, merged file a DatasetSpec.source actually points at."""
    return f"{COMBINED_ROOT}/{_checked_role(role)}"


def languages_for(role: str) -> tuple[tuple[str, str], ...]:
    return TRAIN_LANGUAGES if _checked_role(role) == "train" else EVAL_LANGUAGES
