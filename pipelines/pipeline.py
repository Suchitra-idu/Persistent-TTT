"""
arXiv ML Paper Pipeline — Modal
================================
Everything runs on Modal. Nothing touches your local machine except
terminal output.

Data flow:
  Kaggle API  →  Modal Volume   (metadata snapshot, fetched once)
  arXiv S3    →  Modal Volume   (manifest fetched once; tars streamed
                                 per run and discarded after extraction)
  Volume      →  HuggingFace Hub  (push when a run is complete)

Each pipeline run is isolated by an 8-char UUID (run_id). Running the
same command twice produces two independent datasets in separate Volume
directories — no files are shared or overwritten between runs.

─── One-time setup ───────────────────────────────────────────────────
1.  pip install modal && modal setup
2.  Create three Modal Secrets at modal.com/secrets:
      "aws-arxiv"    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
                     AWS_DEFAULT_REGION (us-east-1)
      "kaggle"       KAGGLE_USERNAME, KAGGLE_KEY
      "huggingface"  HF_TOKEN   (only needed for --push-to-hub)
3.  modal run pipeline.py --fetch-metadata    # pulls ~4 GB from Kaggle

─── Commands ─────────────────────────────────────────────────────────
  # Run with defaults (10k papers, Jan 2024 – Dec 2025)
  modal run pipeline.py

  # Explicit parameters
  modal run pipeline.py --papers 10000 --start 2401 --end 2512

  # Run a second time — completely fresh set, same date window
  modal run pipeline.py --papers 10000 --start 2401 --end 2512

  # Different date range
  modal run pipeline.py --papers 10000 --start 2301 --end 2312

  # List all runs with status and token count
  modal run pipeline.py --list-runs

  # Push a finished run's dataset to HuggingFace Hub (runs on Modal)
  modal run pipeline.py --push-to-hub --run-id-arg a3f1bc92 \
                        --hf-repo your-username/arxiv-ml-10k

  # Inspect the Volume interactively
  modal shell --volume arxiv-pipeline-vol
"""

import io
import json
import re
import gzip
import tarfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import modal

# ─────────────────────────────────────────────────────────────────────────────
# Modal app, container image, volume, secrets
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("arxiv-pipeline")

# Single container image used by every function. All heavy imports
# (boto3, datasets, kaggle) are installed here so Modal can cache the
# image layer and skip re-installing on every invocation.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "boto3",           # S3 access for arXiv tars and manifest
        "botocore",        # boto3 dependency, pinned for retry config
        "datasets",        # HuggingFace Datasets for final output
        "huggingface-hub", # push_to_hub support
        "kagglehub",       # Kaggle's newer Python library — supports KGAT tokens
    )
)

# Persistent NFS-backed volume. Survives container restarts and across
# multiple modal run invocations. All pipeline data lands here.
#
# version=2 is required: v1 volumes cap concurrent writers at ~5, which
# would cause severe contention when clean_batch runs 200 containers in
# parallel. v2 supports hundreds of simultaneous writers to distinct files.
volume = modal.Volume.from_name("arxiv-pipeline-vol", create_if_missing=True, version=2)
VPATH  = "/data"   # mount point inside every container

# Modal Secrets inject credentials as environment variables.
# Functions declare which secrets they need; Modal only injects the
# declared ones, so a function without aws_secret never sees AWS keys.
aws_secret = modal.Secret.from_name("aws-arxiv")
kg_secret  = modal.Secret.from_name("kaggle")
hf_secret  = modal.Secret.from_name("huggingface")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ARXIV_BUCKET    = "arxiv"                      # S3 requester-pays bucket
MANIFEST_S3_KEY = "src/arXiv_src_manifest.xml" # lists every tar and its files
MANIFEST_PATH   = f"{VPATH}/arXiv_src_manifest.xml"
METADATA_PATH   = f"{VPATH}/arxiv-metadata-oai-snapshot.json"
PAPER_CACHE_PATH = f"{VPATH}/paper_cache"      # global cache shared across all runs

# Paper categories to include. All five are ML-adjacent; cross-listed
# papers (e.g. a paper in both cs.LG and cs.CV) are included if ANY
# of their categories appears in this set.
TARGET_CATEGORIES = {"cs.LG", "cs.AI", "cs.CL", "cs.CV", "stat.ML"}

# Text length gates applied after LaTeX cleaning.
# Papers shorter than MIN_CHARS are likely stubs or extraction failures.
# Papers longer than MAX_CHARS are truncated — a reasonable cap for
# very long survey papers that would dominate token counts.
MIN_CHARS = 3_000
MAX_CHARS = 300_000

# How many papers to batch into one clean_batch container call.
# 50 papers per container keeps container count manageable (~200
# containers for 10k papers) while still parallelising well.
CLEAN_BATCH_SIZE = 50

# ─────────────────────────────────────────────────────────────────────────────
# Volume path helpers
# Every run gets its own subdirectory under /data/runs/{run_id}/.
# Functions never write outside their own run directory, so concurrent
# runs cannot interfere with each other.
# ─────────────────────────────────────────────────────────────────────────────

def rdir(run_id):       return f"{VPATH}/runs/{run_id}"
def rmeta(run_id):      return f"{rdir(run_id)}/meta.json"
def rids(run_id):       return f"{rdir(run_id)}/paper_ids.txt"
def rqueue(run_id):     return f"{rdir(run_id)}/tar_queue.jsonl"
def rextracted(run_id): return f"{rdir(run_id)}/extracted"
def rcleaned(run_id):   return f"{rdir(run_id)}/cleaned"
def rdataset(run_id):   return f"{rdir(run_id)}/hf_dataset"

# ─────────────────────────────────────────────────────────────────────────────
# arXiv ID helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_yymm(arxiv_id: str) -> str:
    """
    Extract the YYMM month string from an arXiv ID.

    New-style IDs  (post-2007):  "2504.12345"   → "2504"
    Old-style IDs  (pre-2007):   "cs/0305001"   → "0305"

    The YYMM string is used for date-range filtering: it sorts
    lexicographically the same way it sorts chronologically, so
    simple string comparison works for range checks.
    """
    arxiv_id = arxiv_id.strip()
    if "/" in arxiv_id:
        # Old style: "category/YYMMnnn" — digits start after the slash
        return arxiv_id.split("/")[1][:4]
    # New style: first four characters are YYMM
    return arxiv_id[:4]


def norm_id(filename: str) -> str:
    """
    Normalise a filename from a tar member to a bare arXiv ID.

    Examples:
      "2504.12345v3.gz"  →  "2504.12345"
      "cs0305001.gz"     →  "cs0305001"     (old-style, no slash in filename)
      "main.tex.gz"      →  "main"           (will not match any ID — harmless)

    We strip ALL known extensions iteratively so that "foo.tex.gz"
    becomes "foo", not "foo.tex". Then strip any trailing version
    suffix like "v2" or "v14".
    """
    name = Path(filename).name
    # Strip extensions repeatedly until none remain
    known_exts = {".gz", ".tar", ".zip", ".pdf", ".tex", ".bbl", ".bib"}
    while True:
        suffix = Path(name).suffix
        if suffix in known_exts:
            name = name[: -len(suffix)]
        else:
            break
    # Remove version suffix (v1, v2, … v999)
    return re.sub(r"v\d+$", "", name)

# ─────────────────────────────────────────────────────────────────────────────
# LaTeX → plain text cleaning
# Module-level compiled patterns so they're only compiled once per
# container, not once per paper call.
# ─────────────────────────────────────────────────────────────────────────────

# Environments whose *entire* content we drop (including the tags).
# All common table variants included so & alignment chars don't survive.
_RM_ENV = re.compile(
    r"\\begin\{(thebibliography|bibliography|figure\*?|table\*?"
    r"|tabular[xy]?\*?|longtable|tabulary|tabu|array"
    r"|lstlisting|verbatim|minted|algorithm\*?|algorithmic|filecontents\*?)\}"
    r".*?\\end\{\1\}",
    re.DOTALL | re.IGNORECASE,
)

# Macro definition commands — strip the entire definition line.
# e.g. \newcommand{\foo}[1]{...} and \def\foo{...}
_MACRODEF = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand|DeclareMathOperator"
    r"|newenvironment|renewenvironment)\*?"
    r"(?:\[[^\]]*\])?"          # optional [nargs]
    r"\{[^}]*\}"                # command name
    r"(?:\[[^\]]*\])?"          # optional [default]
    r"(?:\{(?:[^{}]|\{[^{}]*\})*\})?",  # optional body
    re.DOTALL,
)
_DEFCMD = re.compile(r"\\(?:def|let|edef|gdef|xdef)\s*\\[a-zA-Z@]+[^{}\n]*(?:\{[^}]*\})?")

# Author/affiliation metadata — not prose.
_AUTHOR_META = re.compile(
    r"\\(?:author|affil(?:iation)?|address|institute|thanks|email|orcid)"
    r"(?:\[[^\]]*\])?\{(?:[^{}]|\{[^{}]*\})*\}",
    re.DOTALL,
)

_COMMENT = re.compile(r"(?<!\\)%.*?$", re.MULTILINE)
_PREAMBLE = re.compile(r"^.*?\\begin\{document\}", re.DOTALL)
_POSTAMBLE = re.compile(r"\\end\{document\}.*$", re.DOTALL)

# Display math: capture inner content so we keep the LaTeX notation.
# Stripping just the delimiters ($$, \[, \begin{equation}) preserves the
# math for LM training — models trained on arXiv/math text handle raw
# LaTeX notation well.
_DISP_MATH = [
    re.compile(p, re.DOTALL) for p in [
        r"\$\$(.*?)\$\$",
        r"\\\[(.*?)\\\]",
        r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}",
        r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}",
        r"\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}",
    ]
]
# Inline math: strip $...$ delimiters but keep the content.
# $x_i^2$ → x_i^2, $\theta$ → \theta (then converted to θ by _GREEK_RE).
_INLINE_MATH = re.compile(r"\$([^$\n]{0,300})\$")

_SECTION = re.compile(r"\\(?:sub)*section\*?\{([^}]+)\}")
_REFS = re.compile(
    r"\\(?:cite[pt]?|label|footnote)\{[^}]*\}"
)
_REFS_NUM = re.compile(r"\\(?:ref|eqref)\{[^}]*\}")   # replaced with marker, not deleted

# \frac{a}{b} → (a)/(b). Handles one level of nesting in numerator/denominator.
# Also covers \dfrac and \tfrac.
_FRAC = re.compile(
    r"\\[dt]?frac\{((?:[^{}]|\{[^{}]*\})*)\}\{((?:[^{}]|\{[^{}]*\})*)\}"
)
_KEEP_ARG = re.compile(
    r"\\(?:textbf|textit|emph|text|textrm|texttt|textsc"
    r"|mathrm|mathbf|mathcal|mathbb|mathsf|mathtt|mathit"
    r"|boldsymbol|vec|hat|bar|tilde|dot|ddot|overline|underline"
    r"|widehat|widetilde|operatorname|url|href)\{([^}]*)\}"
)
_CMD = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})*")
_WS = re.compile(r"\n{3,}")

# LaTeX accent commands → Unicode. Handles \'x, \"x, \^x, \`x, \~x, \=x
# Must run before the generic backslash stripper.
_ACCENT_TABLE = {
    "'": {"a":"á","e":"é","i":"í","o":"ó","u":"ú","y":"ý",
           "A":"Á","E":"É","I":"Í","O":"Ó","U":"Ú","Y":"Ý"},
    "`": {"a":"à","e":"è","i":"ì","o":"ò","u":"ù",
           "A":"À","E":"È","I":"Ì","O":"Ò","U":"Ù"},
    "^": {"a":"â","e":"ê","i":"î","o":"ô","u":"û",
           "A":"Â","E":"Ê","I":"Î","O":"Ô","U":"Û"},
    '"': {"a":"ä","e":"ë","i":"ï","o":"ö","u":"ü",
           "A":"Ä","E":"Ë","I":"Ï","O":"Ö","U":"Ü"},
    "~": {"a":"ã","n":"ñ","o":"õ","A":"Ã","N":"Ñ","O":"Õ"},
    "=": {"a":"ā","e":"ē","i":"ī","o":"ō","u":"ū",
           "A":"Ā","E":"Ē","I":"Ī","O":"Ō","U":"Ū"},
}
_ACCENT_RE = re.compile(r"\\(['\"`^~=])\{?([a-zA-Z])\}?")

def _replace_accent(m: re.Match) -> str:
    cmd, char = m.group(1), m.group(2)
    return _ACCENT_TABLE.get(cmd, {}).get(char, char)

# Greek letters and common math operators → Unicode.
# Applied before _CMD so these survive command stripping.
_GREEK_TABLE = {
    "alpha":"α","beta":"β","gamma":"γ","delta":"δ","epsilon":"ε",
    "varepsilon":"ε","zeta":"ζ","eta":"η","theta":"θ","vartheta":"θ",
    "iota":"ι","kappa":"κ","lambda":"λ","mu":"μ","nu":"ν","xi":"ξ",
    "pi":"π","varpi":"π","rho":"ρ","varrho":"ρ","sigma":"σ","varsigma":"ς",
    "tau":"τ","upsilon":"υ","phi":"φ","varphi":"φ","chi":"χ","psi":"ψ",
    "omega":"ω",
    "Gamma":"Γ","Delta":"Δ","Theta":"Θ","Lambda":"Λ","Xi":"Ξ","Pi":"Π",
    "Sigma":"Σ","Upsilon":"Υ","Phi":"Φ","Psi":"Ψ","Omega":"Ω",
    "nabla":"∇","partial":"∂","infty":"∞","ell":"ℓ",
    "leq":"≤","geq":"≥","neq":"≠","approx":"≈","sim":"∼","equiv":"≡",
    "times":"×","cdot":"·","circ":"∘","oplus":"⊕","otimes":"⊗",
    "sum":"∑","prod":"∏","int":"∫","oint":"∮",
    "in":"∈","notin":"∉","subset":"⊂","supset":"⊃","subseteq":"⊆",
    "cup":"∪","cap":"∩","emptyset":"∅",
    "forall":"∀","exists":"∃","nexists":"∄",
    "rightarrow":"→","leftarrow":"←","Rightarrow":"⇒","Leftarrow":"⇐",
    "leftrightarrow":"↔","Leftrightarrow":"⇔","mapsto":"↦",
    "ldots":"…","cdots":"…","vdots":"⋮","ddots":"⋱",
    "langle":"⟨","rangle":"⟩",
    "top":"⊤","bot":"⊥","perp":"⊥","mid":"∣","parallel":"∥",
    "propto":"∝","pm":"±","mp":"∓",
}
_GREEK_RE = re.compile(
    r"\\(" + "|".join(sorted(_GREEK_TABLE, key=len, reverse=True)) + r")\b"
)
def _replace_greek(m: re.Match) -> str:
    return _GREEK_TABLE.get(m.group(1), m.group(0))

# Strip any line that contains an email address — catches author affiliation
# blocks that survive \author{} stripping. The \b before [\w.+-]+ is dropped
# so that @ruc.edu.cn (starting with @) is also matched.
_EMAIL_LINE = re.compile(r"[^\n]*[\w.+-]*@[\w.-]+\.\w{2,6}\b[^\n]*", re.MULTILINE)

# Reject papers that look like style/template files rather than research.
# These slip in when someone includes the AAAI/Elsevier author kit .tex.
_TEMPLATE_PHRASES = re.compile(
    r"formatting requirements|camera.?ready|author kit|style file"
    r"|manuscript preparation|submission guidelines|paper template",
    re.IGNORECASE,
)


def clean_latex(src: str) -> str:
    t = _COMMENT.sub("", src)

    m = _PREAMBLE.search(t)
    if m:
        t = t[m.end():]

    m = _POSTAMBLE.search(t)
    if m:
        t = t[:m.start()]

    # Reject style/template files (AAAI author kit, Elsevier guide, etc.)
    if _TEMPLATE_PHRASES.search(t[:3000]):
        return ""

    # Strip macro definitions and author/affiliation blocks before
    # environment removal so nested braces don't confuse _RM_ENV.
    t = _MACRODEF.sub("", t)
    t = _DEFCMD.sub("", t)
    t = _AUTHOR_META.sub("", t)
    t = _EMAIL_LINE.sub("", t)

    # Drop the pre-section author/affiliation block. Conference papers often
    # place author names, university affiliations and emails as raw text
    # between \begin{document} and the first \section — strip all of it.
    _first_sec = re.search(r"\\(?:section|chapter)\*?\{", t)
    if _first_sec and _first_sec.start() < 3000:
        t = t[_first_sec.start():]

    t = _RM_ENV.sub("", t)

    # Strip display-math delimiters but keep the LaTeX content.
    # $$E=mc^2$$ → E=mc^2, \begin{align}...\end{align} → inner content.
    for p in _DISP_MATH:
        t = p.sub(r" \1 ", t)

    # Strip inline-math delimiters, keep content.
    # $x_i^2$ → x_i^2, $\theta$ → \theta (converted to θ below).
    t = _INLINE_MATH.sub(r" \1 ", t)

    # Equation line breaks \\ and alignment & left from align environments
    t = re.sub(r"\\\\", " ", t)
    t = re.sub(r"(?<!\w)&(?!\w)", " ", t)

    t = _SECTION.sub(r"\n\n## \1\n\n", t)
    t = _REFS.sub("", t)
    t = _REFS_NUM.sub("[*]", t)    # \ref{} / \eqref{} → [*] instead of empty hole

    # \frac{a}{b} → (a)/(b) before _CMD strips it entirely.
    t = _FRAC.sub(r"(\1)/(\2)", t)

    # Strip \left and \right size hints, keeping the delimiter that follows.
    t = re.sub(r"\\(?:left|right)\b", "", t)

    t = _KEEP_ARG.sub(r"\1", t)   # \mathcal{E} → E, \vec{v} → v, etc.

    # Accents and Greek letters → Unicode before _CMD strips backslashes.
    t = _ACCENT_RE.sub(_replace_accent, t)
    t = _GREEK_RE.sub(_replace_greek, t)   # \theta → θ, \nabla → ∇, etc.

    t = _CMD.sub("", t)

    t = re.sub(r"[{}]", "", t)

    # LaTeX spacing commands: \, \; \: \! → space (not comma/semicolon).
    t = re.sub(r"\\[,;:!/]", " ", t)
    # Remaining backslash sequences
    t = re.sub(r"\\\w*", "", t)

    t = t.replace("~", " ")

    # Strip leftover optional argument brackets like [itemsep=0pt], [htbp].
    t = re.sub(r"\[[^\]\n]{0,100}\]", " ", t)

    # Drop empty parens left by stripped \eqref: "Eq. ()" → "Eq."
    t = re.sub(r"\(\s*\)", "", t)
    # Collapse multiple spaces/tabs to single space.
    t = re.sub(r"[ \t]{2,}", " ", t)

    t = _WS.sub("\n\n", t)

    return t.strip()


def find_tex(paper_dir: Path) -> Optional[Path]:
    """
    Find the main .tex file for a paper.

    Strategy (in order):
      1. If exactly one .tex file, return it.
      2. If multiple .tex files, prefer the one containing
         \\begin{document} (i.e. the root file, not an included chapter).
      3. Fall back to the largest .tex file.
      4. If no .tex files, try .gz files whose name contains ".tex"
         (some submissions are gzipped tex).
      5. Last resort: any file with no extension (some submissions are
         a single file with no extension at all).
    """
    files = list(paper_dir.glob("*.tex"))
    if not files:
        gz = [f for f in paper_dir.glob("*.gz") if ".tex" in f.name]
        if gz:
            return gz[0]
        plain = [f for f in paper_dir.iterdir() if f.is_file() and not f.suffix]
        return plain[0] if plain else None
    if len(files) == 1:
        return files[0]
    for f in files:
        try:
            if "\\begin{document}" in f.read_text(encoding="utf-8", errors="ignore"):
                return f
        except Exception:
            pass
    return max(files, key=lambda f: f.stat().st_size)


def read_file(path: Path) -> Optional[str]:
    """Read a file, transparently handling gzip compression."""
    try:
        if str(path).endswith(".gz"):
            with gzip.open(str(path), "rt", encoding="utf-8", errors="ignore") as f:
                return f.read()
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


_INPUT_CMD = re.compile(r"\\(?:input|include|subfile)\{([^}]+)\}")

def resolve_inputs(text: str, paper_dir: Path, _depth: int = 0) -> str:
    """
    Inline \\input{} and \\include{} sub-files so the cleaner sees the
    full paper text, not just hollow section headers.
    Capped at depth 5 to prevent circular includes.
    """
    if _depth > 5:
        return text
    def _sub(m: re.Match) -> str:
        fname = m.group(1).strip()
        for candidate in [fname, fname + ".tex"]:
            # Ignore any directory component — all files are flat in paper_dir
            p = paper_dir / Path(candidate).name
            if p.exists():
                try:
                    sub = p.read_text(encoding="utf-8", errors="ignore")
                    return resolve_inputs(sub, paper_dir, _depth + 1)
                except Exception:
                    pass
        return ""
    return _INPUT_CMD.sub(_sub, text)

# ─────────────────────────────────────────────────────────────────────────────
# Modal functions
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={VPATH: volume},
    secrets=[kg_secret],   # injects KAGGLE_USERNAME and KAGGLE_KEY
    timeout=1800,          # 30 min: generous for a 4 GB download
    memory=2048,
    cpu=2,
)
def fetch_metadata_fn():
    """
    Download the Kaggle arXiv metadata snapshot to the Volume.

    The snapshot is a single ~4 GB JSONL file — one record per paper,
    containing arXiv ID, categories, title, and abstract. We use it in
    filter_metadata() to find paper IDs in our target window without
    touching S3 at all.

    This function is idempotent: if the file already exists on the
    Volume it returns immediately. Run it once during setup; never
    again unless you want to refresh the metadata.

    Uses kagglehub (not the legacy kaggle CLI) — kagglehub natively
    supports the new KGAT token format via KAGGLE_USERNAME + KAGGLE_KEY
    environment variables injected by the Modal Secret.
    """
    import shutil
    import kagglehub

    volume.reload()

    if Path(METADATA_PATH).exists():
        size_gb = Path(METADATA_PATH).stat().st_size / 1e9
        print(f"Metadata already present ({size_gb:.1f} GB). Skipping.")
        return

    print("Downloading arXiv metadata from Kaggle via kagglehub ...")

    # kagglehub downloads to a local cache directory and returns the path.
    # KAGGLE_USERNAME and KAGGLE_KEY are read from environment variables
    # injected by the Modal Secret — KGAT tokens work here directly.
    # We pass path= to land the download inside the Volume rather than
    # the default ~/.cache/kagglehub which may not be writable.
    dl_cache = Path(f"{VPATH}/_kagglehub_cache")
    dl_cache.mkdir(parents=True, exist_ok=True)

    # kagglehub.dataset_download returns the local path to the downloaded file.
    # The "Cornell-University/arxiv" dataset contains the metadata JSONL.
    local_path = kagglehub.dataset_download(
        "Cornell-University/arxiv",
        path="arxiv-metadata-oai-snapshot.json",
        force_download=False,          # skip if already in cache
    )

    downloaded = Path(local_path)
    if not downloaded.exists():
        raise FileNotFoundError(
            f"kagglehub returned path {local_path} but file not found. "
            "Check KAGGLE_USERNAME and KAGGLE_KEY in the Modal Secret."
        )

    # Move from kagglehub cache to the permanent Volume path
    shutil.move(str(downloaded), METADATA_PATH)
    shutil.rmtree(dl_cache, ignore_errors=True)

    volume.commit()
    size_gb = Path(METADATA_PATH).stat().st_size / 1e9
    print(f"Metadata saved ({size_gb:.1f} GB).")


@app.function(
    image=image,
    volumes={VPATH: volume},
    secrets=[aws_secret],  # required: S3 requester-pays needs auth
    timeout=600,
    memory=512,
)
def fetch_manifest():
    """
    Download the arXiv S3 source manifest XML to the Volume.

    The manifest lists every tar file in the s3://arxiv/src/ prefix
    along with the filenames inside each tar. It's ~200 MB and is
    shared across all pipeline runs — we parse it in build_tar_queue()
    to find which tars contain which papers.

    Idempotent: skips download if the file already exists.
    """
    import boto3, botocore

    volume.reload()
    mp = Path(MANIFEST_PATH)
    if mp.exists():
        print(f"Manifest present ({mp.stat().st_size/1e6:.0f} MB). Skipping.")
        return

    print("Downloading arXiv manifest from S3 ...")
    s3 = boto3.client(
        "s3",
        config=botocore.config.Config(retries={"max_attempts": 5, "mode": "adaptive"}),
    )
    mp.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(
        Bucket=ARXIV_BUCKET,
        Key=MANIFEST_S3_KEY,
        Filename=str(mp),
        ExtraArgs={"RequestPayer": "requester"},
    )
    volume.commit()
    print(f"Manifest saved ({mp.stat().st_size/1e6:.0f} MB).")


@app.function(
    image=image,
    volumes={VPATH: volume},
    timeout=1800,   # the 4 GB JSONL scan takes ~20 min
    memory=4096,    # file is read line-by-line but json.loads needs headroom
    cpu=2,
)
def filter_metadata(run_id: str, start_month: str, end_month: str, target_papers: int) -> int:
    """
    Scan the Kaggle metadata snapshot and write a list of matching paper IDs.

    A paper matches if:
      - At least one of its categories is in TARGET_CATEGORIES
      - Its submission month (YYMM) falls within [start_month, end_month]

    The output file (paper_ids.txt) is sorted by arXiv ID, which is
    equivalent to chronological order for new-style IDs. This means
    build_tar_queue() will prioritise older papers within the window
    when it trims the queue to hit the target_papers count — consistent
    and reproducible across runs with the same parameters.

    Returns the total number of matched paper IDs.
    """
    volume.reload()

    if not Path(METADATA_PATH).exists():
        raise FileNotFoundError(
            "Metadata not on Volume. "
            "Run:  modal run pipeline.py --fetch-metadata"
        )

    out = Path(rids(run_id))
    out.parent.mkdir(parents=True, exist_ok=True)

    matched, seen = [], 0
    print(f"Scanning metadata: {start_month}–{end_month}, "
          f"categories={sorted(TARGET_CATEGORIES)} ...")

    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Category field is a space-separated string of all categories
            if not set(rec.get("categories", "").split()) & TARGET_CATEGORIES:
                continue

            month = parse_yymm(rec.get("id", ""))
            if not (start_month <= month <= end_month):
                continue

            matched.append(rec["id"])

            if seen % 500_000 == 0:
                print(f"  {seen:,} scanned, {len(matched):,} matched ...")

    matched.sort()   # chronological order
    print(f"Matched {len(matched):,} papers.")

    if len(matched) < target_papers:
        print(
            f"WARNING: only {len(matched):,} papers available in window, "
            f"{target_papers:,} requested. Consider widening date range."
        )

    out.write_text("\n".join(matched) + "\n")
    volume.commit()
    return len(matched)


@app.function(
    image=image,
    volumes={VPATH: volume},
    timeout=600,
    memory=2048,
    cpu=2,
)
def build_tar_queue(run_id: str, target_papers: int) -> int:
    """
    Build a prioritised download queue of S3 tar files.

    Parses the arXiv manifest XML, cross-references it with this run's
    paper_ids.txt, and ranks tars by how many wanted papers they contain
    (highest first). The queue is then trimmed — we stop adding tars once
    the cumulative wanted-paper count reaches target_papers.

    This "greedy density" strategy minimises the total bytes downloaded
    from S3 to reach the paper count target.

    Writes tar_queue.jsonl where each line is:
      {"tar_key": "src/...", "wanted": N, "total": M,
       "density": 0.nnn, "ids": ["2401.xxxxx", ...]}
    """
    import xml.etree.ElementTree as ET

    volume.reload()

    # Load the full set of wanted IDs for this run
    wanted = set(Path(rids(run_id)).read_text().splitlines())
    wanted.discard("")
    print(f"Parsing manifest for {len(wanted):,} wanted IDs ...")

    root = ET.parse(MANIFEST_PATH).getroot()

    # The manifest has one <file> per tar. <filename> holds the S3 key;
    # <first_item>/<last_item> give the lexicographic range of paper IDs
    # inside that tar. There are no per-paper <filename> entries.
    records = []
    for file_elem in root.findall(".//file"):
        fname_elem = file_elem.find("filename")
        if fname_elem is None or not fname_elem.text:
            continue
        tar_key = fname_elem.text.strip()

        first_elem = file_elem.find("first_item")
        last_elem  = file_elem.find("last_item")
        num_elem   = file_elem.find("num_items")
        if first_elem is None or last_elem is None:
            continue

        first_item = first_elem.text.strip() if first_elem.text else ""
        last_item  = last_elem.text.strip()  if last_elem.text  else ""
        num_items  = int(num_elem.text.strip()) if (num_elem is not None and num_elem.text) else 0

        # Lexicographic range check — correct for new-style IDs (e.g. "2401.12345").
        hits = [pid for pid in wanted if first_item <= pid <= last_item]

        if hits:
            records.append({
                "tar_key": tar_key,
                "wanted":  len(hits),
                "total":   num_items,
                "density": round(len(hits) / num_items, 4) if num_items else 0,
                "ids":     hits,
            })

    # Sort descending by count of wanted papers (not by density ratio).
    # A tar with 4000 wanted papers out of 10000 total is more valuable
    # than a tar with 500 wanted out of 600, even though the second is
    # "denser" — we care about absolute yield per download.
    records.sort(key=lambda x: x["wanted"], reverse=True)

    # Take the minimal prefix of tars that covers target_papers
    queue, cumulative = [], 0
    for r in records:
        if cumulative >= target_papers:
            break
        queue.append(r)
        cumulative += r["wanted"]

    qp = Path(rqueue(run_id))
    qp.parent.mkdir(parents=True, exist_ok=True)
    with open(qp, "w") as f:
        for r in queue:
            f.write(json.dumps(r) + "\n")

    volume.commit()
    print(f"Queue: {len(queue)} tars covering ~{cumulative:,} papers.")
    return len(queue)


@app.function(
    image=image,
    volumes={VPATH: volume},
    secrets=[aws_secret],
    timeout=900,    # 15 min: a single tar is ~500 MB; S3 can be slow
    memory=3072,    # tar is loaded into memory before extraction
    cpu=2,
    retries=2,      # automatic retry on transient S3 or network errors
)
def download_and_extract_tar(run_id: str, tar_record: dict) -> dict:
    """
    Download one S3 tar and extract only the wanted paper source files.

    Called once per tar via .starmap(), so many tars are processed in
    parallel. Each invocation is fully independent — it writes to its
    own paper subdirectories under extracted/{arxiv_id}/ and never
    touches files written by other invocations.

    The tar is downloaded into a BytesIO buffer (never written to disk
    as a full tar), and the buffer is discarded after extraction. This
    keeps peak disk usage proportional to wanted papers only.

    Returns a summary dict with extracted/skipped counts and any error.
    """
    import boto3, botocore, shutil

    volume.reload()

    tar_key    = tar_record["tar_key"]
    wanted_ids = set(tar_record["ids"])
    out_dir    = Path(rextracted(run_id))
    cache_dir  = Path(PAPER_CACHE_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check which papers are already in the global cache.
    # Cache hit: create an empty sentinel dir in the run's extracted/ so
    # list_extracted_ids can count it, but don't copy the files — clean_batch
    # reads directly from paper_cache/.
    missing_ids = set()
    from_cache  = 0
    for nid in wanted_ids:
        safe   = nid.replace("/", "_")
        cached = cache_dir / safe
        dest   = out_dir   / safe
        if cached.exists() and any(cached.iterdir()):
            dest.mkdir(parents=True, exist_ok=True)   # sentinel only
            from_cache += 1
        else:
            missing_ids.add(nid)

    if not missing_ids:
        print(f"  {tar_key}  {from_cache} papers served from cache, skipping download")
        volume.commit()
        return {"tar_key": tar_key, "extracted": from_cache,
                "skipped": 0, "error": None}

    s3 = boto3.client(
        "s3",
        config=botocore.config.Config(retries={"max_attempts": 5, "mode": "adaptive"}),
    )

    # Stream the entire tar into memory. A typical monthly tar is
    # ~300–600 MB. We need it all in memory to seek within it for
    # selective extraction.
    buf = io.BytesIO()
    s3.download_fileobj(
        Bucket=ARXIV_BUCKET,
        Key=tar_key,
        Fileobj=buf,
        ExtraArgs={"RequestPayer": "requester"},
    )
    buf.seek(0)
    print(f"  {tar_key}  {buf.getbuffer().nbytes / 1e6:.0f} MB downloaded"
          f"  ({from_cache} already cached)")

    # Only cache tex source files — figures and style files are never read
    # and account for the vast majority of storage.
    _TEX_KEEP = {".tex", ".bbl"}

    def _unpack_paper(data: bytes, member_name: str, sentinel_dir: Path,
                      cache_paper_dir: Path):
        """
        Unpack a paper's source blob into cache_paper_dir (tex/bbl only).
        Creates sentinel_dir so list_extracted_ids can find this paper.
        """
        sentinel_dir.mkdir(parents=True, exist_ok=True)
        cache_paper_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as inner_tf:
                for inner_member in inner_tf.getmembers():
                    if not inner_member.isfile():
                        continue
                    p = Path(inner_member.name)
                    if p.suffix not in _TEX_KEEP and p.suffix != "":
                        continue  # skip figures, .sty, .pdf, .png, etc.
                    inner_fobj = inner_tf.extractfile(inner_member)
                    if inner_fobj:
                        (cache_paper_dir / p.name).write_bytes(inner_fobj.read())
        except Exception:
            # Not a tar.gz — write as-is only if it looks like a tex file
            p = Path(member_name)
            if p.suffix in _TEX_KEEP or p.suffix == "":
                (cache_paper_dir / p.name).write_bytes(data)

    extracted = skipped = 0
    try:
        with tarfile.open(fileobj=buf, mode="r:*") as tf:
            for member in tf.getmembers():
                nid = norm_id(member.name)
                if nid not in missing_ids:
                    continue

                safe            = nid.replace("/", "_")
                sentinel_dir    = out_dir   / safe
                cache_paper_dir = cache_dir / safe
                try:
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        skipped += 1
                        continue
                    _unpack_paper(fobj.read(), member.name,
                                  sentinel_dir, cache_paper_dir)
                    extracted += 1
                except Exception:
                    skipped += 1

    except tarfile.TarError as e:
        # Non-fatal: return the error so the orchestrator can log it
        return {"tar_key": tar_key, "extracted": extracted + from_cache,
                "skipped": skipped, "error": str(e)}

    volume.commit()
    total = extracted + from_cache
    print(f"  {tar_key}  extracted {extracted} new + {from_cache} from cache, skipped {skipped}")
    return {"tar_key": tar_key, "extracted": total,
            "skipped": skipped, "error": None}


@app.function(
    image=image,
    volumes={VPATH: volume},
    timeout=600,    # 10 min per batch of CLEAN_BATCH_SIZE papers
    memory=1024,
    cpu=2,
    retries=1,
)
def clean_batch(run_id: str, arxiv_ids: list[str]) -> list[Optional[dict]]:
    """
    Clean a batch of papers' LaTeX source to plain text.

    Batching is important: spawning one Modal container per paper would
    create 10k containers for a 10k-paper run, overwhelming Modal's
    scheduler and wasting container startup time (~1–2s each). Batching
    50 papers per container gives ~200 containers — fast and efficient.

    Each paper in the batch is processed independently. If one fails
    (missing tex, too short after cleaning), it returns None for that
    paper and the rest continue. The batch result is a list aligned with
    the input arxiv_ids list.

    A single volume.commit() at the end of the batch covers all papers
    in the batch — no per-paper commits.
    """
    volume.reload()

    c_dir = Path(rcleaned(run_id))
    c_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for arxiv_id in arxiv_ids:
        safe_id   = arxiv_id.replace("/", "_")
        # Source files live in the global cache; extracted/ holds only sentinels.
        sentinel  = Path(rextracted(run_id)) / safe_id
        paper_dir = Path(PAPER_CACHE_PATH)   / safe_id
        if not sentinel.exists() or not paper_dir.exists():
            results.append(None)
            continue

        out = c_dir / f"{safe_id}.txt"
        # If already cleaned (shouldn't happen in a fresh run), return
        # the cached result rather than re-processing.
        if out.exists():
            chars = len(out.read_text(encoding="utf-8"))
            results.append({"arxiv_id": arxiv_id, "chars": chars,
                             "tokens_est": chars // 4})
            continue

        tex = find_tex(paper_dir)
        if tex is None:
            results.append(None)
            continue

        raw = read_file(tex)
        if raw is None:
            results.append(None)
            continue

        raw = resolve_inputs(raw, paper_dir)
        cleaned = clean_latex(raw)

        if len(cleaned) < MIN_CHARS:
            # Too short — likely a stub, erratum, or extraction failure
            results.append(None)
            continue

        if len(cleaned) > MAX_CHARS:
            # Truncate very long papers (survey papers, textbooks)
            cleaned = cleaned[:MAX_CHARS]

        out.write_text(cleaned, encoding="utf-8")
        results.append({"arxiv_id": arxiv_id, "chars": len(cleaned),
                         "tokens_est": len(cleaned) // 4})

    # Single commit for the whole batch
    volume.commit()
    return results


@app.function(
    image=image,
    volumes={VPATH: volume},
    timeout=3600,
    memory=8192,
    cpu=4,
)
def build_hf_dataset(run_id: str) -> tuple[str, int]:
    """
    Assemble cleaned text files into a HuggingFace Dataset.

    Scans cleaned/ directly — no manifest needed. Returns (dataset_path,
    total_tokens_estimate). Uses a thread pool for parallel reads.
    """
    from datasets import Dataset, DatasetDict, Features, Value
    from concurrent.futures import ThreadPoolExecutor

    volume.reload()
    c_dir = Path(rcleaned(run_id))
    out   = Path(rdataset(run_id))

    if not c_dir.exists():
        raise FileNotFoundError(f"No cleaned/ directory for run {run_id}")

    txt_files = sorted(c_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {c_dir}. "
            "volume.commit() may not have been called after cleaning."
        )

    total = len(txt_files)
    print(f"Reading {total:,} cleaned files ...")

    def _read(p):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            return None
        if not text.strip():
            return None
        stem = p.stem
        arxiv_id = stem.replace("_", "/", 1) if "_" in stem and not stem[:4].isdigit() else stem
        chars = len(text)
        return {"arxiv_id": arxiv_id, "text": text, "chars": chars, "tokens_est": chars // 4}

    rows = []
    with ThreadPoolExecutor(max_workers=32) as ex:
        for i, result in enumerate(ex.map(_read, txt_files), 1):
            if result is not None:
                rows.append(result)
            if i % 2000 == 0 or i == total:
                print(f"  {i:,}/{total:,} read, {len(rows):,} valid ...")

    if not rows:
        raise RuntimeError(f"All .txt files were empty under {c_dir}")

    print(f"Building dataset from {len(rows):,} papers ...")
    features = Features({
        "arxiv_id":   Value("string"),
        "text":       Value("string"),
        "chars":      Value("int64"),
        "tokens_est": Value("int64"),
    })
    out.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_dict(
        {k: [r[k] for r in rows] for k in ("arxiv_id", "text", "chars", "tokens_est")},
        features=features,
    )
    DatasetDict({"train": ds}).save_to_disk(str(out))
    volume.commit()
    total_tokens = sum(r["tokens_est"] for r in rows)
    print(f"Dataset saved: {len(ds):,} papers, ~{total_tokens:,} tokens")
    return str(out), total_tokens


@app.function(
    image=image,
    volumes={VPATH: volume},
    timeout=600,
    memory=2048,
)
def audit_fn(run_id: str, sample_size: int = 200, cleaned_dir: str = "") -> dict:
    """
    Sample cleaned papers and check for under-cleaning and over-cleaning.

    Under-cleaning checks
    ─────────────────────
    latex_residue   \\command patterns still in text (threshold: >5)
    env_tags        \\begin{} / \\end{} survived cleaning
    item_residue    \\item list markers (even 1 is wrong)
    dollar_signs    bare $ from inline math (>10)
    table_align     & alignment chars from tables (>2)
    stray_braces    unmatched { } (>10)
    backslash_misc  non-alpha backslash sequences like \\ \\, \\; (>3)
    email_leak      email address survived author stripping
    encoding_errors Unicode replacement char (\\ufffd) present

    Over-cleaning checks
    ────────────────────
    hollow_sections most ## sections have <40 words of prose (\\input not resolved)
    math_heavy      >15% of word-tokens are [MATH] (may be fine for theory papers)
    consec_math     3+ [MATH] tokens in a row appearing >2 times
    over_stripped   cleaned is <3% of source .tex size
    under_stripped  cleaned is >85% of source .tex size (almost nothing removed)

    Content quality checks
    ──────────────────────
    low_prose       <55% of non-whitespace chars are alphabetic (symbol-heavy)
    fragmented      >35% of lines have <6 non-whitespace chars (table/formula noise)
    junk_start      first 80 chars are symbol-heavy (paper begins with LaTeX junk)
    """
    import random

    volume.reload()
    c_dir     = Path(cleaned_dir) if cleaned_dir else Path(rcleaned(run_id))
    cache_dir = Path(PAPER_CACHE_PATH)

    if not c_dir.exists():
        return {"error": f"Directory not found: {c_dir}"}
    txt_files = sorted(c_dir.glob("*.txt"))
    if not txt_files:
        return {"error": "No .txt files found"}

    rng    = random.Random(42)
    sample = rng.sample(txt_files, min(sample_size, len(txt_files)))

    # Compiled patterns (once, not per paper)
    _LATEX_CMD    = re.compile(r"\\[a-zA-Z]+")
    _EMAIL_PAT    = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w{2,6}\b")
    _ENV_PAT      = re.compile(r"\\(?:begin|end)\{")
    _ITEM_PAT     = re.compile(r"\\item\b")
    _BSL_MISC     = re.compile(r"\\[^a-zA-Z\s]")   # \\, \; \: \. \, etc.
    _CONSEC_MATH  = re.compile(r"(\[MATH\]\s*){3,}")
    _SECTION_PAT  = re.compile(r"^##\s", re.MULTILINE)

    rows = []
    for p in sample:
        text  = p.read_text(encoding="utf-8")
        stem  = p.stem

        # ── Token / word counts ──────────────────────────────────────────
        words        = len(text.split())
        math_count   = text.count("[MATH]")
        math_ratio   = math_count / max(words, 1)
        n_sections   = len(_SECTION_PAT.findall(text))

        # ── Under-cleaning signals ───────────────────────────────────────
        latex_cmds   = len(_LATEX_CMD.findall(text))
        has_env_tags = bool(_ENV_PAT.search(text))
        item_count   = len(_ITEM_PAT.findall(text))
        dollar_count = text.count("$")
        table_align  = text.count(" & ")
        stray_braces = text.count("{") + text.count("}")
        bsl_misc     = len(_BSL_MISC.findall(text))
        has_email    = bool(_EMAIL_PAT.search(text))
        has_enc_err  = "�" in text

        # ── Over-cleaning / structural signals ──────────────────────────
        consec_math  = len(_CONSEC_MATH.findall(text))

        # Hollow sections: split on ## headers, count chunks <40 words
        parts        = re.split(r"\n##[^\n]+\n", text)
        hollow_parts = sum(1 for part in parts if len(part.split()) < 40)
        hollow_ratio = hollow_parts / max(len(parts), 1)

        # Compression vs source .tex
        paper_dir    = cache_dir / stem
        source_chars = (
            sum(f.stat().st_size for f in paper_dir.glob("*.tex"))
            if paper_dir.exists() else 0
        )
        cleaned_chars = len(text)
        compression   = cleaned_chars / max(source_chars, 1)

        # ── Content quality ──────────────────────────────────────────────
        nonspace     = [c for c in text if not c.isspace()]
        alpha_ratio  = sum(1 for c in nonspace if c.isalpha()) / max(len(nonspace), 1)

        lines        = [l for l in text.split("\n") if l.strip()]
        short_lines  = sum(1 for l in lines if len(l.strip()) < 6)
        short_ratio  = short_lines / max(len(lines), 1)

        # Junk start: first 80 non-whitespace chars should be mostly letters
        head         = text.lstrip()[:80]
        head_ns      = [c for c in head if not c.isspace()]
        head_junk    = sum(1 for c in head_ns if c in r"\{}$%^_&~") / max(len(head_ns), 1)

        # ── Flag assembly ────────────────────────────────────────────────
        flags = []

        # Under-cleaning
        if latex_cmds > 5:
            flags.append(f"latex_residue({latex_cmds})")
        if has_env_tags:
            flags.append("env_tags")
        if item_count > 0:
            flags.append(f"item_residue({item_count})")
        if dollar_count > 10:
            flags.append(f"dollar_signs({dollar_count})")
        if table_align > 2:
            flags.append(f"table_align({table_align})")
        if stray_braces > 10:
            flags.append(f"stray_braces({stray_braces})")
        if bsl_misc > 3:
            flags.append(f"backslash_misc({bsl_misc})")
        if has_email:
            flags.append("email_leak")
        if has_enc_err:
            flags.append(f"encoding_errors({text.count(chr(0xfffd))})")

        # Over-cleaning / structural
        if hollow_ratio > 0.5 and n_sections >= 3:
            flags.append(f"hollow({hollow_ratio:.0%})")
        if math_ratio > 0.15:
            flags.append(f"math_heavy({math_ratio:.2f})")
        if consec_math > 2:
            flags.append(f"consec_math({consec_math})")
        if 0 < compression < 0.03 and source_chars > 5_000:
            flags.append(f"over_stripped({compression:.3f})")
        if compression > 0.85 and source_chars > 5_000:
            flags.append(f"under_stripped({compression:.3f})")

        # Content quality
        if alpha_ratio < 0.55:
            flags.append(f"low_prose({alpha_ratio:.2f})")
        if short_ratio > 0.35:
            flags.append(f"fragmented({short_ratio:.2f})")
        if head_junk > 0.15:
            flags.append(f"junk_start({head_junk:.2f})")

        rows.append({
            "arxiv_id":     stem,
            "chars":        cleaned_chars,
            "words":        words,
            "source_chars": source_chars,
            "compression":  round(compression, 3),
            "alpha_ratio":  round(alpha_ratio, 3),
            "math_ratio":   round(math_ratio, 3),
            "n_sections":   n_sections,
            "hollow_ratio": round(hollow_ratio, 3),
            "latex_cmds":   latex_cmds,
            "flags":        flags,
            "preview":      text[:300].replace("\n", " "),
        })

    flagged = [r for r in rows if r["flags"]]

    def _count(key):
        return sum(1 for r in rows if any(key in f for f in r["flags"]))

    return {
        "run_id":          run_id,
        "total_cleaned":   len(txt_files),
        "sampled":         len(rows),
        "flagged":         len(flagged),
        "flag_rate_pct":   round(100 * len(flagged) / max(len(rows), 1), 1),
        "avg_chars":       round(sum(r["chars"]       for r in rows) / len(rows)),
        "avg_words":       round(sum(r["words"]       for r in rows) / len(rows)),
        "avg_compression": round(sum(r["compression"] for r in rows) / len(rows), 3),
        "avg_alpha_ratio": round(sum(r["alpha_ratio"] for r in rows) / len(rows), 3),
        "breakdown": {
            # under-cleaning
            "latex_residue":   _count("latex_residue"),
            "env_tags":        _count("env_tags"),
            "item_residue":    _count("item_residue"),
            "dollar_signs":    _count("dollar_signs"),
            "table_align":     _count("table_align"),
            "stray_braces":    _count("stray_braces"),
            "backslash_misc":  _count("backslash_misc"),
            "email_leak":      _count("email_leak"),
            "encoding_errors": _count("encoding_errors"),
            # over-cleaning / structural
            "hollow_sections": _count("hollow"),
            "math_heavy":      _count("math_heavy"),
            "consec_math":     _count("consec_math"),
            "over_stripped":   _count("over_stripped"),
            "under_stripped":  _count("under_stripped"),
            # content quality
            "low_prose":       _count("low_prose"),
            "fragmented":      _count("fragmented"),
            "junk_start":      _count("junk_start"),
        },
        "flagged_examples": flagged[:25],
    }


@app.function(
    image=image,
    volumes={VPATH: volume},
    secrets=[hf_secret],   # injects HF_TOKEN
    timeout=3600,          # large datasets can take a while to push
    memory=4096,
)
def push_to_hub_fn(run_id: str, hf_repo: str):
    """
    Push a completed run's HuggingFace Dataset to the Hub.

    Runs entirely on Modal — no data passes through your local machine.
    The dataset is loaded from the Volume and uploaded directly to HF.

    The target repo will be created if it doesn't exist (as a dataset
    repo, not a model repo). Set the repo to private in HF settings
    afterwards if needed.
    """
    from datasets import load_from_disk
    import os

    volume.reload()
    ds_path = Path(rdataset(run_id))
    if not ds_path.exists():
        raise FileNotFoundError(
            f"Dataset for run '{run_id}' not found at {ds_path}. "
            "Has the pipeline finished successfully?"
        )

    print(f"Loading dataset from {ds_path} ...")
    dd = load_from_disk(str(ds_path))
    print(f"Pushing {len(dd['train']):,} examples to {hf_repo} ...")
    dd.push_to_hub(hf_repo, token=os.environ["HF_TOKEN"])
    print("Push complete.")


# ── Small utility functions used by the orchestrator ─────────────────────────

@app.function(image=image, volumes={VPATH: volume}, timeout=60)
def save_run_meta(run_id: str, meta: dict):
    """Persist run metadata to meta.json. Called at start and end of a run."""
    Path(rdir(run_id)).mkdir(parents=True, exist_ok=True)
    with open(rmeta(run_id), "w") as f:
        json.dump(meta, f, indent=2)
    volume.commit()


@app.function(image=image, volumes={VPATH: volume}, timeout=60)
def read_tar_queue(run_id: str) -> list[dict]:
    """Read tar_queue.jsonl back from the Volume."""
    volume.reload()
    return [
        json.loads(l)
        for l in Path(rqueue(run_id)).read_text().splitlines()
        if l.strip()
    ]


@app.function(image=image, volumes={VPATH: volume}, timeout=60)
def list_extracted_ids(run_id: str) -> list[str]:
    """
    Return sorted list of paper IDs that were successfully extracted.
    Directory names use '_' instead of '/' (safe filesystem names).
    The original arXiv ID can be reconstructed later where needed.
    """
    volume.reload()
    d = Path(rextracted(run_id))
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())


@app.function(image=image, volumes={VPATH: volume}, timeout=60)
def list_all_runs() -> list[dict]:
    """Return meta.json contents for all runs, sorted by run_id."""
    volume.reload()
    base = Path(f"{VPATH}/runs")
    if not base.exists():
        return []
    out = []
    for p in sorted(base.iterdir()):
        mf = p / "meta.json"
        if mf.exists():
            try:
                out.append(json.loads(mf.read_text()))
            except Exception:
                pass
    return out


@app.function(image=image, volumes={VPATH: volume}, timeout=60)
def count_lines(path: str) -> int:
    """Count non-empty lines in a file on the Volume. Used to report
    how many paper IDs or tars were loaded from a previous run."""
    volume.reload()
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for l in p.read_text().splitlines() if l.strip())


@app.function(image=image, volumes={VPATH: volume}, timeout=120, memory=1024)
def load_cleaned_manifest(run_id: str, arxiv_ids: list[str]) -> list[dict]:
    """
    Read the char/token metadata for a set of already-cleaned papers
    without re-loading the full text. Used on resume to reconstruct
    the manifest for papers cleaned in a previous run.
    """
    volume.reload()
    c_dir = Path(rcleaned(run_id))
    result = []
    for arxiv_id in arxiv_ids:
        safe = arxiv_id.replace("/", "_")
        p = c_dir / f"{safe}.txt"
        if p.exists():
            chars = p.stat().st_size
            result.append({"arxiv_id": arxiv_id, "chars": chars,
                            "tokens_est": chars // 4})
    return result


@app.function(image=image, volumes={VPATH: volume}, timeout=60)
def check_run_state(run_id: str) -> dict:
    """
    Inspect the Volume and return which steps have already completed
    for the given run_id.

    Used by the orchestrator to decide which steps to skip on resume.
    Returns a dict of booleans, one per step:

      {
        "meta":     True if meta.json exists (run was started),
        "ids":      True if paper_ids.txt exists and non-empty,
        "queue":    True if tar_queue.jsonl exists and non-empty,
        "extracted_ids": [list of already-extracted paper dir names],
        "cleaned_ids":   [list of already-cleaned arxiv_ids],
        "dataset":  True if hf_dataset/dataset_info.json exists,
      }

    The orchestrator uses extracted_ids and cleaned_ids to skip only
    the tars / paper batches that are already done, rather than
    re-running everything.
    """
    volume.reload()

    base = Path(rdir(run_id))

    # Step 1 output
    ids_path  = Path(rids(run_id))
    ids_done  = ids_path.exists() and ids_path.stat().st_size > 0

    # Step 2 output
    queue_path = Path(rqueue(run_id))
    queue_done = queue_path.exists() and queue_path.stat().st_size > 0

    # Step 3 output — which paper dirs already exist in extracted/
    ext_dir = Path(rextracted(run_id))
    extracted_ids = sorted(p.name for p in ext_dir.iterdir() if p.is_dir()) \
        if ext_dir.exists() else []

    # Step 4 output — which .txt files already exist in cleaned/
    cln_dir = Path(rcleaned(run_id))
    cleaned_ids = [p.stem for p in cln_dir.glob("*.txt")] \
        if cln_dir.exists() else []

    # Step 5 output — HF dataset directory with sentinel file
    ds_done = (Path(rdataset(run_id)) / "dataset_info.json").exists()

    return {
        "meta":          (base / "meta.json").exists(),
        "ids":           ids_done,
        "queue":         queue_done,
        "extracted_ids": extracted_ids,
        "cleaned_ids":   cleaned_ids,
        "dataset":       ds_done,
    }


@app.function(
    image=image,
    volumes={VPATH: volume},
    timeout=1800,
    memory=2048,
    cpu=2,
)
def test_clean_fn(run_id: str, sample_n: int = 500) -> dict:
    """
    Clean a random sample of papers from the global cache into a numbered
    test directory: runs/{run_id}/test_cleans/test_{N}/

    Useful for quickly testing cleaning-code changes without re-running
    the full pipeline. Each call creates a new test_N directory so multiple
    test runs can coexist and be compared.

    The test directory can be audited with:
      --audit --run-id-arg <run_id> --audit-test <N>
    """
    import random

    volume.reload()
    cache_dir = Path(PAPER_CACHE_PATH)

    all_papers = [p for p in cache_dir.iterdir() if p.is_dir()] \
        if cache_dir.exists() else []
    if not all_papers:
        raise RuntimeError(
            "No papers in paper_cache. Run the pipeline at least once first."
        )

    # Auto-number the test directory
    test_base = Path(rdir(run_id)) / "test_cleans"
    test_base.mkdir(parents=True, exist_ok=True)
    existing = sorted(test_base.glob("test_*"))
    test_num  = len(existing) + 1
    test_dir  = test_base / f"test_{test_num}"
    test_dir.mkdir(parents=True, exist_ok=True)

    rng    = random.Random()   # unseeded — different sample each call
    sample = rng.sample(all_papers, min(sample_n, len(all_papers)))

    cleaned = failed = too_short = 0
    for paper_dir in sample:
        arxiv_id = paper_dir.name
        tex = find_tex(paper_dir)
        if tex is None:
            failed += 1
            continue
        raw = read_file(tex)
        if raw is None:
            failed += 1
            continue
        raw  = resolve_inputs(raw, paper_dir)
        text = clean_latex(raw)
        if len(text) < MIN_CHARS:
            too_short += 1
            continue
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]
        (test_dir / f"{arxiv_id}.txt").write_text(text, encoding="utf-8")
        cleaned += 1

    volume.commit()
    print(f"Test clean {test_num}: {cleaned} cleaned, "
          f"{too_short} too short, {failed} failed  →  {test_dir}")
    return {
        "test_num":  test_num,
        "test_dir":  str(test_dir),
        "sampled":   len(sample),
        "cleaned":   cleaned,
        "too_short": too_short,
        "failed":    failed,
    }


@app.function(
    image=image,
    volumes={VPATH: volume},
    timeout=3600,
    memory=512,
)
def backfill_cache_fn() -> int:
    """
    One-time migration: copy already-extracted papers from all existing
    runs into the global paper cache so future runs can reuse them.

    Safe to run multiple times — skips papers already in the cache.
    Returns the number of paper directories added to the cache.
    """
    import shutil

    volume.reload()
    runs_base  = Path(f"{VPATH}/runs")
    cache_dir  = Path(PAPER_CACHE_PATH)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _TEX_KEEP = {".tex", ".bbl"}

    added = skipped = 0
    for run_dir in sorted(runs_base.iterdir()):
        extracted_dir = run_dir / "extracted"
        if not extracted_dir.exists():
            continue
        for paper_dir in extracted_dir.iterdir():
            if not paper_dir.is_dir():
                continue
            dest = cache_dir / paper_dir.name
            if dest.exists():
                skipped += 1
                continue
            dest.mkdir(parents=True, exist_ok=True)
            # Copy only tex/bbl files — skip figures, .sty, etc.
            for f in paper_dir.iterdir():
                if f.is_file() and (f.suffix in _TEX_KEEP or f.suffix == ""):
                    shutil.copy2(f, dest / f.name)
            if any(dest.iterdir()):
                added += 1
            else:
                dest.rmdir()  # nothing useful — don't cache empty dirs

    volume.commit()
    print(f"Cache backfill complete: {added} added, {skipped} already cached.")
    return added


# ─────────────────────────────────────────────────────────────────────────────
# Local entrypoint — runs on your machine, orchestrates Modal functions
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    papers:           int  = 10_000,  # number of papers to collect
    start:            str  = "2401",  # start month YYMM (Jan 2024)
    end:              str  = "2512",  # end month YYMM   (Dec 2025)
    fetch_metadata:   bool = False,   # --fetch-metadata  (one-time setup)
    list_runs:        bool = False,   # --list-runs
    push_to_hub:      bool = False,   # --push-to-hub
    run_id_arg:        str  = "",      # --run-id-arg  (resume or push-to-hub)
    hf_repo:           str  = "",      # --hf-repo     (for push-to-hub)
    resume:            bool = False,   # --resume      (continue a failed run)
    backfill_cache:    bool = False,   # --backfill-cache  (one-time migration)
    rebuild_dataset:   bool = False,   # --rebuild-dataset  (rebuild HF dataset from all cleaned files)
    audit:             bool = False,   # --audit  (sample and score cleaning quality)
    audit_n:           int  = 200,     # --audit-n  (how many papers to sample)
    audit_test:        int  = 0,       # --audit-test N  (audit test_clean N instead of main cleaned/)
    test_clean:        bool = False,   # --test-clean  (clean a sample into a numbered test dir)
    test_n:            int  = 500,     # --test-n  (sample size for --test-clean)
):
    """
    Orchestrator — coordinates the pipeline steps.

    This function runs locally (on your machine) but every .remote()
    call executes on Modal's infrastructure. The local process just
    waits for results and prints progress — no data is downloaded.

    Resume behaviour:
      Pass --resume --run-id-arg <id> to continue a failed or interrupted
      run from where it stopped. Each step checks whether its output
      already exists on the Volume and skips itself if so.

      Step 3 (extraction) is partially resumable: only tars whose papers
      are not yet in extracted/ are re-downloaded.
      Step 4 (cleaning) is partially resumable: clean_batch skips any
      paper whose .txt file already exists in cleaned/.

    Steps:
      0. fetch_manifest      — download S3 manifest (shared, once)
      1. filter_metadata     — find paper IDs in date/category window
      2. build_tar_queue     — rank S3 tars by density, trim to target
      3. download_and_extract_tar (parallel) — one container per tar
      4. clean_batch         (parallel) — CLEAN_BATCH_SIZE papers per container
      5. build_hf_dataset    — assemble final Dataset from cleaned text
    """

    # ── Utility modes — no pipeline run ──────────────────────────────────────

    if test_clean:
        if not run_id_arg:
            print("ERROR: --test-clean requires --run-id-arg <run_id>")
            return
        print(f"Running test clean ({test_n} papers) for run {run_id_arg} ...")
        result = test_clean_fn.remote(run_id_arg, test_n)
        print(f"\nTest clean {result['test_num']} complete:")
        print(f"  Sampled  : {result['sampled']}")
        print(f"  Cleaned  : {result['cleaned']}")
        print(f"  Too short: {result['too_short']}")
        print(f"  Failed   : {result['failed']}")
        print(f"  Directory: {result['test_dir']}")
        print(f"\nAudit this test clean:")
        print(f"  modal run pipeline.py --audit "
              f"--run-id-arg {run_id_arg} --audit-test {result['test_num']} --audit-n {test_n}")
        return

    if audit:
        if not run_id_arg:
            print("ERROR: --audit requires --run-id-arg <run_id>")
            return
        override_dir = ""
        if audit_test > 0:
            override_dir = f"{rdir(run_id_arg)}/test_cleans/test_{audit_test}"
            print(f"Auditing test_clean {audit_test} for run {run_id_arg} ...")
        else:
            print(f"Auditing {audit_n} sampled papers from run {run_id_arg} ...")
        report = audit_fn.remote(run_id_arg, audit_n, override_dir)
        if "error" in report:
            print(f"ERROR: {report['error']}")
            return

        print(f"\n{'='*60}")
        print(f"AUDIT REPORT   run={run_id_arg}")
        print(f"{'='*60}")
        print(f"Total cleaned files : {report['total_cleaned']:,}")
        print(f"Sampled             : {report['sampled']}")
        print(f"Flagged             : {report['flagged']}  "
              f"({report['flag_rate_pct']}%)")
        print(f"Avg chars / paper   : {report['avg_chars']:,}")
        print(f"Avg words / paper   : {report['avg_words']:,}")
        print(f"Avg compression     : {report['avg_compression']:.3f}  "
              f"(cleaned chars / source .tex bytes)")
        print(f"Avg alpha ratio     : {report['avg_alpha_ratio']:.3f}  "
              f"(fraction of non-whitespace chars that are letters)")

        groups = {
            "Under-cleaning": [
                "latex_residue", "env_tags", "item_residue", "dollar_signs",
                "table_align", "stray_braces", "backslash_misc",
                "email_leak", "encoding_errors",
            ],
            "Over-cleaning / structural": [
                "hollow_sections", "math_heavy", "consec_math",
                "over_stripped", "under_stripped",
            ],
            "Content quality": ["low_prose", "fragmented", "junk_start"],
        }
        bd = report["breakdown"]
        n  = max(report["sampled"], 1)
        for group, keys in groups.items():
            print(f"\n  {group}:")
            for k in keys:
                count = bd.get(k, 0)
                bar   = "█" * min(count, 30)
                pct   = 100 * count / n
                print(f"    {k:<22} {count:3d} ({pct:4.1f}%)  {bar}")

        if report["flagged_examples"]:
            shown = report["flagged_examples"][:10]
            print(f"\nFlagged examples  ({len(shown)} of "
                  f"{report['flagged']} shown):")
            for ex in shown:
                print(f"\n  [{ex['arxiv_id']}]")
                print(f"  flags       : {ex['flags']}")
                print(f"  chars={ex['chars']:,}  words={ex['words']:,}  "
                      f"compression={ex['compression']}  "
                      f"math_ratio={ex['math_ratio']}")
                print(f"  preview     : {ex['preview'][:200]}")

        flag_pct = report["flag_rate_pct"]
        verdict  = (
            "CLEAN — cleaning looks good"         if flag_pct < 10 else
            "SOME NOISE — minor issues detected"  if flag_pct < 30 else
            "NEEDS REVIEW — significant problems"
        )
        print(f"\nVerdict: {verdict}")
        print(f"{'='*60}\n")
        return

    if backfill_cache:
        print("Backfilling paper cache from all existing runs ...")
        added = backfill_cache_fn.remote()
        print(f"Done. {added} papers added to cache.")
        return

    if rebuild_dataset:
        if not run_id_arg:
            print("ERROR: --rebuild-dataset requires --run-id-arg <run_id>")
            return
        print(f"Rebuilding HF dataset for run {run_id_arg} from all cleaned files ...")
        ds_path, _ = build_hf_dataset.remote(run_id_arg)
        print(f"Done. Dataset at {ds_path}")
        return

    if fetch_metadata:
        print("Fetching Kaggle metadata to Volume (~4 GB, ~10 min) ...")
        # Call the Modal function (named fetch_metadata_fn to avoid shadowing
        # the local parameter)
        fetch_metadata_fn.remote()
        return

    if list_runs:
        runs = list_all_runs.remote()
        if not runs:
            print("No runs found.")
            return
        print(f"\n{'ID':<10} {'status':<10} {'papers':<8} {'tokens':<10} "
              f"{'range':<12} {'started'}")
        print("-" * 72)
        for r in runs:
            tok = (f"{r['tokens_est']/1e6:.0f}M"
                   if r.get("tokens_est") else "-")
            rng = f"{r.get('start','?')}–{r.get('end','?')}"
            print(
                f"{r['run_id']:<10} {r.get('status','?'):<10} "
                f"{r.get('cleaned', r.get('papers','?')):<8} {tok:<10} "
                f"{rng:<12} {r.get('started','?')}"
            )
        print()
        return

    if push_to_hub:
        if not run_id_arg or not hf_repo:
            print("Usage:  modal run pipeline.py --push-to-hub "
                  "--run-id-arg <id> --hf-repo <user/repo>")
            return
        print(f"Pushing run {run_id_arg} to {hf_repo} ...")
        push_to_hub_fn.remote(run_id_arg, hf_repo)
        return

    # ── Resolve run_id ────────────────────────────────────────────────────────

    if resume:
        # Resume mode: run_id_arg is required
        if not run_id_arg:
            print("ERROR: --resume requires --run-id-arg <run_id>")
            print("       Use --list-runs to find the run_id to resume.")
            return
        run_id = run_id_arg
        print(f"\n{'='*56}")
        print(f"RESUMING run_id={run_id}")
        print(f"{'='*56}\n")
        print("Checking which steps have already completed ...")
        state = check_run_state.remote(run_id)
    else:
        # Fresh run: generate a new run_id
        run_id = str(uuid.uuid4())[:8]
        state  = {
            "meta": False, "ids": False, "queue": False,
            "extracted_ids": [], "cleaned_ids": [], "dataset": False,
        }

    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    if not resume:
        print(f"\n{'='*56}")
        print(f"run_id:   {run_id}")
        print(f"papers:   {papers:,}")
        print(f"range:    {start} → {end}")
        print(f"started:  {ts}")
        print(f"{'='*56}\n")

    # Write/update run metadata so --list-runs always shows current state
    save_run_meta.remote(run_id, {
        "run_id":  run_id,
        "started": ts,
        "papers":  papers,
        "start":   start,
        "end":     end,
        "status":  "running",
    })

    # ── Step 0: manifest ─────────────────────────────────────────────────────
    # Always runs — fetch_manifest is idempotent (skips if file present).
    print("[0/5] Checking manifest ...")
    fetch_manifest.remote()

    # ── Step 1: filter metadata ───────────────────────────────────────────────
    if state["ids"]:
        # paper_ids.txt already exists from a previous run — read its count
        # instead of re-scanning the 4 GB metadata file.
        n_matched = count_lines.remote(rids(run_id))
        print(f"[1/5] Filter metadata — SKIPPED (resuming, {n_matched:,} IDs already on Volume)")
    else:
        print("[1/5] Filtering metadata ...")
        n_matched = filter_metadata.remote(run_id, start, end, papers)
        if n_matched == 0:
            print("ERROR: no papers matched. "
                  "Run --fetch-metadata first, or widen --start/--end.")
            return

    # ── Step 2: build tar queue ───────────────────────────────────────────────
    if state["queue"]:
        n_tars = count_lines.remote(rqueue(run_id))
        print(f"[2/5] Build tar queue — SKIPPED (resuming, {n_tars} tars already queued)")
    else:
        print("[2/5] Building tar queue ...")
        n_tars = build_tar_queue.remote(run_id, papers)
        print(f"      {n_tars} tars queued")

    # ── Step 3: download + extract ────────────────────────────────────────────
    tar_records = read_tar_queue.remote(run_id)

    # On resume, skip tars whose papers are already fully extracted.
    # A tar is considered done if every one of its wanted paper IDs
    # already has a directory in extracted/.
    already_extracted = set(state["extracted_ids"])
    tars_to_run = [
        r for r in tar_records
        if not all(pid.replace("/", "_") in already_extracted
                   for pid in r["ids"])
    ]

    if not tars_to_run:
        total_extracted = len(already_extracted)
        print(f"[3/5] Download + extract — SKIPPED "
              f"(resuming, {total_extracted:,} papers already extracted)")
    else:
        skipped_tars = len(tar_records) - len(tars_to_run)
        if skipped_tars:
            print(f"[3/5] Downloading + extracting tars in parallel ...")
            print(f"      Skipping {skipped_tars} already-extracted tars, "
                  f"running {len(tars_to_run)} remaining ...")
        else:
            print("[3/5] Downloading + extracting tars in parallel ...")

        extract_results = list(
            download_and_extract_tar.starmap(
                [(run_id, r) for r in tars_to_run],
                return_exceptions=True,
            )
        )
        good_extracts   = [r for r in extract_results if isinstance(r, dict)]
        bad_extracts    = [r for r in extract_results if not isinstance(r, dict)]
        newly_extracted = sum(r["extracted"] for r in good_extracts)
        total_extracted = len(already_extracted) + newly_extracted
        failed_tars     = [r for r in good_extracts if r["error"]]
        print(f"      {newly_extracted:,} new + {len(already_extracted):,} existing "
              f"= {total_extracted:,} total source files")
        if bad_extracts:
            print(f"      WARNING: {len(bad_extracts)} tar containers raised exceptions")
            for exc in bad_extracts[:3]:
                print(f"        {exc}")
        if failed_tars:
            print(f"      WARNING: {len(failed_tars)} tars had tar-level errors: "
                  f"{[r['tar_key'] for r in failed_tars]}")

        if total_extracted == 0:
            raise RuntimeError(
                "0 papers extracted. All download containers failed.\n"
                "Likely causes:\n"
                "  - AWS credentials missing or wrong (check Modal secret 'aws-arxiv')\n"
                "  - Secret name mismatch — verify with: modal secret list\n"
                "  - Requester-pays not authorised for this AWS account\n"
                "Re-run after fixing credentials; this run_id can be resumed:\n"
                f"  modal run pipeline.py --resume --run-id-arg {run_id}"
            )

    # ── Step 4: clean LaTeX ───────────────────────────────────────────────────
    all_ids         = list_extracted_ids.remote(run_id)[:papers]
    already_cleaned = set(state["cleaned_ids"])

    # Only dispatch IDs that genuinely need cleaning — no wasted containers
    # for papers whose .txt already exists.
    pending_ids = [pid for pid in all_ids if pid not in already_cleaned]

    # Manifest for already-cleaned papers (metadata only, no text reads).
    done_ids       = [pid for pid in all_ids if pid in already_cleaned]
    prior_manifest = load_cleaned_manifest.remote(run_id, done_ids) if done_ids else []

    if not pending_ids:
        manifest = prior_manifest
        print(f"[4/5] Clean LaTeX — SKIPPED "
              f"(resuming, {len(manifest):,} papers already cleaned)")
    else:
        if done_ids:
            print(f"[4/5] Cleaning LaTeX sources in parallel ...")
            print(f"      {len(done_ids):,} already cleaned, "
                  f"{len(pending_ids):,} remaining ...")
        else:
            print("[4/5] Cleaning LaTeX sources in parallel ...")

        pending_batches = [
            pending_ids[i : i + CLEAN_BATCH_SIZE]
            for i in range(0, len(pending_ids), CLEAN_BATCH_SIZE)
        ]

        batch_results = list(
            clean_batch.starmap(
                [(run_id, batch) for batch in pending_batches],
                order_outputs=False,
                return_exceptions=True,
            )
        )
        new_manifest = []
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"      WARNING: a clean_batch container raised: {result}")
                continue
            for rec in result:
                if rec is not None:
                    new_manifest.append(rec)

        manifest = prior_manifest + new_manifest
        print(f"      {len(new_manifest):,} newly cleaned + "
              f"{len(prior_manifest):,} previously cleaned "
              f"= {len(manifest):,} total")

    if len(manifest) == 0:
        raise RuntimeError(
            "No papers were successfully cleaned (manifest is empty). "
            "LaTeX extraction likely failed for all papers in this run.\n"
            "Inspect the Volume to diagnose:\n"
            f"  modal shell --volume arxiv-pipeline-vol\n"
            f"  ls /data/runs/{run_id}/extracted/\n"
            f"  ls /data/runs/{run_id}/cleaned/"
        )

    if len(manifest) < papers * 0.8:
        print(
            f"WARNING: only {len(manifest):,} papers cleaned vs {papers:,} "
            "requested. LaTeX source may be missing for many papers in this "
            "date range. Consider checking a few paper directories manually:\n"
            f"  modal shell --volume arxiv-pipeline-vol\n"
            f"  ls /data/runs/{run_id}/extracted/ | head -5"
        )

    # ── Step 5: build HF dataset ──────────────────────────────────────────────
    if state["dataset"]:
        print(f"[5/5] Build HF dataset — SKIPPED (resuming, dataset already exists)")
        ds_path = rdataset(run_id)
        total_tokens = sum(m["tokens_est"] for m in manifest)
    else:
        print("[5/5] Building HuggingFace dataset ...")
        ds_path, total_tokens = build_hf_dataset.remote(run_id)

    # ── Finalise ──────────────────────────────────────────────────────────────
    save_run_meta.remote(run_id, {
        "run_id":      run_id,
        "started":     ts,
        "finished":    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "papers":      papers,
        "start":       start,
        "end":         end,
        "cleaned":     len(manifest),
        "tokens_est":  total_tokens,
        "status":      "done",
        "ds_path":     ds_path,
    })

    print(f"\n{'='*56}")
    print(f"DONE   run_id={run_id}")
    print(f"papers:   {len(manifest):,}")
    print(f"tokens:   ~{total_tokens / 1e6:.0f}M estimated")
    print(f"volume:   {ds_path}")
    print(f"\nNext steps:")
    print(f"  Push to HF Hub:")
    print(f"    modal run pipeline.py --push-to-hub "
          f"--run-id-arg {run_id} --hf-repo your-username/your-dataset-name")
    print(f"  Inspect on Modal:")
    print(f"    modal shell --volume arxiv-pipeline-vol")
    print(f"    ls /data/runs/{run_id}/")
    print(f"{'='*56}\n")