"""Ring 5 — append-only compositions. Nothing imports this.

One file per entrypoint, named `*_v1.py`. A new variant is a new file; an old
one is never edited, so a result stays reproducible from the file that made it.
`_runtime.py` is the exception — it is the composition root, not an experiment.
"""
