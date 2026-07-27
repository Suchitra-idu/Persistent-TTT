"""Ring 0 — pure logic and math.

May import: ``torch``, ``math``, ``dataclasses``, ``typing`` (D1).
May not import: any outer ring, or ``transformers`` / ``peft`` / ``modal`` /
``wandb`` / ``datasets``.
May not touch: ``os.environ``, ``time.time()``, ``torch.cuda``,
``torch.manual_seed``, ``random.seed``, the filesystem.

Deterministic given its inputs, and fast enough that the whole ring's test
suite runs on a laptop in seconds.
"""
