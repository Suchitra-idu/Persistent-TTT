"""Ring 3 — concrete implementations, real and fake side by side.

Nothing is re-exported here: an adapter is chosen by name at the composition
root in Ring 5, and a package-level import would drag transformers, wandb and
modal into every process that only wanted the fakes.
"""
