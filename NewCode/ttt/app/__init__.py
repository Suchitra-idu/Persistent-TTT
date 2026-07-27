"""Ring 4 — use cases: the loops.

Orchestration only. Every substantive decision is delegated to a plugin or a
port. Imports rings 0-2 via interfaces and never a concrete adapter
(enforced by the ``app-never-touches-a-concrete-adapter`` contract).
"""
