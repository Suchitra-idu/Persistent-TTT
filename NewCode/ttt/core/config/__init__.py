"""Ring 0 — frozen config dataclasses and their pure resolution.

Env and CLI resolution happens exactly once, in Ring 5, and produces the
frozen objects defined here (D3). Nothing in this package reads the
environment.
"""
