"""In-Place TTT, rebuilt under the Research Hexagon.

Six rings, imports point strictly inward:

    experiments -> app -> adapters -> ports -> extensions -> core

See ``ARCHITECTURE.md`` for the map and ``../RESEARCH_ARCHITECTURE.md`` for
the specification the rings come from. The rule is enforced by
``.importlinter`` and ``tests/architecture/``, not by convention.
"""
