# Conventions

## Comments and documentation

**No excessive comments, no unnecessary documentation.** The code says what it
does; a comment exists only to say *why*, and only when the why is not obvious
from the code.

- Never restate the line below in words. No section-banner comments.
- No per-field commentary on config dataclasses. A field whose name needs a
  paragraph needs a better name.
- Docstrings are one line. Add more only for a contract the signature cannot
  express: an invariant, a unit, an edge case, a failure mode.
- Design rationale lives in `NewCode/PLAN.md` (decisions D1–D14) and
  `NewCode/ARCHITECTURE.md`. Do not restate it in the module that implements
  it — cite the decision id in a few words if it matters.
- A test's name is its documentation. Do not add a comment explaining what a
  test asserts.

Enforced mechanically by `NewCode/tests/architecture/test_comment_budget.py`.
