"""
Pure training utilities, extracted from train_modal.py.
"""


def make_session_schedule(num_docs: int, lo: int, hi: int, rng) -> list:
    """Shuffle all docs, then partition the order into sessions of
    random size n ~ Uniform[lo, hi]. Every doc appears exactly once per
    epoch; only the grouping (and therefore the fast weight carry
    structure) is random. The final session may be shorter than lo."""
    order = rng.permutation(num_docs)
    sessions, i = [], 0
    while i < num_docs:
        n = int(rng.integers(lo, hi + 1))
        sessions.append(order[i:i + n].tolist())
        i += n
    return sessions


def grad_norms(named_groups: dict) -> dict:
    """L2 norm of accumulated gradients per parameter group. The
    grad/new signal is the wiring-broken detector; see observability."""
    import torch

    out = {}
    for name, params in named_groups.items():
        sq = sum(
            p.grad.detach().float().pow(2).sum()
            for p in params if p.grad is not None
        )
        out[name] = float(torch.as_tensor(sq).sqrt())
    return out
