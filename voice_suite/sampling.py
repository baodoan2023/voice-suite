"""Deterministic utterance subset selection for cost-bounded eval runs."""
from __future__ import annotations

import random

from voice_suite.protocol import Utterance


def sample_items(items: list[Utterance], limit: int | None,
                 seed: int = 0) -> list[Utterance]:
    """Return up to *limit* items, chosen reproducibly by *seed*.

    The result is always sorted by ``id`` for stable downstream ordering.
    Same ``(items, limit, seed)`` → same subset, so ``run`` and ``score``
    can independently select the *same* utterances and stay in sync.

    - ``limit`` is ``None`` or ``>= len(items)`` → all items (id-sorted).
    - ``limit <= 0`` → empty list.
    """
    ordered = sorted(items, key=lambda it: it.id)
    if limit is None or limit >= len(ordered):
        return ordered
    if limit <= 0:
        return []
    rng = random.Random(seed)
    picked = rng.sample(ordered, limit)
    return sorted(picked, key=lambda it: it.id)
