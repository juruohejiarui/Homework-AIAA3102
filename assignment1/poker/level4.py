"""Level 4 — compare two 7-card hands.

Build this. You MUST use `best_hand` from Level 3 (already imported). Return 1 if
hand A wins, -1 if hand B wins, 0 if they tie. Two hands of the same category
are decided by their cards in compare order (the kickers).
"""
from __future__ import annotations

from level3 import best_hand  # noqa: F401
from poker_common import RANK_STR  # noqa: F401


def compare_hands(a: list[str], b: list[str]) -> int:
    # TODO(student): use best_hand to decide the winner.
    raise NotImplementedError("Level 4: implement compare_hands")
