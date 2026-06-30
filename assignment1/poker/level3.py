"""Level 3 — find the best 5-card hand out of 7 cards.

Build this. You MUST use `classify` (Level 1) and `order_five` (Level 2), both
already imported. Enumerate the 5-card hands with `five_card_hands`, score each
by (rank, ordered card values), and return the best one's rank plus its five
cards as strings in compare order.
"""
from __future__ import annotations

from hand_rank import HandRank  # noqa: F401
from level1 import classify  # noqa: F401
from level2 import order_five  # noqa: F401
from poker_common import five_card_hands, parse  # noqa: F401


def best_hand(seven: list[str]) -> tuple[HandRank, list[str]]:
    # TODO(student): enumerate, score, and return (rank, 5 cards in order).
    raise NotImplementedError("Level 3: implement best_hand")
