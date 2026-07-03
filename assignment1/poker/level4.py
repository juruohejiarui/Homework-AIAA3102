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
    rank_a, cards_a = best_hand(a)
    rank_b, cards_b = best_hand(b)
    
    from hand_rank import HandRank

    def score(rank: HandRank, cards: list[str]) -> tuple[int, tuple[int, ...]]:
        values = tuple(RANK_STR.index(card[0]) + 2 for card in cards)
        return rank.value, values

    score_a = score(rank_a, cards_a)
    score_b = score(rank_b, cards_b)

    if score_a > score_b:
        return 1
    if score_a < score_b:
        return -1
    return 0