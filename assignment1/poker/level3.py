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
    from poker_common import Card
    card_lst = parse(seven)
    combs: list[tuple[HandRank, list[Card]]] = []

    for comb in five_card_hands(card_lst):
        rank = classify(comb)
        combs.append((rank, order_five(comb)))

    best_rank, best_cards = max(
        combs,
        key=lambda item: (item[0].value, tuple(card.value for card in item[1])),
    )

    return best_rank, [str(card) for card in best_cards]