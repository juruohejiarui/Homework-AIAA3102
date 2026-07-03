"""Level 2 — order one 5-card hand into canonical compare order.

Like Level 1, this is written for you but has bugs. Some are easy to spot on the
worked examples — but make sure you find them ALL, not just the obvious ones. Fix
them in place (don't rewrite the file). You MUST keep using `classify`.

Canonical order = most significant cards first, then kickers high->low (e.g. two
pair: high pair, low pair, kicker).
"""
from __future__ import annotations

from collections import Counter

from hand_rank import HandRank
from level1 import classify
from poker_common import Card


def order_five(cards: tuple[Card, ...]) -> list[Card]:
    rank = classify(cards)
    counts = Counter(c.value for c in cards)
    if rank in (HandRank.STRAIGHT, HandRank.STRAIGHT_FLUSH, HandRank.ROYAL_FLUSH):
        # Wheel straight (A-2-3-4-5) is 5-high, so ace must be treated as low.
        if {c.value for c in cards} == {14, 2, 3, 4, 5}:
            return sorted(cards, key=lambda c: 1 if c.value == 14 else c.value, reverse=True)
        return sorted(cards, key=lambda c: c.value, reverse=True)
    if rank in (HandRank.FLUSH, HandRank.HIGH_CARD):
        return sorted(cards, key=lambda c: c.value, reverse=True)
    return sorted(cards, key=lambda c: (counts[c.value], c.value), reverse=True)
