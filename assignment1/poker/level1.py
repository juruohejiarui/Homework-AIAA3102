"""Level 1 — classify a single 5-card hand into a HandRank.

This is written for you, but it has bugs. Some are easy to spot on the worked
examples; others only show up on less-common hands. Find them ALL and fix them
in place (do not rewrite the file from scratch). You are NOT told how many there
are. Document each in BUGS.md.
"""
from __future__ import annotations

from collections import Counter

from hand_rank import HandRank
from poker_common import Card


def classify(cards: tuple[Card, ...]) -> HandRank:
    counts = Counter(c.value for c in cards)
    is_flush = len({c.suit for c in cards}) == 1
    distinct = sorted(counts)
    is_straight = False
    if len(distinct) == 5 and distinct[4] - distinct[0] == 4:
        is_straight = True
    pattern = sorted(counts.values(), reverse=True)

    if is_straight and is_flush:
        return HandRank.STRAIGHT_FLUSH
    if pattern == [4]:
        return HandRank.FOUR_OF_A_KIND
    if pattern == [2, 3]:
        return HandRank.FULL_HOUSE
    if is_flush:
        return HandRank.FLUSH
    if is_straight:
        return HandRank.STRAIGHT
    if pattern == [3, 1, 1]:
        return HandRank.THREE_OF_A_KIND
    if pattern == [2, 2]:
        return HandRank.TWO_PAIR
    if pattern == [2, 1, 1, 1]:
        return HandRank.ONE_PAIR
    return HandRank.HIGH_CARD
