"""Level 5 — the 'Skew' variant.

This is a DIFFERENT card game from the one in Levels 1-4. Read the Skew rules in
the README carefully: the ace is low and a flush is ranked differently. Implement
`skew_best_hand` from that spec — do NOT assume standard poker; your poker
instincts will be wrong here.
"""
from __future__ import annotations

from collections import Counter

from hand_rank import HandRank
from poker_common import Card, five_card_hands, parse  # noqa: F401


def _skew_value(card: Card) -> int:
    # In Skew, ace is always the lowest rank.
    return 1 if card.rank == "A" else card.value


def _classify_five_skew(cards: tuple[Card, ...]) -> HandRank:
    values = [_skew_value(card) for card in cards]
    counts = Counter(values)
    pattern = sorted(counts.values(), reverse=True)
    is_flush = len({card.suit for card in cards}) == 1

    distinct = sorted(set(values))
    is_straight = len(distinct) == 5 and distinct[-1] - distinct[0] == 4

    # Skew has no separate royal flush label.
    if is_straight and is_flush:
        return HandRank.STRAIGHT_FLUSH
    if pattern == [4, 1]:
        return HandRank.FOUR_OF_A_KIND
    if pattern == [3, 2]:
        return HandRank.FULL_HOUSE
    if is_straight:
        return HandRank.STRAIGHT
    if is_flush:
        return HandRank.FLUSH
    if pattern == [3, 1, 1]:
        return HandRank.THREE_OF_A_KIND
    if pattern == [2, 2, 1]:
        return HandRank.TWO_PAIR
    if pattern == [2, 1, 1, 1]:
        return HandRank.ONE_PAIR
    return HandRank.HIGH_CARD


def _order_five_skew(cards: tuple[Card, ...], rank: HandRank) -> list[Card]:
    values = Counter(_skew_value(card) for card in cards)

    if rank in (HandRank.STRAIGHT, HandRank.STRAIGHT_FLUSH):
        return sorted(cards, key=_skew_value, reverse=True)

    if rank in (HandRank.FLUSH, HandRank.HIGH_CARD):
        return sorted(cards, key=_skew_value, reverse=True)

    return sorted(cards, key=lambda card: (values[_skew_value(card)], _skew_value(card)), reverse=True)


_SKEW_RANK_STRENGTH = {
    HandRank.HIGH_CARD: 0,
    HandRank.ONE_PAIR: 1,
    HandRank.TWO_PAIR: 2,
    HandRank.THREE_OF_A_KIND: 3,
    HandRank.FLUSH: 4,
    HandRank.STRAIGHT: 5,
    HandRank.FULL_HOUSE: 6,
    HandRank.FOUR_OF_A_KIND: 7,
    HandRank.STRAIGHT_FLUSH: 8,
    HandRank.ROYAL_FLUSH: 8,
}


def skew_best_hand(seven: list[str]) -> tuple[HandRank, list[str]]:
    parsed = parse(seven)
    scored: list[tuple[HandRank, list[Card]]] = []

    for hand in five_card_hands(parsed):
        rank = _classify_five_skew(hand)
        ordered = _order_five_skew(hand, rank)
        scored.append((rank, ordered))

    best_rank, best_cards = max(
        scored,
        key=lambda item: (
            _SKEW_RANK_STRENGTH[item[0]],
            tuple(_skew_value(card) for card in item[1]),
        ),
    )
    return best_rank, [str(card) for card in best_cards]
