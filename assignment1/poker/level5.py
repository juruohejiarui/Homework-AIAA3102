"""Level 5 — the 'Skew' variant.

This is a DIFFERENT card game from the one in Levels 1-4. Read the Skew rules in
the README carefully: the ace is low and a flush is ranked differently. Implement
`skew_best_hand` from that spec — do NOT assume standard poker; your poker
instincts will be wrong here.
"""
from __future__ import annotations

from hand_rank import HandRank
from poker_common import Card, five_card_hands, parse  # noqa: F401


def skew_best_hand(seven: list[str]) -> tuple[HandRank, list[str]]:
    # TODO(student): implement the Skew rules (ace low; flush demoted below
    # straight and three of a kind). Return (HandRank, 5 cards in compare order).
    raise NotImplementedError("Level 5: implement the Skew rules from the README")
