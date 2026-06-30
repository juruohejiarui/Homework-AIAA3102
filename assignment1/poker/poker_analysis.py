"""Public interface. FROZEN — do not modify. It wires your level modules into
the two functions the grader calls. Implement the levels in level1-4.py.
"""
from level3 import best_hand
from level4 import compare_hands  # noqa: F401

from hand_rank import HandRank


def get_best_hand(cards: list[str]) -> tuple[HandRank, list[str]]:
    return best_hand(cards)
