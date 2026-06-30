"""Shared plumbing. FROZEN — do not modify this file."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

RANK_STR = "23456789TJQKA"


@dataclass(frozen=True)
class Card:
    rank: str
    suit: str

    @property
    def value(self) -> int:
        return RANK_STR.index(self.rank) + 2

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"


def parse(cards: list[str]) -> list[Card]:
    return [Card(s[0], s[1]) for s in cards]


def five_card_hands(seven: list[Card]) -> Iterator[tuple[Card, ...]]:
    return combinations(seven, 5)
