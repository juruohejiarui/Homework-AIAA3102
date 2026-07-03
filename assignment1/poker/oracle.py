"""
oracle.py — Clean-room reference implementation of best_hand and compare_hands.

Deliberately uses a DIFFERENT algorithmic approach from the student's code:

  Student code approach:
    - Counter-based pattern matching for classification
    - Separate order_five function with rank-dispatch
    - Explicit HandRank enum comparisons

  This oracle's approach:
    - Represent each 5-card hand as a single sortable integer score
      (no Counter, no pattern list, no HandRank enum internally)
    - Hand category detected via bit-mask / arithmetic on a sorted value array
    - Tiebreaking encoded directly into the score integer (base-15 encoding)
    - best_hand: pick the combo with the highest score integer, then decode
    - compare_hands: compare score integers directly

The two implementations share only the public data types (HandRank, Card strings)
required by the assignment interface.
"""
from __future__ import annotations

from itertools import combinations

from hand_rank import HandRank

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RANK_ORDER = "23456789TJQKA"   # index 0 → value 2, index 12 → value 14
_SUIT_CHARS  = "SHDC"

# Category codes — deliberately NOT using HandRank.value order here;
# we map to HandRank only at the very end.
_CAT_HIGH_CARD      = 0
_CAT_ONE_PAIR       = 1
_CAT_TWO_PAIR       = 2
_CAT_THREE_OF_A_KIND= 3
_CAT_STRAIGHT       = 4
_CAT_FLUSH          = 5
_CAT_FULL_HOUSE     = 6
_CAT_FOUR_OF_A_KIND = 7
_CAT_STRAIGHT_FLUSH = 8
_CAT_ROYAL_FLUSH    = 9

_CAT_TO_HANDRANK = {
    _CAT_HIGH_CARD:       HandRank.HIGH_CARD,
    _CAT_ONE_PAIR:        HandRank.ONE_PAIR,
    _CAT_TWO_PAIR:        HandRank.TWO_PAIR,
    _CAT_THREE_OF_A_KIND: HandRank.THREE_OF_A_KIND,
    _CAT_STRAIGHT:        HandRank.STRAIGHT,
    _CAT_FLUSH:           HandRank.FLUSH,
    _CAT_FULL_HOUSE:      HandRank.FULL_HOUSE,
    _CAT_FOUR_OF_A_KIND:  HandRank.FOUR_OF_A_KIND,
    _CAT_STRAIGHT_FLUSH:  HandRank.STRAIGHT_FLUSH,
    _CAT_ROYAL_FLUSH:     HandRank.ROYAL_FLUSH,
}

# Base for tiebreak encoding: values run 1..14, use base 15
_BASE = 15


# ---------------------------------------------------------------------------
# Core: encode a 5-card hand as a single integer
# ---------------------------------------------------------------------------

def _card_val(s: str) -> int:
    """'AH' → 14, 'TC' → 10, '2S' → 2"""
    return _RANK_ORDER.index(s[0]) + 2


def _encode(hand: tuple[str, ...]) -> int:
    """
    Encode a 5-card hand (tuple of card strings) as a single comparable integer.

    Layout (big-endian base-15 digits):
        digit 5 (most significant): category code  0..9
        digits 4..0: tiebreak values in compare order (most significant first)

    Compare-order rules encoded here without any reference to the student's logic:
      - Straight / SF / RF : high card first (wheel: 5-4-3-2-A → treat A as 1)
      - Flush / High card   : all five values descending
      - Four of a kind      : quad rank, then kicker
      - Full house          : trips rank, then pair rank
      - Three of a kind     : trips rank, then two kickers descending
      - Two pair            : high pair, low pair, kicker
      - One pair            : pair rank, then three kickers descending
    """
    vals = sorted((_card_val(c) for c in hand), reverse=True)  # e.g. [14,13,12,11,10]
    val_set = set(vals)
    suits = {c[1] for c in hand}

    # --- detect flush & straight ---
    is_flush = len(suits) == 1

    # Normal straight: 5 distinct values spanning exactly 4
    is_straight = (len(val_set) == 5) and (vals[0] - vals[4] == 4)
    # Wheel: A-2-3-4-5
    is_wheel = (val_set == {14, 2, 3, 4, 5})
    if is_wheel:
        is_straight = True

    # --- build frequency table without Counter ---
    freq: dict[int, int] = {}
    for v in vals:
        freq[v] = freq.get(v, 0) + 1

    # Sort by (frequency desc, value desc) — the canonical group order
    groups = sorted(freq.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # groups[i] = (value, count), most important group first

    # --- determine category ---
    counts_only = [g[1] for g in groups]

    if is_straight and is_flush:
        cat = _CAT_ROYAL_FLUSH if val_set == {10, 11, 12, 13, 14} else _CAT_STRAIGHT_FLUSH
    elif counts_only[0] == 4:
        cat = _CAT_FOUR_OF_A_KIND
    elif counts_only[:2] == [3, 2]:
        cat = _CAT_FULL_HOUSE
    elif is_flush:
        cat = _CAT_FLUSH
    elif is_straight:
        cat = _CAT_STRAIGHT
    elif counts_only[0] == 3:
        cat = _CAT_THREE_OF_A_KIND
    elif counts_only[:2] == [2, 2]:
        cat = _CAT_TWO_PAIR
    elif counts_only[0] == 2:
        cat = _CAT_ONE_PAIR
    else:
        cat = _CAT_HIGH_CARD

    # --- build tiebreak sequence (5 values) ---
    if cat in (_CAT_STRAIGHT, _CAT_STRAIGHT_FLUSH, _CAT_ROYAL_FLUSH):
        if is_wheel:
            tb = [5, 4, 3, 2, 1]   # 5-high straight; ace treated as 1
        else:
            tb = sorted(vals, reverse=True)  # already sorted desc
    elif cat in (_CAT_FLUSH, _CAT_HIGH_CARD):
        tb = sorted(vals, reverse=True)
    else:
        # Expand groups back into a flat tiebreak list
        tb = []
        for v, cnt in groups:
            tb.extend([v] * cnt)
        # Pad / trim to exactly 5 (should always be 5 for a valid hand)

    # Encode: cat * BASE^5 + tb[0]*BASE^4 + tb[1]*BASE^3 + ...
    score = cat
    for t in tb:
        score = score * _BASE + t
    return score


# ---------------------------------------------------------------------------
# Decode score back to (HandRank, ordered card strings)
# ---------------------------------------------------------------------------

def _decode_tiebreak(score: int) -> tuple[int, list[int]]:
    """Extract (category, [tb0, tb1, tb2, tb3, tb4]) from an encoded score."""
    tb = []
    for _ in range(5):
        tb.append(score % _BASE)
        score //= _BASE
    cat = score
    tb.reverse()
    return cat, tb


def _ordered_cards(hand: tuple[str, ...], cat: int, tb: list[int]) -> list[str]:
    """
    Given the 5-card hand and its decoded tiebreak sequence, return the cards
    in compare order (matching the tiebreak sequence).

    Strategy: for each tiebreak slot, pick the card whose value matches tb[i],
    preferring cards not yet used.  For the wheel, ace maps to tb value 1.
    """
    remaining = list(hand)
    result: list[str] = []

    is_wheel = (cat in (_CAT_STRAIGHT, _CAT_STRAIGHT_FLUSH)) and (tb == [5, 4, 3, 2, 1])

    for t in tb:
        for i, card in enumerate(remaining):
            v = _card_val(card)
            # In a wheel, ace has tiebreak value 1
            effective = 1 if (is_wheel and v == 14) else v
            if effective == t:
                result.append(card)
                remaining.pop(i)
                break

    # Fallback: append anything left (shouldn't happen for valid hands)
    result.extend(remaining)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def best_hand_ref(seven: list[str]) -> tuple[HandRank, list[str]]:
    """
    Reference best_hand: pick the highest-scoring 5-card combo from 7 cards.
    Returns (HandRank, 5 card strings in compare order).
    """
    best_score = -1
    best_combo: tuple[str, ...] | None = None

    for combo in combinations(seven, 5):
        s = _encode(combo)
        if s > best_score:
            best_score = s
            best_combo = combo

    assert best_combo is not None
    cat, tb = _decode_tiebreak(best_score)
    ordered = _ordered_cards(best_combo, cat, tb)
    return _CAT_TO_HANDRANK[cat], ordered


def compare_hands_ref(a: list[str], b: list[str]) -> int:
    """
    Reference compare_hands: return 1 if A wins, -1 if B wins, 0 on tie.
    Compares the best-hand scores directly as integers.
    """
    best_a = max(_encode(combo) for combo in combinations(a, 5))
    best_b = max(_encode(combo) for combo in combinations(b, 5))

    if best_a > best_b:
        return 1
    if best_a < best_b:
        return -1
    return 0
