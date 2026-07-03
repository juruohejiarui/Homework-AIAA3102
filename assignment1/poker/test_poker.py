"""YOUR tests go here. No grader test file is provided — writing good tests is
part of the assignment and is graded (see README "Write your own tests").

Run them with:  uv run python -m unittest test_poker      (or: uv run pytest)

The example below shows the mechanics. The README's worked examples cover only a
few basic situations on purpose — the interesting cases (and the ones you are
graded on) are NOT shown. Think about what your examples DON'T cover and add
tests for those.
"""
import unittest

from hand_rank import HandRank
from level1 import classify
from level2 import order_five
from level3 import best_hand
from optimize import best_hand as best_hand_opt
from level4 import compare_hands
from level5 import skew_best_hand
from poker_common import parse


class TestPoker(unittest.TestCase):
    @staticmethod
    def _ordered_values(cards: list[str]) -> list[int]:
        ordered = order_five(tuple(parse(cards)))
        return [c.value for c in ordered]

    def test_classify_basic_example(self):
        # Mechanics demo: classify takes Card objects, so wrap strings in parse().
        # This passes on the handout (a common hand) — it just confirms your setup.
        self.assertEqual(classify(parse(["7H", "7D", "KS", "4C", "2H"])), HandRank.ONE_PAIR)

    def test_classify_key_hand_ranks(self):
        cases = [
            (["TH", "JH", "QH", "KH", "AH"], HandRank.ROYAL_FLUSH),
            (["9H", "TH", "JH", "QH", "KH"], HandRank.STRAIGHT_FLUSH),
            (["AH", "2H", "3H", "4H", "5H"], HandRank.STRAIGHT_FLUSH),
            (["AS", "2H", "3H", "4H", "5H"], HandRank.STRAIGHT),
            (["AS", "2H", "3D", "4C", "5S"], HandRank.STRAIGHT),
            (["AS", "AH", "AD", "AC", "2S"], HandRank.FOUR_OF_A_KIND),
            (["AS", "AH", "AD", "2C", "2S"], HandRank.FULL_HOUSE),
            (["2H", "5H", "7H", "9H", "KH"], HandRank.FLUSH),
            (["2S", "3H", "4D", "5C", "6S"], HandRank.STRAIGHT),
            (["7H", "7D", "KS", "KC", "2H"], HandRank.TWO_PAIR),
            (["7H", "7D", "7S", "KC", "2H"], HandRank.THREE_OF_A_KIND),
            (["AH", "KD", "7S", "QC", "2H"], HandRank.HIGH_CARD),
                (["6H", "7D", "8S", "9C", "AH"], HandRank.HIGH_CARD),
        ]

        for cards, expected in cases:
            with self.subTest(cards=cards):
                self.assertEqual(classify(parse(cards)), expected)

    def test_order_five_canonical_all_hand_ranks(self):
        cases = [
            (["TH", "JH", "KH", "QH", "AH"], [14, 13, 12, 11, 10]),  # royal flush
            (["9H", "TH", "JH", "QH", "KH"], [13, 12, 11, 10, 9]),   # straight flush
            (["AS", "AH", "AD", "AC", "2S"], [14, 14, 14, 14, 2]),  # four of a kind
            (["TS", "TH", "TD", "3C", "3D"], [10, 10, 10, 3, 3]),   # full house
            (["2H", "5H", "7H", "9H", "KH"], [13, 9, 7, 5, 2]),     # flush
            (["9S", "8H", "7D", "6C", "5S"], [9, 8, 7, 6, 5]),      # straight
            (["7H", "7D", "7S", "KC", "2H"], [7, 7, 7, 13, 2]),     # three of a kind
            (["KH", "KD", "5S", "AC", "5H"], [13, 13, 5, 5, 14]),   # two pair
            (["QH", "QD", "AS", "7C", "4D"], [12, 12, 14, 7, 4]),   # one pair
            (["AH", "KD", "7S", "QC", "2H"], [14, 13, 12, 7, 2]),   # high card
        ]

        for cards, expected_values in cases:
            with self.subTest(cards=cards):
                self.assertEqual(self._ordered_values(cards), expected_values)

    def test_order_five_wheel_is_five_high(self):
        self.assertEqual(self._ordered_values(["AS", "2H", "3D", "4C", "5S"]), [5, 4, 3, 2, 14])
        self.assertEqual(self._ordered_values(["AH", "2H", "3H", "4H", "5H"]), [5, 4, 3, 2, 14])

    def test_best_hand_key_scenarios(self):
        cases = [
            (
                ["TH", "JH", "QH", "KH", "AH", "2C", "3D"],
                (HandRank.ROYAL_FLUSH, ["AH", "KH", "QH", "JH", "TH"]),
            ),
            (
                ["AH", "2H", "3H", "4H", "5H", "KD", "QS"],
                (HandRank.STRAIGHT_FLUSH, ["5H", "4H", "3H", "2H", "AH"]),
            ),
             (
                ["AH", "2H", "3H", "4H", "5H", "AD", "AS"],
                (HandRank.STRAIGHT_FLUSH, ["5H", "4H", "3H", "2H", "AH"]),
            ),
            (
                ["9H", "8H", "7H", "6H", "5H", "4H", "AH"],
                (HandRank.STRAIGHT_FLUSH, ["9H", "8H", "7H", "6H", "5H"]),
            ),
            (
                ["AS", "AH", "AD", "AC", "KH", "QD", "2S"],
                (HandRank.FOUR_OF_A_KIND, ["AS", "AH", "AD", "AC", "KH"]),
            ),
            (
                ["AS", "AH", "AD", "KS", "KH", "KD", "2C"],
                (HandRank.FULL_HOUSE, ["AS", "AH", "AD", "KS", "KH"]),
            ),
            (
                ["AH", "KH", "QH", "JH", "9H", "8H", "2C"],
                (HandRank.FLUSH, ["AH", "KH", "QH", "JH", "9H"]),
            ),
            (
                ["9S", "8H", "7D", "6C", "5S", "4H", "3D"],
                (HandRank.STRAIGHT, ["9S", "8H", "7D", "6C", "5S"]),
            ),
            (
                ["AS", "2D", "3C", "4H", "5S", "6D", "7C"],
                (HandRank.STRAIGHT, ["7C", "6D", "5S", "4H", "3C"]),
            ),
            (
                ["AS", "2D", "3C", "4H", "5S", "AD", "AC"],
                (HandRank.STRAIGHT, ["5S", "4H", "3C", "2D", "AS"]),
            ),
            (
                ["7H", "7D", "7S", "KC", "QD", "9C", "2H"],
                (HandRank.THREE_OF_A_KIND, ["7H", "7D", "7S", "KC", "QD"]),
            ),
            (
                ["AH", "AD", "KH", "KD", "QH", "QD", "2S"],
                (HandRank.TWO_PAIR, ["AH", "AD", "KH", "KD", "QH"]),
            ),
            (
                ["AH", "AD", "KS", "QC", "9H", "4D", "2S"],
                (HandRank.ONE_PAIR, ["AH", "AD", "KS", "QC", "9H"]),
            ),
            (
                ["AH", "KD", "QC", "9S", "7D", "4H", "2C"],
                (HandRank.HIGH_CARD, ["AH", "KD", "QC", "9S", "7D"]),
            ),
        ]

        for cards, expected in cases:
            with self.subTest(cards=cards):
                self.assertEqual(best_hand(cards), expected)
                
        for cards, expected in cases:
            with self.subTest(cards=cards):
                self.assertEqual(best_hand_opt(cards), expected)

    def test_best_hand_prefers_stronger_combo_among_candidates(self):
        cases = [
            (
                ["AS", "AH", "AD", "KS", "KD", "KC", "QH"],
                (HandRank.FULL_HOUSE, ["AS", "AH", "AD", "KS", "KD"]),
            ),
            (
                ["2H", "4H", "6H", "8H", "KH", "3S", "5D"],
                (HandRank.FLUSH, ["KH", "8H", "6H", "4H", "2H"]),
            ),
        ]

        for cards, expected in cases:
            with self.subTest(cards=cards):
                self.assertEqual(best_hand(cards), expected)
                self.assertEqual(best_hand_opt(cards), expected)

    def test_compare_hands_comprehensive(self):
        cases = [
            # Different hand-rank categories.
            (
                ["AH", "KH", "QH", "JH", "TH", "2C", "3D"],
                ["9S", "8S", "7S", "6S", "5S", "2D", "3C"],
                1,
            ),  # royal flush > straight flush
            (
                ["9S", "8S", "7S", "6S", "5S", "2D", "3C"],
                ["AS", "AH", "AD", "AC", "KD", "2C", "3H"],
                1,
            ),  # straight flush > four of a kind
            (
                ["AS", "AH", "AD", "AC", "KD", "2C", "3H"],
                ["KS", "KH", "KD", "QC", "QH", "2S", "3D"],
                1,
            ),  # four of a kind > full house
            (
                ["KS", "KH", "KD", "QC", "QH", "2S", "3D"],
                ["AH", "QH", "9H", "7H", "3H", "2D", "4C"],
                1,
            ),  # full house > flush
            (
                ["AH", "QH", "9H", "7H", "3H", "2D", "4C"],
                ["9S", "8D", "7C", "6H", "5S", "2C", "KD"],
                1,
            ),  # flush > straight
            (
                ["9S", "8D", "7C", "6H", "5S", "2C", "KD"],
                ["7H", "7D", "7S", "KC", "QD", "2H", "3C"],
                1,
            ),  # straight > three of a kind
            (
                ["7H", "7D", "7S", "KC", "QD", "2H", "3C"],
                ["AH", "AD", "KH", "KD", "QS", "2C", "3D"],
                1,
            ),  # three of a kind > two pair
            (
                ["AH", "AD", "KH", "KD", "QS", "2C", "3D"],
                ["QH", "QD", "AS", "9C", "7D", "2H", "3S"],
                1,
            ),  # two pair > one pair
            (
                ["QH", "QD", "AS", "9C", "7D", "2H", "3S"],
                ["AH", "KD", "QC", "9S", "7H", "3D", "2C"],
                1,
            ),  # one pair > high card
            # Same-category tie-breaks.
            (
                ["AS", "2H", "3D", "4C", "5S", "9D", "KD"],
                ["2S", "3H", "4D", "5C", "6S", "9H", "QD"],
                -1,
            ),  # wheel straight < 6-high straight
            (
                ["9H", "8H", "7H", "6H", "5H", "2C", "3D"],
                ["8S", "7S", "6S", "5S", "4S", "AC", "KD"],
                1,
            ),  # 9-high straight flush > 8-high straight flush
            (
                ["AS", "AH", "AD", "AC", "KS", "2D", "3H"],
                ["AS", "AH", "AD", "AC", "QS", "2C", "3D"],
                1,
            ),  # quads same rank, kicker decides
            (
                ["AS", "AH", "AD", "KS", "KH", "2C", "3D"],
                ["KS", "KH", "KD", "AS", "AH", "2D", "3C"],
                1,
            ),  # full house: trips rank decides
            (
                ["AS", "AH", "AD", "KS", "KH", "2C", "3D"],
                ["AS", "AH", "AD", "QS", "QH", "2D", "3C"],
                1,
            ),  # full house same trips, pair decides
            (
                ["AH", "KH", "QH", "9H", "6H", "2C", "3D"],
                ["AS", "QS", "JS", "9S", "6S", "2D", "3C"],
                1,
            ),  # flush: second card decides
            (
                ["7H", "7D", "7S", "AC", "KD", "2H", "3C"],
                ["7C", "7S", "7D", "AC", "QD", "2D", "3H"],
                1,
            ),  # trips: second kicker decides
            (
                ["AH", "AD", "KH", "KD", "QS", "2C", "3D"],
                ["AH", "AD", "QH", "QD", "KS", "2D", "3C"],
                1,
            ),  # two pair: top pair decides
            (
                ["AH", "AD", "KH", "KD", "QS", "2C", "3D"],
                ["AS", "AC", "KS", "KC", "JS", "2D", "3C"],
                1,
            ),  # two pair same pairs, kicker decides
            (
                ["AH", "AD", "KS", "QC", "9H", "2D", "3C"],
                ["AS", "AC", "KS", "QC", "8H", "2C", "3D"],
                1,
            ),  # one pair: third kicker decides
            (
                ["AH", "KD", "QC", "9S", "7H", "2D", "3C"],
                ["AS", "KD", "QC", "9H", "6D", "2C", "3D"],
                1,
            ),  # high card: fifth card decides
            (
                ["2H", "4H", "6H", "8H", "KH", "3S", "5D"],
                ["2S", "3D", "4C", "5H", "6S", "9C", "QD"],
                1,
            ),  # best hand from A is a flush, which beats B's straight
            # Ties.
            (
                ["AH", "KH", "QH", "JH", "TH", "2C", "3D"],
                ["AS", "KS", "QS", "JS", "TS", "4C", "5D"],
                0,
            ),  # both royal flush
            (
                ["AS", "2H", "3D", "4C", "5S", "9D", "KD"],
                ["AH", "2D", "3C", "4H", "5D", "QC", "JS"],
                0,
            ),  # both wheel straight
        ]

        self.assertGreaterEqual(len(cases), 20)
        for a, b, expected in cases:
            with self.subTest(a=a, b=b, expected=expected):
                self.assertEqual(compare_hands(a, b), expected)

    def test_skew_best_hand_core_rules(self):
        cases = [
            (
                ["AH", "2D", "3C", "4S", "5H", "9D", "KD"],
                (HandRank.STRAIGHT, ["5H", "4S", "3C", "2D", "AH"]),
            ),  # ace-low straight exists
            (
                ["TH", "JH", "QH", "KH", "AH", "2C", "3D"],
                (HandRank.FLUSH, ["KH", "QH", "JH", "TH", "AH"]),
            ),  # TJQKA is flush, not straight flush, in ace-low rules
            (
                ["9H", "TH", "JH", "QH", "KH", "2C", "3D"],
                (HandRank.STRAIGHT_FLUSH, ["KH", "QH", "JH", "TH", "9H"]),
            ),  # straight flush labeling remains STRAIGHT_FLUSH
            (
                ["2H", "4H", "6H", "8H", "KH", "3D", "5S"],
                (HandRank.STRAIGHT, ["6H", "5S", "4H", "3D", "2H"]),
            ),  # straight beats flush in Skew
            (
                ["AS", "AH", "AD", "KS", "KH", "KD", "2C"],
                (HandRank.FULL_HOUSE, ["KS", "KH", "KD", "AS", "AH"]),
            ),  # trips rank uses ace-low values (KKKAA beats AAAKK)
            (
                ["AH", "AD", "2S", "2D", "KS", "QD", "3C"],
                (HandRank.TWO_PAIR, ["2S", "2D", "AH", "AD", "KS"]),
            ),  # 22AA beats AA22 in ace-low ordering
            (
                ["AH", "KD", "QC", "9H", "8D", "2C", "3S"],
                (HandRank.HIGH_CARD, ["KD", "QC", "9H", "8D", "3S"]),
            ),  # ace is lowest, often excluded from best high-card hand
        ]

        for cards, expected in cases:
            with self.subTest(cards=cards):
                self.assertEqual(skew_best_hand(cards), expected)

if __name__ == "__main__":
    unittest.main()
