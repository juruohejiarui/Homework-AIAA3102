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
from poker_common import parse


class TestPoker(unittest.TestCase):
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
        ]

        for cards, expected in cases:
            with self.subTest(cards=cards):
                self.assertEqual(classify(parse(cards)), expected)

    # TODO(student): the worked examples and the line above cover only easy,
    # common hands. Add tests for the situations they DON'T show — the unusual
    # hands, the edge cases, and the tie-breaks. For each, decide the correct
    # answer yourself, then write a test that pins it down. Cover every level.


if __name__ == "__main__":
    unittest.main()
