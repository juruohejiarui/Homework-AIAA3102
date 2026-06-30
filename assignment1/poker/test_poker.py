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

    # TODO(student): the worked examples and the line above cover only easy,
    # common hands. Add tests for the situations they DON'T show — the unusual
    # hands, the edge cases, and the tie-breaks. For each, decide the correct
    # answer yourself, then write a test that pins it down. Cover every level.


if __name__ == "__main__":
    unittest.main()
