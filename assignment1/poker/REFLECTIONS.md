# Reflections

<!-- Be specific and honest. Generic filler scores low. -->

## What I built and how I worked with the AI

I implemented all five levels and built tests around the edge cases that are easy
to miss in poker logic (wheel straights, kicker ordering, full house tie-breaks,
and 7-card best-hand selection).

I used AI in two different ways:
1. Interactive debugging and implementation support for Levels 1-5.
2. Differential testing with a clean-room oracle: I asked separate AI instances
	to generate independent evaluators, then compared outputs on randomized hands.

Manual test cases were still important for targeted boundary checks, but they are
not enough for broad coverage. Differential testing helped scale from "a few known
cases" to "many random and adversarial cases".


## Where the AI was wrong or incomplete
<!-- Which edge cases did it miss? How did you find out (which test)? How did you fix it? -->

The AI initially missed or mis-specified some edge behavior:
1. A non-wheel hand containing an ace (for example A-6-7-8-9) was incorrectly
	accepted as a straight in an early version. I found this through a failing
	regression test and fixed it by requiring the exact wheel set for ace-low
	straight logic.
2. Some expected outputs in early Level 5 tests were wrong even though the code
	was correct. Differential checks against the clean-room oracle and focused unit
	tests exposed the mismatch, and I corrected the expected compare-order outputs.

The main lesson: AI suggestions are useful, but every rule must be pinned by
tests, especially around special cases and tie-break ordering.


## What was hardest

The hardest part was not writing the basic classification branches; it was making
all comparison layers consistent:
1. Category ranking.
2. In-category compare order.
3. 7-card selection of the best 5-card candidate.

Level 5 increased difficulty because ace-low changes many consequences and can
silently invert expected outcomes (for example in full house or two-pair ordering).


## What I'd do differently next time

1. Build the differential-testing harness earlier (oracle + judge) instead of
	adding it after most implementation was done.
2. Start with a formal checklist of edge classes (wheel, near-wheel, same-rank
	tie-breaks, duplicate ranks with different kickers) and require at least one
	direct test for each class before coding new features.
3. Keep optimization separate from correctness from day one: first make a simple,
	clearly correct version, then optimize in an isolated file (`optimize.py`).
