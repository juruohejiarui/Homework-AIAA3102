# Design

<!-- Describe how you decided to structure the solution BEFORE/AS you coded:
     - How do you represent a card and a hand?
     - How do you enumerate the candidate 5-card hands?
     - How do you classify a 5-card hand into a HandRank?
     - How do you order the 5 cards for Level 3 comparison?
     - Which edge cases did you anticipate (wheel, kickers, flush vs straight flush)?
     Put any AI-assisted design notes here too. -->

## Main strategy: differential testing first

My core design choice was to treat this as a verification-heavy problem, not only
an implementation problem.

1. `oracle.py` contains a clean-room evaluator generated independently from the
     main implementation path (different prompts and development context), so it can
     act as an oracle rather than echoing the same bug.
2. `judge.py` generates randomized and targeted 7-card inputs, runs both
     implementations, and reports mismatches for investigation.
3. Manual unit tests in `test_poker.py` are used to lock known boundary behavior,
     while oracle-based differential checks provide scale.

This combination gives both precision (named edge cases) and breadth (many random
cases).

## Representation and evaluation

I kept card representation compatible with the provided plumbing (`Card`, rank,
suit), and evaluated 7-card hands by enumerating all 5-card combinations.

Each 5-card candidate is scored by:
1. Hand category strength.
2. Ordered card values in compare order (most significant first).

The best candidate is selected by tuple comparison on these score components.

## Edge-case checklist

I explicitly tracked these high-risk areas:
1. Wheel handling (`A-2-3-4-5`) and near-wheel false positives.
2. Kicker ordering for pair/two-pair/trips/quads.
3. Full-house compare order when multiple trip/pair choices exist in 7 cards.
4. Straight vs flush ordering differences in Skew.
5. Skew ace-low consequences (ace as the minimum rank in all comparisons).

## Optimize path

For optional acceleration (`optimize.py`), I separated performance work from the
baseline correctness implementation:

1. Use bit operations to accelerate straight detection (including straight flush
     checks by applying straight detection to suit-filtered rank masks).
2. Use nested `dict` + `list` structures to group by rank and suit efficiently.
3. Keep final compare-order construction with Python-friendly sequence operations
     (sorting and tuple/list building), balancing speed and correctness readability.

This keeps the optimized version fast in hot paths while preserving predictable
comparison semantics.
