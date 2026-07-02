# Bugs found

<!-- One entry per bug. You are not told how many there are — find them all.
     For each: where it was, what was wrong, your fix, and WHY the fix is correct
     (what poker rule / case the original code violated). -->

## Bug 1
- **Where**: `level1.py`, `classify()` straight-flush / straight detection
- **Symptom**: a wheel straight like `AH 2H 3H 4H 5H` was not handled as an ace-low straight/straight flush case by the plain `max - min == 4` check
- **Root cause**: the ordinary straight test only works for consecutive ranks in numeric order and misses the special A-2-3-4-5 pattern
- **Fix**: added an explicit ace-low branch that classifies `A2345` as `STRAIGHT` or `STRAIGHT_FLUSH` depending on suit
- **Why the fix is correct**: in poker, A-2-3-4-5 is a valid 5-high straight, so it must be recognized even though ace has value 14 in the rank mapping

## Bug 2
- **Where**: `level1.py`, `classify()` four-of-a-kind check
- **Symptom**: a hand with ranks like `AAAAK` was not recognized as four of a kind because the count pattern is `[4, 1]`, not `[4]`
- **Root cause**: the original code compared against an incomplete multiplicity pattern and ignored the singleton kicker
- **Fix**: changed the check to `pattern == [4, 1]`
- **Why the fix is correct**: a 5-card four-of-a-kind always consists of four equal ranks plus one unrelated card

## Bug 3
- **Where**: `level1.py`, `classify()` full-house check
- **Symptom**: a hand like `AAAKK` was not classified as a full house because the sorted multiplicity is `[3, 2]`
- **Root cause**: the original comparison used the pair-first order and did not match the sorted `Counter` output
- **Fix**: changed the check to `pattern == [3, 2]`
- **Why the fix is correct**: a full house is exactly three cards of one rank and two cards of another rank

## Bug 4
- **Where**: `level1.py`, `classify()` two-pair check
- **Symptom**: a hand like `AAKKQ` was not recognized as two pair because the multiplicities are `[2, 2, 1]`
- **Root cause**: the original code omitted the kicker from the pattern match
- **Fix**: changed the check to `pattern == [2, 2, 1]`
- **Why the fix is correct**: two pair is two matching ranks, another two matching ranks, and one unrelated card

## Bug 5
- **Where**: `level1.py`, `classify()` royal-flush branch
- **Symptom**: a true royal flush like `TH JH QH KH AH` needed to be recognized separately from a generic straight flush
- **Root cause**: the original code treated every straight flush the same and did not distinguish the ace-high case
- **Fix**: added a special case that returns `HandRank.ROYAL_FLUSH` when the hand is both a straight and a flush and the highest card is an ace
- **Why the fix is correct**: in poker, a royal flush is the highest possible straight flush and must be classified distinctly from all other straight flushes

<!-- ## Bug 2 ... add as many as you find -->
