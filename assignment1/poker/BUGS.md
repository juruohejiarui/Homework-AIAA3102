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

## Bug 6
- **Where**: `level2.py`, `order_five()` flush/high-card ordering
- **Symptom**: flush and high-card hands were returned in ascending order (low to high), which breaks tie comparison order
- **Root cause**: the `sorted(..., key=value)` call for these ranks omitted `reverse=True`
- **Fix**: changed flush/high-card sorting to descending value order
- **Why the fix is correct**: poker compares same-rank hands from the highest relevant card downward

## Bug 7
- **Where**: `level2.py`, `order_five()` grouped-hand ordering for pairs/trips/quads/full house
- **Symptom**: grouped hands were ordered with low-frequency and low-value cards first, so kickers could appear before the primary made hand
- **Root cause**: sorting by `(count, value)` was ascending, putting less significant cards first
- **Fix**: changed sorting by `(count, value)` to descending with `reverse=True`
- **Why the fix is correct**: canonical compare order must put the most significant combination first, then kickers high to low

## Bug 8
- **Where**: `level2.py`, `order_five()` straight / straight-flush wheel case
- **Symptom**: `A-2-3-4-5` was ordered as ace-high when sorted by raw value, incorrectly treating it like an ace-high straight
- **Root cause**: ace is encoded as value 14 and needed a special low-ace adjustment for the wheel
- **Fix**: added an explicit wheel check and sort key that treats ace as 1 in this specific case
- **Why the fix is correct**: in standard poker, `A-2-3-4-5` is the lowest straight (5-high), so compare order must be `5,4,3,2,A`

## Bug 9
- **Where**: `level1.py`, `classify()` ace-low straight detection
- **Symptom**: hands like `6H 7D 8S 9C AH` were incorrectly classified as a straight, even though they are not consecutive poker ranks
- **Root cause**: the ace-low branch used only a span check (`distinct[3] - distinct[0] == 3`) and matched non-wheel hands that happened to include an ace
- **Fix**: tightened the special case to exactly the wheel rank set `{2, 3, 4, 5, 14}`
- **Why the fix is correct**: only `A-2-3-4-5` is a valid ace-low straight in standard poker; `A-6-7-8-9` is not

<!-- ## Bug 2 ... add as many as you find -->
