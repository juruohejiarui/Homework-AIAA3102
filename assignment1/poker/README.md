# Poker Best-Hand Analyzer — Debug & Build

In poker, your *hand* is the best five cards you can make. This project works
with **seven** cards: the goal is to determine the best five-card poker hand they
contain.

Poker hands are ranked, from weakest to strongest: high card, one pair, two
pair, three of a kind, straight, flush, full house, four of a kind, straight
flush, and — best of all — a royal flush (the `HandRank` enum in
`hand_rank.py`). When two hands are the same category, the higher cards win, and
the leftover cards ("kickers") break ties.

The work is split across five small modules, one per level. **Levels 1 and 2 are
already written for you — and they contain bugs you must find.** Levels 3-5 are
stubbed for you to build.

**Budget: ~2-3 hours.**

## Examples
Cards are 2-character strings — a rank (`2`-`9`, `T`, `J`, `Q`, `K`, `A`) followed
by a suit (`S`, `H`, `D`, `C`). `classify` works on `Card` objects (use
`parse` from `poker_common`); `best_hand` takes the raw strings and parses for
you. `HandRank` names come from `hand_rank.py`.

    classify(parse(["9H","8D","7S","6C","5H"]))  ->  STRAIGHT
    classify(parse(["7H","7D","KS","4C","2H"]))  ->  ONE_PAIR

    best_hand(["KH","QH","9H","5H","2H","7D","3S"])
      ->  (FLUSH, ["KH","QH","9H","5H","2H"])

These are the easy cases. The hidden grading set is deliberately broader — the
rare and edge-case hands are where the bugs hide.

## What you submit

| File | What it is |
|---|---|
| `level1.py` | `classify` — **contains bugs; fix it in place** (don't rewrite the file). |
| `level2.py` | `order_five` — **contains bugs; fix it in place** (must use `classify`). |
| `level3.py` | `best_hand` — you implement it (must use `classify` + `order_five`). |
| `level4.py` | `compare_hands` — you implement it (must use `best_hand`). |
| `level5.py` | `skew_best_hand` — implement a DIFFERENT game's rules from the spec. |
| `test_poker.py` | the tests you write to find the bugs and check each level (graded). |
| `BUGS.md` | one entry per bug you fix in levels 1-2: where, symptom, root cause, fix, why it's correct. |
| `REFLECTIONS.md` | what was hard, how you found the bugs, what you'd do differently. |
| `chat.md` | a transcript of your conversation with the AI assistant. |

## The five levels

Do them in order. Levels 3 and 4 import and build on the earlier levels (the
imports are already written — keep them); Level 5 stands alone. You earn partial
credit per level.

### Level 1 — debug `classify` (`level1.py`)
`classify(five_cards)` returns the `HandRank` of a 5-card hand. The version you're
given is wrong on some hands. Find every bug and **fix it in place** — do not
rewrite the file from scratch. You are NOT told how many bugs there are, or which
hands they affect. Document each in `BUGS.md`.

### Level 2 — debug `order_five` (`level2.py`)
`order_five(five_cards)` returns a hand's five cards in canonical compare order:
most significant cards first, then kickers high→low. The version you're given
**has bugs — find them all and fix them in place**, and keep using `classify`. As
in Level 1, you are not told how many.

### Level 3 — build `best_hand` (`level3.py`)
Given seven cards, `best_hand` returns the best five-card poker hand they contain:
its `HandRank` and those five cards as strings, in compare order (most significant
first). Build it on top of `classify` and `order_five`.

### Level 4 — build `compare_hands` (`level4.py`)
Given two seven-card hands, `compare_hands` returns `1` if the first wins, `-1` if
the second wins, `0` on a tie. Build it on top of `best_hand`.

### Level 5 — build `skew_best_hand` (`level5.py`) — a variant game
"Skew" is a variant of 5-card poker with exactly **two** rule changes from the
standard game:

1. **The ace is the lowest card** (it ranks below the 2).
2. **A straight beats a flush** (those two categories swap rank).

Everything else is standard poker. Implement
`skew_best_hand(seven) -> (HandRank, 5 cards)`: the best 5-card hand from 7,
with the 5 cards in compare order (most significant first).

Work out for yourself what these two changes imply for every kind of hand — the
spec states only the two rules, not their consequences. `level5.py` is
independent (it does not import the other levels).

## Which files you may change

| File | May edit? |
|---|---|
| `level1.py` – `level5.py` | ✅ this is your work |
| `test_poker.py` | ✅ write your tests |
| `BUGS.md`, `Design.md`, `REFLECTIONS.md`, `chat.md` | ✅ |
| `hand_rank.py`, `poker_common.py`, `poker_analysis.py` | ❌ **frozen — do not change.** The grader uses its own copies; edits are ignored and flagged. |

## Write tests
```bash
uv run python -m unittest test_poker      # or: uv run pytest
```
A bug you can't write a failing test for is a bug you haven't really found. Keep
tests for every level.

## Keep it clean
```bash
uv run ruff check level1.py level2.py level3.py level4.py level5.py
uv run mypy --strict level1.py level2.py level3.py level4.py level5.py
```

## Performance (optional)
The straightforward approach (checking all C(7,5) hands) is perfectly fine for
correctness, but slow if you evaluate a great many hands. **Once your code is
correct**, you may optionally make it faster — it's a small bonus and must never
come at the expense of correctness.

## How you are graded
Grading is automatic, against **hidden, comprehensive test sets** that are
deliberately broader than the examples above — passing the README examples is not
enough. You are scored on: correctness of levels 1-5 on the hidden sets; the tests
you write (do they catch the bugs and cover the levels); `BUGS.md` (real bugs,
correct root cause); and an **optional speed bonus** (only if your correctness is
already high).

The detailed point breakdown is released after the deadline. Rewriting a buggy
file instead of fixing it in place is detected and flagged.
