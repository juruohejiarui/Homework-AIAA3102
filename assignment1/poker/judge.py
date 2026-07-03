"""
judge.py — Differential tester for best_hand and compare_hands.

Usage:
    python judge.py [--nr_cases N] [--workers W] [--seed S] [--output FILE]

Compares the student's implementation (level3.best_hand, level4.compare_hands)
against the clean-room oracle (oracle.best_hand_ref, oracle.compare_hands_ref)
on randomly generated 7-card hands.

Card generation strategy
------------------------
Pure random draws from a 52-card deck produce rare hands (straight flush,
royal flush, quads, etc.) extremely infrequently. To ensure good coverage this
program uses a WEIGHTED mix of generators:

  - random_hand        : pure random 7 cards from a 52-card deck (baseline)
  - force_flush        : guarantee ≥5 cards of the same suit
  - force_straight     : guarantee 5 consecutive ranks (possibly mixed suits)
  - force_straight_flush: guarantee a straight flush (5 consecutive same-suit)
  - force_royal_flush  : guarantee a royal flush (T-J-Q-K-A same suit)
  - force_quads        : guarantee four of a kind
  - force_full_house   : guarantee a full house
  - force_wheel        : guarantee A-2-3-4-5 (wheel straight)
  - force_two_pair     : guarantee two pairs

Each generator produces a list of 7 card strings.  The mix weights are tuned
so that rare hands appear in ~20-30% of test cases.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from multiprocessing import Pool, cpu_count
from typing import Callable

# ---------------------------------------------------------------------------
# Card universe
# ---------------------------------------------------------------------------
RANKS = list("23456789TJQKA")
SUITS = list("SHDC")
FULL_DECK = [r + s for r in RANKS for s in SUITS]  # 52 cards


# ---------------------------------------------------------------------------
# Hand generators
# ---------------------------------------------------------------------------

def _sample_fill(deck: list[str], chosen: list[str], total: int = 7) -> list[str]:
    """Fill `chosen` up to `total` cards by sampling without replacement from `deck`."""
    remaining = [c for c in deck if c not in chosen]
    need = total - len(chosen)
    extra = random.sample(remaining, need)
    result = chosen + extra
    random.shuffle(result)
    return result


def random_hand(rng: random.Random) -> list[str]:
    return rng.sample(FULL_DECK, 7)


def force_flush(rng: random.Random) -> list[str]:
    suit = rng.choice(SUITS)
    suited = [r + suit for r in RANKS]
    flush_cards = rng.sample(suited, 5)
    return _sample_fill(FULL_DECK, flush_cards)


def force_straight(rng: random.Random) -> list[str]:
    # Pick a starting rank index (0..8 for 2..T, plus wheel A-2-3-4-5)
    wheel = rng.random() < 0.15
    if wheel:
        ranks = ["A", "2", "3", "4", "5"]
    else:
        start = rng.randint(0, 8)  # 2..T as low card
        ranks = RANKS[start: start + 5]
    # Use different suits to avoid accidental flush
    suits_chosen = rng.sample(SUITS * 4, 5)
    # Ensure not all same suit
    while len(set(suits_chosen)) == 1:
        suits_chosen = rng.sample(SUITS * 4, 5)
    straight_cards = [r + s for r, s in zip(ranks, suits_chosen)]
    return _sample_fill(FULL_DECK, straight_cards)


def force_straight_flush(rng: random.Random) -> list[str]:
    suit = rng.choice(SUITS)
    # Avoid royal flush (start index 0..7 → 2..9 as low card)
    start = rng.randint(0, 7)
    ranks = RANKS[start: start + 5]
    sf_cards = [r + suit for r in ranks]
    return _sample_fill(FULL_DECK, sf_cards)


def force_royal_flush(rng: random.Random) -> list[str]:
    suit = rng.choice(SUITS)
    rf_cards = [r + suit for r in ["T", "J", "Q", "K", "A"]]
    return _sample_fill(FULL_DECK, rf_cards)


def force_quads(rng: random.Random) -> list[str]:
    rank = rng.choice(RANKS)
    quads = [rank + s for s in SUITS]
    return _sample_fill(FULL_DECK, quads)


def force_full_house(rng: random.Random) -> list[str]:
    r3, r2 = rng.sample(RANKS, 2)
    trips_suits = rng.sample(SUITS, 3)
    pair_suits = rng.sample(SUITS, 2)
    fh_cards = [r3 + s for s in trips_suits] + [r2 + s for s in pair_suits]
    return _sample_fill(FULL_DECK, fh_cards)


def force_wheel(rng: random.Random) -> list[str]:
    ranks = ["A", "2", "3", "4", "5"]
    suits_chosen = rng.sample(SUITS * 4, 5)
    while len(set(suits_chosen)) == 1:
        suits_chosen = rng.sample(SUITS * 4, 5)
    wheel_cards = [r + s for r, s in zip(ranks, suits_chosen)]
    return _sample_fill(FULL_DECK, wheel_cards)


def force_two_pair(rng: random.Random) -> list[str]:
    r1, r2 = rng.sample(RANKS, 2)
    s1 = rng.sample(SUITS, 2)
    s2 = rng.sample(SUITS, 2)
    tp_cards = [r1 + s for s in s1] + [r2 + s for s in s2]
    return _sample_fill(FULL_DECK, tp_cards)


# Weighted generator table: (generator_fn, weight)
GENERATORS: list[tuple[Callable[[random.Random], list[str]], float]] = [
    (random_hand,          50.0),
    (force_flush,          10.0),
    (force_straight,       10.0),
    (force_straight_flush,  8.0),
    (force_royal_flush,     4.0),
    (force_quads,           6.0),
    (force_full_house,      6.0),
    (force_wheel,           4.0),
    (force_two_pair,        2.0),
]

_GEN_FNS, _GEN_WEIGHTS = zip(*GENERATORS)


def generate_hand(rng: random.Random) -> list[str]:
    """Pick a generator according to weights and produce a 7-card hand."""
    [fn] = rng.choices(_GEN_FNS, weights=_GEN_WEIGHTS, k=1)
    return fn(rng)


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess)
# ---------------------------------------------------------------------------

def _worker(args: tuple[int, int, str]) -> list[dict]:
    """
    Generate and test `count` random hands.
    Returns a list of dicts describing mismatches found.

    args = (count, seed, student_dir)
    """
    count, seed, student_dir = args

    # Import student modules from their directory
    if student_dir not in sys.path:
        sys.path.insert(0, student_dir)

    from level3 import best_hand as student_best_hand          # type: ignore
    from optimize import best_hand as student_best_hand_opt          # type: ignore
    from level4 import compare_hands as student_compare_hands  # type: ignore
    from oracle import best_hand_ref, compare_hands_ref        # type: ignore

    rng = random.Random(seed)
    mismatches: list[dict] = []

    for i in range(count):
        hand_a = generate_hand(rng)
        hand_b = generate_hand(rng)

        # --- Test best_hand ---
        try:
            ref_rank_a, ref_cards_a = best_hand_ref(hand_a)
            stu_rank_a, stu_cards_a = student_best_hand(hand_a)
            stu_rank_a_opt, stu_cards_a_opt = student_best_hand_opt(hand_a)
        except Exception as exc:
            mismatches.append({
                "type": "best_hand_exception",
                "hand": hand_a,
                "error": str(exc),
            })
            continue

        if ref_rank_a != stu_rank_a or ref_cards_a != stu_cards_a or stu_rank_a != stu_rank_a_opt or stu_cards_a != stu_cards_a_opt:
            mismatches.append({
                "type": "best_hand_mismatch",
                "hand": hand_a,
                "ref_rank": ref_rank_a.name,
                "ref_cards": ref_cards_a,
                "stu_rank": stu_rank_a.name,
                "stu_cards": stu_cards_a,
                "stu_rank_opt": stu_rank_a_opt.name,
                "stu_cards_opt": stu_cards_a_opt,
            })

        try:
            ref_rank_b, ref_cards_b = best_hand_ref(hand_b)
            stu_rank_b, stu_cards_b = student_best_hand(hand_b)
        except Exception as exc:
            mismatches.append({
                "type": "best_hand_exception",
                "hand": hand_b,
                "error": str(exc),
            })
            continue

        if ref_rank_b != stu_rank_b or ref_cards_b != stu_cards_b:
            mismatches.append({
                "type": "best_hand_mismatch",
                "hand": hand_b,
                "ref_rank": ref_rank_b.name,
                "ref_cards": ref_cards_b,
                "stu_rank": stu_rank_b.name,
                "stu_cards": stu_cards_b,
            })

        # --- Test compare_hands ---
        try:
            ref_cmp = compare_hands_ref(hand_a, hand_b)
            stu_cmp = student_compare_hands(hand_a, hand_b)
        except Exception as exc:
            mismatches.append({
                "type": "compare_hands_exception",
                "hand_a": hand_a,
                "hand_b": hand_b,
                "error": str(exc),
            })
            continue

        if ref_cmp != stu_cmp:
            mismatches.append({
                "type": "compare_hands_mismatch",
                "hand_a": hand_a,
                "hand_b": hand_b,
                "ref_result": ref_cmp,
                "stu_result": stu_cmp,
                "ref_best_a": {"rank": ref_rank_a.name, "cards": ref_cards_a},
                "ref_best_b": {"rank": ref_rank_b.name, "cards": ref_cards_b},
                "stu_best_a": {"rank": stu_rank_a.name, "cards": stu_cards_a},
                "stu_best_b": {"rank": stu_rank_b.name, "cards": stu_cards_b},
            })

    return mismatches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Differential tester: student vs oracle for best_hand / compare_hands"
    )
    parser.add_argument(
        "--nr_cases", type=int, default=1000_000,
        help="Total number of 7-card hand pairs to test (default: 10000)"
    )
    parser.add_argument(
        "--workers", type=int, default=min(cpu_count(), 8),
        help="Number of worker processes (default: min(cpu_count, 8))"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Master random seed (default: None)"
    )
    parser.add_argument(
        "--output", type=str, default="bugs_found.json",
        help="Output file for mismatches (JSON, default: bugs_found.json)"
    )
    parser.add_argument(
        "--student_dir", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "./"),
        help="Path to the directory containing the student's level*.py files"
    )
    args = parser.parse_args()

    student_dir = os.path.abspath(args.student_dir)
    if not os.path.isdir(student_dir):
        print(f"[ERROR] student_dir not found: {student_dir}", file=sys.stderr)
        sys.exit(1)

    nr_cases = args.nr_cases
    workers = max(1, args.workers)
    master_rng = random.Random(args.seed)

    # Distribute cases across workers
    base = nr_cases // workers
    remainder = nr_cases % workers
    counts = [base + (1 if i < remainder else 0) for i in range(workers)]
    seeds = [master_rng.randint(0, 2**31) for _ in range(workers)]
    work_items = [(counts[i], seeds[i], student_dir) for i in range(workers)]

    print(f"[judge] Testing {nr_cases} cases with {workers} workers (seed={args.seed})")
    print(f"[judge] Student dir: {student_dir}")
    t0 = time.perf_counter()

    all_mismatches: list[dict] = []
    with Pool(processes=workers) as pool:
        results = pool.map(_worker, work_items)

    for batch in results:
        all_mismatches.extend(batch)

    elapsed = time.perf_counter() - t0

    # Deduplicate by serializing the mismatch key
    seen: set[str] = set()
    unique_mismatches: list[dict] = []
    for m in all_mismatches:
        key = json.dumps(m, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_mismatches.append(m)

    # Summary statistics
    bh_mismatches = [m for m in unique_mismatches if "best_hand" in m["type"]]
    cmp_mismatches = [m for m in unique_mismatches if "compare_hands" in m["type"]]

    print(f"\n[judge] Finished in {elapsed:.2f}s")
    print(f"[judge] Total cases tested : {nr_cases}")
    print(f"[judge] best_hand  mismatches: {len(bh_mismatches)}")
    print(f"[judge] compare_hands mismatches: {len(cmp_mismatches)}")
    print(f"[judge] Total unique mismatches: {len(unique_mismatches)}")

    # Write output
    output_path = os.path.abspath(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "nr_cases": nr_cases,
                    "workers": workers,
                    "seed": args.seed,
                    "elapsed_seconds": round(elapsed, 3),
                    "best_hand_mismatches": len(bh_mismatches),
                    "compare_hands_mismatches": len(cmp_mismatches),
                    "total_unique_mismatches": len(unique_mismatches),
                },
                "mismatches": unique_mismatches,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[judge] Results written to: {output_path}")

    # Also write a human-readable CSV for quick inspection
    csv_path = output_path.replace(".json", ".csv")
    if unique_mismatches:
        fieldnames = sorted({k for m in unique_mismatches for k in m.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for m in unique_mismatches:
                row = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                       for k, v in m.items()}
                writer.writerow(row)
        print(f"[judge] CSV summary written to: {csv_path}")


if __name__ == "__main__":
    main()
