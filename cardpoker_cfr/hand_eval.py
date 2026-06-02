"""
Hand evaluation for 21 Card Poker.

Ported from the original game repo. Lower hand_rank index = stronger hand.
Tiebreakers are appended as rank values (higher = better) after the index.
"""

from __future__ import annotations
from typing import List, Tuple

from .cards import Card, RANKS, SUITS, RANK_VALUES

FIVE_OF_A_KIND = 1
ROYAL_FLUSH = 2
FOUR_OF_A_KIND = 3
FULL_HOUSE = 4
STRAIGHT = 5
THREE_OF_A_KIND = 6
TWO_PAIR = 7
PAIR = 8

HAND_NAMES = {
    FIVE_OF_A_KIND: "Five of a Kind",
    ROYAL_FLUSH: "Royal Flush",
    FOUR_OF_A_KIND: "Four of a Kind",
    FULL_HOUSE: "Full House",
    STRAIGHT: "Straight",
    THREE_OF_A_KIND: "Three of a Kind",
    TWO_PAIR: "Two Pair",
    PAIR: "Pair",
}

Score = Tuple[int, ...]


def _rv(rank: str) -> int:
    return RANK_VALUES.get(rank, 0)


def evaluate_no_joker(cards: List[Card]) -> Score:
    ranks = [c.rank for c in cards]
    suits = [c.suit for c in cards]

    counts: dict[str, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    sorted_counts = sorted(counts.items(), key=lambda x: (x[1], _rv(x[0])), reverse=True)

    if sorted_counts[0][1] == 5:
        return (FIVE_OF_A_KIND, _rv(sorted_counts[0][0]))

    if len(set(suits)) == 1 and set(ranks) == {"A", "K", "Q", "J", "10"}:
        return (ROYAL_FLUSH,)

    if sorted_counts[0][1] == 4:
        return (FOUR_OF_A_KIND, _rv(sorted_counts[0][0]), _rv(sorted_counts[1][0]))

    if sorted_counts[0][1] == 3 and sorted_counts[1][1] == 2:
        return (FULL_HOUSE, _rv(sorted_counts[0][0]), _rv(sorted_counts[1][0]))

    if set(ranks) == {"A", "K", "Q", "J", "10"}:
        return (STRAIGHT,)

    if sorted_counts[0][1] == 3:
        kickers = sorted([_rv(r) for r, c in sorted_counts if c != 3], reverse=True)
        return (THREE_OF_A_KIND, _rv(sorted_counts[0][0]), *kickers)

    pairs = [(r, c) for r, c in sorted_counts if c == 2]
    if len(pairs) == 2:
        pair_vals = sorted([_rv(p[0]) for p in pairs], reverse=True)
        kicker = [_rv(r) for r, c in sorted_counts if c == 1]
        return (TWO_PAIR, pair_vals[0], pair_vals[1], *kicker)

    if sorted_counts[0][1] == 2:
        kickers = sorted([_rv(r) for r, c in sorted_counts if c == 1], reverse=True)
        return (PAIR, _rv(sorted_counts[0][0]), *kickers)

    # Should not happen with 5 cards over 5 ranks (pigeonhole).
    vals = sorted([_rv(r) for r in ranks], reverse=True)
    return (9, *vals)


def evaluate(cards: List[Card]) -> Score:
    """Evaluate any 5-card hand, resolving Joker to its best substitution."""
    if not any(c.is_joker for c in cards):
        return evaluate_no_joker(cards)

    non_joker = [c for c in cards if not c.is_joker]
    best: Score | None = None
    for r in RANKS:
        for s in SUITS:
            score = evaluate_no_joker(non_joker + [Card(r, s)])
            if best is None or compare(score, best) > 0:
                best = score
    assert best is not None
    return best


def compare(a: Score, b: Score) -> int:
    """Return positive if a beats b, negative if b beats a, 0 on tie."""
    if a[0] != b[0]:
        return 1 if a[0] < b[0] else -1
    n = max(len(a), len(b))
    for i in range(1, n):
        va = a[i] if i < len(a) else 0
        vb = b[i] if i < len(b) else 0
        if va != vb:
            return 1 if va > vb else -1
    return 0


def hand_name(score: Score) -> str:
    return HAND_NAMES.get(score[0], "Unknown")
