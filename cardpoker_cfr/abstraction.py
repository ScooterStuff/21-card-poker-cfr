"""
Abstractions to keep the CFR game tree tractable.

Three knobs:
1. Hand bucketing — group similar 5-card hands into a small key.
2. Discard policy — given a hand and a `discard_count` (0..5), choose which
   indices to discard. We always keep the Joker and any made groups (pair / trips / quads).
3. Bet abstraction — fixed action set (Fold / Check-Call / Raise{Min,Pot,AllIn}).
"""

from __future__ import annotations
from typing import List, Tuple

from .cards import Card, RANK_VALUES
from .hand_eval import (
    evaluate,
    PAIR,
    TWO_PAIR,
    THREE_OF_A_KIND,
    STRAIGHT,
    FULL_HOUSE,
    FOUR_OF_A_KIND,
    ROYAL_FLUSH,
    FIVE_OF_A_KIND,
)

# ── Hand bucketing ────────────────────────────────────────────────

# A coarse bucket key: (category_index, primary_rank, secondary_rank, has_joker).
# This keeps the abstraction small (~ a few hundred buckets) while preserving
# strategic differences between strong and weak hands of the same category.

def hand_bucket(hand: List[Card]) -> Tuple[int, int, int, int]:
    """Return a compact tuple key representing the strategic class of `hand`.

    Components:
        cat   : hand category (1..8, lower = stronger)
        primary, secondary : key rank values (0 if not applicable)
        has_joker : 0/1
    """
    score = evaluate(hand)
    cat = score[0]
    primary = score[1] if len(score) > 1 else 0
    secondary = score[2] if len(score) > 2 else 0
    has_joker = 1 if any(c.is_joker for c in hand) else 0
    return (cat, primary, secondary, has_joker)


def bucket_str(b: Tuple[int, int, int, int]) -> str:
    return f"B{b[0]}.{b[1]}.{b[2]}.J{b[3]}"


# ── Discard heuristic ─────────────────────────────────────────────

def choose_discard_indices(hand: List[Card], count: int) -> List[int]:
    """Pick `count` indices to discard from `hand`.

    Strategy: never discard the Joker; preserve any rank-groups (pair, trips,
    quads); discard the lowest-ranked off-group cards first.

    Returns a sorted list of indices into `hand`.
    """
    if count <= 0:
        return []

    # Group indices by rank (skip the Joker entirely).
    groups: dict[str, List[int]] = {}
    joker_idxs: List[int] = []
    for i, c in enumerate(hand):
        if c.is_joker:
            joker_idxs.append(i)
        else:
            groups.setdefault(c.rank, []).append(i)

    # Off-group = ranks that appear exactly once. These are the first to discard.
    off_group: List[int] = []
    in_group: List[int] = []
    for rank, idxs in groups.items():
        if len(idxs) == 1:
            off_group.append(idxs[0])
        else:
            in_group.extend(idxs)

    # Sort off-group ascending by rank value (discard worst first).
    off_group.sort(key=lambda i: RANK_VALUES.get(hand[i].rank, 0))

    chosen: List[int] = off_group[:count]
    if len(chosen) >= count:
        return sorted(chosen)

    # Need more — start breaking up the smallest groups (lowest rank first).
    remaining = count - len(chosen)
    in_group.sort(key=lambda i: RANK_VALUES.get(hand[i].rank, 0))
    chosen.extend(in_group[:remaining])
    return sorted(chosen)


# ── Bet-size abstraction ──────────────────────────────────────────

# Action labels used at decision nodes. The exact set legal at a node depends
# on the betting context (pre-/post-draw, whether a bet is outstanding).
ACTION_FOLD = "F"
ACTION_CHECK_CALL = "C"  # Check if no bet outstanding, otherwise Call.
ACTION_RAISE_MIN = "Rmin"
ACTION_RAISE_POT = "Rpot"
ACTION_RAISE_ALLIN = "Rallin"

ALL_ACTIONS = (
    ACTION_FOLD,
    ACTION_CHECK_CALL,
    ACTION_RAISE_MIN,
    ACTION_RAISE_POT,
    ACTION_RAISE_ALLIN,
)


def raise_amount(action: str, pot: int, max_raise: int) -> int:
    """Translate a raise action label into a chip amount, clamped to legality."""
    if action == ACTION_RAISE_MIN:
        amt = 1
    elif action == ACTION_RAISE_POT:
        amt = max(1, pot)
    elif action == ACTION_RAISE_ALLIN:
        amt = max_raise
    else:
        raise ValueError(f"not a raise action: {action}")
    return max(1, min(amt, max_raise))


# Discard count is itself an action at a draw node.
DISCARD_ACTIONS = ("D0", "D1", "D2", "D3", "D4", "D5")


def discard_action_count(a: str) -> int:
    return int(a[1:])
