"""
Card / deck primitives for 21 Card Poker.

Deck: 20 cards (A K Q J 10 in 4 suits) + 1 Joker = 21 cards.
"""

from __future__ import annotations
from dataclasses import dataclass
import random
from typing import List

RANKS = ("A", "K", "Q", "J", "10")
SUITS = ("S", "H", "D", "C")  # ASCII for portability
RANK_VALUES = {"A": 5, "K": 4, "Q": 3, "J": 2, "10": 1}


@dataclass(frozen=True, order=False)
class Card:
    rank: str
    suit: str  # "" for joker
    is_joker: bool = False

    def __str__(self) -> str:
        return "Jk" if self.is_joker else f"{self.rank}{self.suit}"

    @property
    def value(self) -> int:
        return 100 if self.is_joker else RANK_VALUES.get(self.rank, 0)


JOKER = Card(rank="Joker", suit="", is_joker=True)


def make_deck(rng: random.Random | None = None) -> List[Card]:
    """Return a freshly shuffled 21-card deck."""
    cards: List[Card] = [Card(r, s) for r in RANKS for s in SUITS]
    cards.append(JOKER)
    (rng or random).shuffle(cards)
    return cards
