"""Tests for hand evaluation and Joker resolution."""

from cardpoker_cfr.cards import Card, JOKER
from cardpoker_cfr.hand_eval import (
    evaluate,
    compare,
    PAIR,
    TWO_PAIR,
    THREE_OF_A_KIND,
    STRAIGHT,
    FULL_HOUSE,
    FOUR_OF_A_KIND,
    ROYAL_FLUSH,
    FIVE_OF_A_KIND,
)


def C(rank: str, suit: str = "S") -> Card:
    return Card(rank, suit)


def test_pair():
    score = evaluate([C("A"), C("A", "H"), C("K"), C("Q"), C("J")])
    assert score[0] == PAIR


def test_two_pair():
    score = evaluate([C("A"), C("A", "H"), C("K"), C("K", "H"), C("J")])
    assert score[0] == TWO_PAIR


def test_three_of_a_kind():
    score = evaluate([C("A"), C("A", "H"), C("A", "D"), C("K"), C("J")])
    assert score[0] == THREE_OF_A_KIND


def test_straight():
    # Mixed suits — A-K-Q-J-10 all same suit would be a Royal Flush.
    score = evaluate([C("A", "S"), C("K", "H"), C("Q", "D"), C("J", "C"), C("10", "S")])
    assert score[0] == STRAIGHT


def test_full_house():
    score = evaluate([C("A"), C("A", "H"), C("A", "D"), C("K"), C("K", "H")])
    assert score[0] == FULL_HOUSE


def test_four_of_a_kind():
    score = evaluate([C("A"), C("A", "H"), C("A", "D"), C("A", "C"), C("J")])
    assert score[0] == FOUR_OF_A_KIND


def test_royal_flush():
    score = evaluate([C("A", "H"), C("K", "H"), C("Q", "H"), C("J", "H"), C("10", "H")])
    assert score[0] == ROYAL_FLUSH


def test_joker_makes_five_of_a_kind():
    score = evaluate([C("A", "S"), C("A", "H"), C("A", "D"), C("A", "C"), JOKER])
    assert score[0] == FIVE_OF_A_KIND


def test_joker_makes_royal_flush():
    score = evaluate([C("K", "H"), C("Q", "H"), C("J", "H"), C("10", "H"), JOKER])
    assert score[0] == ROYAL_FLUSH


def test_higher_pair_wins():
    a = evaluate([C("A"), C("A", "H"), C("K"), C("Q"), C("J")])
    b = evaluate([C("K"), C("K", "H"), C("A"), C("Q"), C("J")])
    assert compare(a, b) > 0


def test_full_house_beats_straight():
    fh = evaluate([C("A"), C("A", "H"), C("A", "D"), C("K"), C("K", "H")])
    st = evaluate([C("A", "S"), C("K", "H"), C("Q", "D"), C("J", "C"), C("10", "S")])
    assert compare(fh, st) > 0


def test_tie_on_identical_pair():
    a = evaluate([C("A"), C("A", "H"), C("K"), C("Q"), C("J")])
    b = evaluate([C("A", "D"), C("A", "C"), C("K", "H"), C("Q", "H"), C("J", "H")])
    assert compare(a, b) == 0
