"""
Abstracted extensive-form game for 21 Card Poker.

Scope: ONE round (forced bets → pre-draw bet → draw → post-draw bet → showdown).
Stack size is fixed per round. Bankroll dynamics across rounds are deliberately
not modelled — the trained strategy is a single-round equilibrium.

Player numbering inside CFR:
    0 = Starter (S)  — posts 2b, acts first POST-draw
    1 = Follower (F) — posts 1b, acts first PRE-draw

The game is represented as a `GameNode` recursive structure. Chance nodes
(deal, draw replacements) are sampled by the CFR caller; the game class
itself only enumerates *decision* nodes given fully-dealt hands.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple
import random

from .cards import Card, make_deck
from .hand_eval import compare, evaluate
from .abstraction import (
    ACTION_FOLD,
    ACTION_CHECK_CALL,
    ACTION_RAISE_MIN,
    ACTION_RAISE_POT,
    ACTION_RAISE_ALLIN,
    DISCARD_ACTIONS,
    discard_action_count,
    raise_amount,
    choose_discard_indices,
    hand_bucket,
)

# Phase tags (kept tiny — they enter the infoset key).
P_PRE = "pre"     # pre-draw betting
P_DRAW_F = "drF"  # follower picks discard count
P_DRAW_S = "drS"  # starter picks discard count
P_POST = "post"   # post-draw betting
P_TERM = "term"

STARTER = 0
FOLLOWER = 1


@dataclass
class GameConfig:
    starting_stack: int = 10
    starter_blind: int = 2
    follower_blind: int = 1
    max_raises_per_street: int = 3  # cap raise depth so the tree is finite


@dataclass
class GameState:
    """A node in the abstracted extensive-form game."""

    cfg: GameConfig
    hands: Tuple[List[Card], List[Card]]            # (starter_hand, follower_hand)
    deck: List[Card]                                # remaining draw pile
    stacks: List[int]                               # chips behind for each player
    contrib: List[int]                              # chips already in pot from each player
    phase: str = P_PRE
    to_act: int = FOLLOWER
    bet_to_match: int = 0                           # additional chips current player owes
    raises_this_street: int = 0
    folded: Optional[int] = None
    history: Tuple[str, ...] = ()                   # full action history (used in infoset key)

    # Cached after each draw decision so the second-mover can see the count.
    follower_discarded: int = -1
    starter_discarded: int = -1

    @property
    def pot(self) -> int:
        return self.contrib[0] + self.contrib[1]


# ── Construction ──────────────────────────────────────────────────

def deal(cfg: GameConfig, rng: random.Random) -> GameState:
    """Sample a fresh round: shuffle deck, deal 5 to each, post blinds."""
    deck = make_deck(rng)
    starter_hand = [deck.pop() for _ in range(5)]
    follower_hand = [deck.pop() for _ in range(5)]

    sb = min(cfg.starter_blind, cfg.starting_stack)
    fb = min(cfg.follower_blind, cfg.starting_stack)
    return GameState(
        cfg=cfg,
        hands=(starter_hand, follower_hand),
        deck=deck,
        stacks=[cfg.starting_stack - sb, cfg.starting_stack - fb],
        contrib=[sb, fb],
        phase=P_PRE,
        to_act=FOLLOWER,
        bet_to_match=sb - fb,  # follower owes the difference
        raises_this_street=0,
    )


# ── Legal actions ─────────────────────────────────────────────────

def legal_actions(s: GameState) -> List[str]:
    if s.phase in (P_PRE, P_POST):
        actions: List[str] = []
        opp = 1 - s.to_act
        # Always can fold (except when checking is free — but folding is dominated then,
        # so we just disallow it to shrink the tree).
        if s.bet_to_match > 0:
            actions.append(ACTION_FOLD)
        actions.append(ACTION_CHECK_CALL)

        room = s.stacks[s.to_act] - s.bet_to_match
        opp_room = s.stacks[opp]
        max_raise = max(0, min(room, opp_room))
        if (
            max_raise > 0
            and s.raises_this_street < s.cfg.max_raises_per_street
        ):
            # Min raise is always available.
            actions.append(ACTION_RAISE_MIN)
            # Pot raise: only if it's distinct from min (>1) and ≤ max_raise.
            pot_amt = max(1, s.pot)
            if pot_amt > 1 and pot_amt < max_raise:
                actions.append(ACTION_RAISE_POT)
            # All-in: only if distinct from min and pot.
            if max_raise > 1:
                actions.append(ACTION_RAISE_ALLIN)
        return actions

    if s.phase in (P_DRAW_F, P_DRAW_S):
        # Limit by deck supply (very rarely binding here, but be safe).
        max_d = min(5, len(s.deck))
        return [a for a in DISCARD_ACTIONS if discard_action_count(a) <= max_d]

    return []


# ── Apply action ──────────────────────────────────────────────────

def apply(s: GameState, action: str, rng: random.Random) -> GameState:
    """Return a new GameState resulting from `action` taken at `s`."""
    if s.phase in (P_PRE, P_POST):
        return _apply_betting(s, action)
    if s.phase in (P_DRAW_F, P_DRAW_S):
        return _apply_draw(s, action, rng)
    raise RuntimeError(f"cannot apply at terminal phase: {s.phase}")


def _next_phase_after_betting_close(s: GameState) -> GameState:
    """Called when a betting street closes (call or both-checked)."""
    if s.phase == P_PRE:
        return replace(
            s,
            phase=P_DRAW_F,
            to_act=FOLLOWER,
            bet_to_match=0,
            raises_this_street=0,
        )
    # Post-draw close → terminal showdown.
    return replace(s, phase=P_TERM, to_act=-1)


def _apply_betting(s: GameState, action: str) -> GameState:
    actor = s.to_act
    opp = 1 - actor
    hist = s.history + (action,)

    if action == ACTION_FOLD:
        return replace(s, phase=P_TERM, folded=actor, history=hist, to_act=-1)

    if action == ACTION_CHECK_CALL:
        # Pay the call (if any).
        pay = min(s.bet_to_match, s.stacks[actor])
        new_stacks = list(s.stacks)
        new_stacks[actor] -= pay
        new_contrib = list(s.contrib)
        new_contrib[actor] += pay

        # If there was a bet to match, this closes the street.
        # If both players checked (no bet, action passed once already), close too.
        # We detect "both checked" via raises_this_street == 0 and the prior
        # action also being CHECK_CALL with no bet.
        if s.bet_to_match > 0:
            ns = replace(
                s,
                stacks=new_stacks,
                contrib=new_contrib,
                bet_to_match=0,
                history=hist,
            )
            return _next_phase_after_betting_close(ns)

        # No bet outstanding: this is a check.
        # If the previous action was also a check on this street, close.
        prior_check = (
            len(s.history) > 0
            and s.history[-1] == ACTION_CHECK_CALL
            and s.raises_this_street == 0
        )
        if prior_check:
            ns = replace(s, history=hist)
            return _next_phase_after_betting_close(ns)
        # Otherwise pass action.
        return replace(s, to_act=opp, history=hist)

    # A raise action.
    room = s.stacks[actor] - s.bet_to_match
    opp_room = s.stacks[opp]
    max_raise = max(0, min(room, opp_room))
    raise_by = raise_amount(action, s.pot, max_raise)
    pay = s.bet_to_match + raise_by  # call + raise

    new_stacks = list(s.stacks)
    new_stacks[actor] -= pay
    new_contrib = list(s.contrib)
    new_contrib[actor] += pay

    # Opponent now owes `raise_by`.
    return replace(
        s,
        stacks=new_stacks,
        contrib=new_contrib,
        bet_to_match=raise_by,
        raises_this_street=s.raises_this_street + 1,
        to_act=opp,
        history=hist,
    )


def _apply_draw(s: GameState, action: str, rng: random.Random) -> GameState:
    n = discard_action_count(action)
    actor = s.to_act
    hand = list(s.hands[actor])
    deck = list(s.deck)

    # Translate the abstract count into concrete indices via the heuristic.
    idxs = choose_discard_indices(hand, n)
    # Discard, then draw replacements.
    for i in sorted(idxs, reverse=True):
        hand.pop(i)
    drawn = min(n, len(deck))
    for _ in range(drawn):
        hand.append(deck.pop())

    new_hands = list(s.hands)
    new_hands[actor] = hand
    hist = s.history + (action,)

    if s.phase == P_DRAW_F:
        return replace(
            s,
            hands=tuple(new_hands),  # type: ignore[arg-type]
            deck=deck,
            phase=P_DRAW_S,
            to_act=STARTER,
            follower_discarded=n,
            history=hist,
        )

    # P_DRAW_S → enter post-draw betting; starter acts first.
    return replace(
        s,
        hands=tuple(new_hands),  # type: ignore[arg-type]
        deck=deck,
        phase=P_POST,
        to_act=STARTER,
        bet_to_match=0,
        raises_this_street=0,
        starter_discarded=n,
        history=hist,
    )


# ── Terminals ─────────────────────────────────────────────────────

def is_terminal(s: GameState) -> bool:
    return s.phase == P_TERM


def utility(s: GameState, player: int) -> float:
    """Return the chip delta for `player` at a terminal node.

    Sign convention: net chips won (positive) or lost (negative) this round.
    Since contributions are committed before the terminal, payoff = pot won
    minus own contribution. On a tie, pot is split.
    """
    assert is_terminal(s), "utility called on non-terminal"

    pot = s.pot
    own = s.contrib[player]
    opp = 1 - player

    if s.folded is not None:
        # The folder loses their contribution; the opponent wins the pot minus their own contribution.
        if s.folded == player:
            return -float(own)
        return float(s.contrib[opp])

    # Showdown.
    cmp = compare(evaluate(s.hands[STARTER]), evaluate(s.hands[FOLLOWER]))
    if cmp == 0:
        # Tie: split. Net for either side = floor((pot - 2*own) / 2) — pot is symmetric here.
        return (pot / 2) - own
    winner = STARTER if cmp > 0 else FOLLOWER
    if winner == player:
        return float(s.contrib[opp])
    return -float(own)


# ── Information set key ───────────────────────────────────────────

def infoset_key(s: GameState) -> str:
    """Return the string key identifying the current player's information set.

    Includes: current player, betting/draw history, and the player's own
    *bucketed* hand (revealed/refreshed at pre-draw and post-draw entry).
    The opponent's hand is hidden, but the opponent's discard count is public.
    """
    actor = s.to_act
    bucket = hand_bucket(s.hands[actor])
    # Opponent's discard count (public information for the second mover and beyond).
    opp_disc = (
        s.follower_discarded if actor == STARTER else s.starter_discarded
    )
    return (
        f"P{actor}|ph={s.phase}|h={bucket[0]}.{bucket[1]}.{bucket[2]}.J{bucket[3]}"
        f"|opd={opp_disc}|bet={s.bet_to_match}|r={s.raises_this_street}"
        f"|hist={','.join(s.history)}"
    )
