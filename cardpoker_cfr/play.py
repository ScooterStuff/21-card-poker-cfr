"""
CLI: play a hand of 21 Card Poker against a trained CFR strategy.

    python -m cardpoker_cfr.play --strategy strategies/baseline.json

The human plays as the Follower in odd rounds, Starter in even rounds (roles
alternate). Bet sizes are restricted to the abstracted set (min/pot/all-in).
"""

from __future__ import annotations
import argparse
import random
import sys
from typing import Dict, List

from . import game as G
from .abstraction import (
    ACTION_FOLD,
    ACTION_CHECK_CALL,
    ACTION_RAISE_MIN,
    ACTION_RAISE_POT,
    ACTION_RAISE_ALLIN,
    DISCARD_ACTIONS,
)
from .strategy import load

PRETTY = {
    ACTION_FOLD: "Fold",
    ACTION_CHECK_CALL: "Check/Call",
    ACTION_RAISE_MIN: "Raise (min)",
    ACTION_RAISE_POT: "Raise (pot)",
    ACTION_RAISE_ALLIN: "All-in",
    "D0": "Discard 0",
    "D1": "Discard 1",
    "D2": "Discard 2",
    "D3": "Discard 3",
    "D4": "Discard 4",
    "D5": "Discard 5",
}


def _sample(probs: List[float], rng: random.Random) -> int:
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1


def ai_choose(state: G.GameState, table: Dict[str, dict], rng: random.Random) -> str:
    legal = G.legal_actions(state)
    key = G.infoset_key(state)
    entry = table.get(key)
    if entry is None or not entry.get("actions"):
        # Unseen infoset → uniform over legal actions.
        return rng.choice(legal)

    # Filter to actions the AI was trained on AND that are legal here.
    pairs = [(a, p) for a, p in zip(entry["actions"], entry["probs"]) if a in legal]
    if not pairs:
        return rng.choice(legal)
    total = sum(p for _, p in pairs) or 1.0
    actions = [a for a, _ in pairs]
    probs = [p / total for _, p in pairs]
    return actions[_sample(probs, rng)]


def human_choose(state: G.GameState, human_player: int) -> str:
    legal = G.legal_actions(state)
    print()
    print(f"Your hand: {[str(c) for c in state.hands[human_player]]}")
    print(f"Pot: {state.pot}b   Your stack: {state.stacks[human_player]}b   "
          f"Opp stack: {state.stacks[1-human_player]}b   To call: {state.bet_to_match}b")
    print(f"Phase: {state.phase}")
    for i, a in enumerate(legal):
        print(f"  [{i}] {PRETTY.get(a, a)}")
    while True:
        choice = input("Choose action #: ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(legal):
            return legal[int(choice)]
        print("Invalid; try again.")


def play_round(table: Dict[str, dict], cfg: G.GameConfig, human_player: int,
               rng: random.Random) -> float:
    state = G.deal(cfg, rng)
    while not G.is_terminal(state):
        if state.to_act == human_player:
            action = human_choose(state, human_player)
        else:
            action = ai_choose(state, table, rng)
            print(f"Opponent: {PRETTY.get(action, action)}")
        state = G.apply(state, action, rng)

    payoff = G.utility(state, human_player)
    print()
    print(f"=== Result ===  Your hand: {[str(c) for c in state.hands[human_player]]}")
    print(f"             Opp hand:   {[str(c) for c in state.hands[1-human_player]]}")
    print(f"Net: {payoff:+.1f}b")
    return payoff


def main() -> None:
    p = argparse.ArgumentParser(description="Play vs a trained CFR strategy.")
    p.add_argument("--strategy", required=True, help="Path to a strategy JSON file")
    p.add_argument("--rounds", type=int, default=5, help="How many rounds to play")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg, infosets = load(args.strategy)
    rng = random.Random(args.seed or None)

    print(f"Loaded strategy: {len(infosets):,} infosets, stack={cfg.starting_stack}b")

    total = 0.0
    for r in range(args.rounds):
        human_player = r % 2  # alternate roles
        role = "Starter" if human_player == 0 else "Follower"
        print(f"\n──── Round {r+1} ────  You are: {role}")
        total += play_round(infosets, cfg, human_player, rng)
        print(f"Cumulative: {total:+.1f}b")

    print(f"\nFinal total: {total:+.1f}b over {args.rounds} rounds.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)
