"""
External-Sampling Monte-Carlo CFR (MCCFR-ES) for the abstracted 21-card poker game.

For each iteration we:
  1. Sample a chance outcome (deal hands).
  2. For each "training player" p ∈ {0, 1}, recursively traverse the game tree:
     - At p's nodes: compute counterfactual utility of every action; update regret
       and average-strategy tables.
     - At opponent's nodes: sample a single action from their current strategy.
     - At chance/draw nodes inside the game: handled by the game module
       (replacement draws). We additionally sample-only here.

Reference: Lanctot et al. 2009, "Monte Carlo Sampling for Regret Minimization in
Extensive Games."
"""

from __future__ import annotations
from dataclasses import dataclass, field
import random
from typing import Dict, List, Tuple

from . import game as G


@dataclass
class Node:
    """Per-infoset regret + cumulative-strategy storage."""

    actions: Tuple[str, ...]
    regret_sum: List[float] = field(default_factory=list)
    strategy_sum: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.regret_sum:
            self.regret_sum = [0.0] * len(self.actions)
        if not self.strategy_sum:
            self.strategy_sum = [0.0] * len(self.actions)

    def current_strategy(self) -> List[float]:
        pos = [max(r, 0.0) for r in self.regret_sum]
        total = sum(pos)
        if total > 0:
            return [r / total for r in pos]
        n = len(self.actions)
        return [1.0 / n] * n

    def average_strategy(self) -> List[float]:
        total = sum(self.strategy_sum)
        if total > 0:
            return [s / total for s in self.strategy_sum]
        n = len(self.actions)
        return [1.0 / n] * n


class CFRTrainer:
    def __init__(self, cfg: G.GameConfig | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or G.GameConfig()
        self.rng = random.Random(seed)
        self.nodes: Dict[str, Node] = {}

    # ── Public API ────────────────────────────────────────────────

    def train(self, iterations: int, log_every: int = 0) -> None:
        for t in range(1, iterations + 1):
            for training_player in (0, 1):
                state = G.deal(self.cfg, self.rng)
                self._traverse(state, training_player)
            if log_every and t % log_every == 0:
                print(f"[iter {t:>9d}]  infosets={len(self.nodes):,}")

    def average_strategy(self, key: str) -> List[float] | None:
        node = self.nodes.get(key)
        return node.average_strategy() if node else None

    # ── Core recursion ────────────────────────────────────────────

    def _traverse(self, state: G.GameState, training_player: int) -> float:
        if G.is_terminal(state):
            return G.utility(state, training_player)

        actor = state.to_act
        actions = G.legal_actions(state)
        if not actions:
            # Defensive: no legal action (shouldn't happen if tree is well-formed).
            return G.utility(state, training_player)

        key = G.infoset_key(state)
        node = self.nodes.get(key)
        if node is None:
            node = Node(actions=tuple(actions))
            self.nodes[key] = node
        elif node.actions != tuple(actions):
            # Action set should be a function of the infoset key. If it diverges,
            # rebuild — this only happens when the abstraction is changed mid-run.
            node = Node(actions=tuple(actions))
            self.nodes[key] = node

        strategy = node.current_strategy()

        if actor == training_player:
            # External sampling: enumerate all actions for the training player.
            util_per_action: List[float] = [0.0] * len(actions)
            node_util = 0.0
            for i, a in enumerate(actions):
                next_state = G.apply(state, a, self.rng)
                util_per_action[i] = self._traverse(next_state, training_player)
                node_util += strategy[i] * util_per_action[i]
            # Regret update.
            for i in range(len(actions)):
                node.regret_sum[i] += util_per_action[i] - node_util
                node.strategy_sum[i] += strategy[i]
            return node_util

        # Opponent's turn — sample a single action.
        i = _sample(strategy, self.rng)
        next_state = G.apply(state, actions[i], self.rng)
        return self._traverse(next_state, training_player)


def _sample(distribution: List[float], rng: random.Random) -> int:
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(distribution):
        cum += p
        if r <= cum:
            return i
    return len(distribution) - 1
