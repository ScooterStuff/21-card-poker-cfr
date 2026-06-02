"""Smoke tests for the abstracted game tree and CFR solver."""

import random

from cardpoker_cfr import game as G
from cardpoker_cfr.cfr import CFRTrainer


def test_deal_round_reaches_terminal_via_random_play():
    rng = random.Random(42)
    cfg = G.GameConfig(starting_stack=8, max_raises_per_street=2)
    state = G.deal(cfg, rng)
    steps = 0
    while not G.is_terminal(state):
        actions = G.legal_actions(state)
        assert actions, f"no legal actions at phase {state.phase}"
        state = G.apply(state, rng.choice(actions), rng)
        steps += 1
        assert steps < 200, "round did not terminate"
    # Utilities for both players must sum to zero (zero-sum).
    u0 = G.utility(state, 0)
    u1 = G.utility(state, 1)
    assert abs(u0 + u1) < 1e-6


def test_cfr_runs_a_few_iterations():
    cfg = G.GameConfig(starting_stack=6, max_raises_per_street=1)
    trainer = CFRTrainer(cfg=cfg, seed=123)
    trainer.train(50, log_every=0)
    assert len(trainer.nodes) > 0
    # Pick any node and verify the average strategy sums to 1.
    node = next(iter(trainer.nodes.values()))
    s = node.average_strategy()
    assert abs(sum(s) - 1.0) < 1e-6
    assert all(p >= 0 for p in s)
