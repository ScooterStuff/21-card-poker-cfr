# 21 Card Poker — CFR AI

A self-trained AI for [21 Card Poker](https://github.com/ScooterStuff/21-card-poker) built with **Counterfactual Regret Minimization** (specifically, External-Sampling Monte Carlo CFR).

This is a sibling research repo for the original [21-card-poker](https://github.com/ScooterStuff/21-card-poker) game. The game rules, hand rankings, and betting flow are reproduced here; the focus is on producing a near-Nash equilibrium strategy.

## What this repo does

- Models 21 Card Poker as a finite extensive-form imperfect-information game.
- Applies **action**, **discard**, and **hand-strength** abstractions to make CFR tractable.
- Trains an average strategy via External-Sampling MCCFR.
- Saves a compact JSON strategy file you can load to play against (CLI), or reuse from any client.

## Game (recap)

- 20 cards (A K Q J 10, four suits) + 1 Joker, dealt 5 each.
- Forced bets: Starter 2b, Follower 1b. F acts first pre-draw.
- Single discard/draw phase (0–5 cards each). S acts first post-draw.
- Showdown: best 5-card hand wins, Joker resolves to its best substitution.
- Hand ranks (high→low): Five of a Kind, Royal Flush, Four of a Kind, Full House, Straight (only A-K-Q-J-10), Three of a Kind, Two Pair, Pair.

Full rules: see the [game repo](https://github.com/ScooterStuff/21-card-poker/blob/main/docs/21-card-poker-rules.md).

## Abstractions used

CFR scales with the size of the game tree, so we abstract:

- **Round-based stacks.** Each training round uses a fixed effective stack (default 10b). Bankroll dynamics across rounds are out of scope — the strategy is a single-round equilibrium.
- **Bet sizes.** Three discrete raise sizes: `min` (1b), `pot` (≈ pot-sized), `allin`. Plus Fold / Call / Check.
- **Discards.** The player chooses *how many* of the lowest off-group cards to discard (0–5). Joker is never discarded; pairs/trips/quads are always preserved.
- **Hand buckets.** Hands are mapped to a small key derived from hand category + key ranks + joker presence, so similar hands share a strategy.

These abstractions are documented and isolated in [`cardpoker_cfr/abstraction.py`](cardpoker_cfr/abstraction.py); swap them out to study the trade-offs.

## Quick start

```powershell
# Python 3.10+
pip install -r requirements.txt

# Train (default: 200_000 iterations, ~a few minutes)
python -m cardpoker_cfr.train --iters 200000 --out strategies/baseline.json

# Play vs the trained AI in a CLI
python -m cardpoker_cfr.play --strategy strategies/baseline.json
```

Run tests:

```powershell
pytest -q
```

## Project layout

```
cardpoker_cfr/
  cards.py         # Deck, Card, ranks
  hand_eval.py     # Hand evaluator + Joker resolution (ported from the game repo)
  abstraction.py   # Action / discard / hand-strength bucketing
  game.py          # Abstracted extensive-form game tree (round-based)
  cfr.py           # External-Sampling MCCFR
  strategy.py      # Strategy save/load (JSON)
  train.py         # CLI: train and save a strategy
  play.py          # CLI: play vs a trained strategy
tests/
  test_hand_eval.py
.github/workflows/
  test.yml         # CI: run tests on push/PR
```

## License

MIT — see [LICENSE](LICENSE).

## Credits

- Game design and original implementation: [ScooterStuff/21-card-poker](https://github.com/ScooterStuff/21-card-poker).
- CFR / MCCFR references: Zinkevich et al. 2007; Lanctot et al. 2009.
