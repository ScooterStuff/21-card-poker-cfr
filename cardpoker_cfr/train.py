"""
CLI: train a CFR strategy and save it as JSON.

    python -m cardpoker_cfr.train --iters 200000 --out strategies/baseline.json

By default writes both the full per-history strategy and a `*.compact.json`
companion suitable for game clients (web/desktop). Use `--no-full` to skip the
larger file.
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

from .cfr import CFRTrainer
from .game import GameConfig
from .strategy import save, save_compact


def main() -> None:
    p = argparse.ArgumentParser(description="Train a CFR strategy for 21 Card Poker.")
    p.add_argument("--iters", type=int, default=100_000, help="MCCFR iterations")
    p.add_argument("--stack", type=int, default=10, help="Effective stack per round (bb)")
    p.add_argument("--max-raises", type=int, default=3, help="Raise cap per street")
    p.add_argument("--seed", type=int, default=0, help="Random seed (0 = nondeterministic)")
    p.add_argument(
        "--log-every",
        type=int,
        default=10_000,
        help="Print progress every N iterations (0 disables)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="strategies/baseline.json",
        help="Output strategy file (full per-history)",
    )
    p.add_argument(
        "--no-full",
        action="store_true",
        help="Skip writing the full per-history strategy; only emit the compact file",
    )
    args = p.parse_args()

    cfg = GameConfig(starting_stack=args.stack, max_raises_per_street=args.max_raises)
    trainer = CFRTrainer(cfg=cfg, seed=(args.seed or None))

    print(f"Training MCCFR-ES: iters={args.iters:,}, stack={cfg.starting_stack}b, "
          f"max_raises={cfg.max_raises_per_street}")
    t0 = time.time()
    trainer.train(args.iters, log_every=args.log_every)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s. Discovered {len(trainer.nodes):,} infosets.")

    out = Path(args.out)
    if not args.no_full:
        save(trainer, out)
        print(f"Saved full   → {out}  ({out.stat().st_size/1024:.1f} KB)")

    compact = out.with_suffix(".compact.json")
    n = save_compact(trainer, compact)
    print(f"Saved compact → {compact}  ({compact.stat().st_size/1024:.1f} KB, {n:,} classes)")


if __name__ == "__main__":
    main()
