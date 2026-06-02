"""
Strategy file format and load/save helpers.

Two formats:

  * cardpoker_cfr.v1  — one entry per (full) infoset key, including action history.
                        Best for further training / analysis.

  * cardpoker_cfr.compact.v1 — entries are bucketed by (player, phase, hand-bucket,
                               opponent-discard, bet-to-match, raises-this-street),
                               averaging over action histories weighted by visits.
                               Much smaller and easier for game clients to consume.
"""

from __future__ import annotations
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from .cfr import CFRTrainer
from .game import GameConfig

FORMAT_TAG = "cardpoker_cfr.v1"
COMPACT_TAG = "cardpoker_cfr.compact.v1"

# Strip the "|hist=..." suffix from a full infoset key.
_HIST_RE = re.compile(r"\|hist=[^|]*$")


def _strip_history(key: str) -> str:
    return _HIST_RE.sub("", key)


def save(trainer: CFRTrainer, path: str | Path) -> None:
    """Save the full per-history strategy table (for analysis / re-training)."""
    out = {
        "format": FORMAT_TAG,
        "config": asdict(trainer.cfg),
        "infosets": {
            key: {
                "actions": list(node.actions),
                "probs": [round(p, 6) for p in node.average_strategy()],
            }
            for key, node in trainer.nodes.items()
        },
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))


def save_compact(trainer: CFRTrainer, path: str | Path) -> int:
    """Aggregate infosets by class key (history-stripped) and save a compact strategy.

    Within a class, action probabilities are averaged weighted by each node's
    cumulative strategy sum (i.e., how often that infoset was visited).

    Returns the number of class entries written.
    """
    by_class: Dict[str, Dict[str, float]] = {}
    weights: Dict[str, float] = {}

    for key, node in trainer.nodes.items():
        cls = _strip_history(key)
        avg = node.average_strategy()
        weight = sum(node.strategy_sum) or 1.0
        bucket = by_class.setdefault(cls, {})
        for a, p in zip(node.actions, avg):
            bucket[a] = bucket.get(a, 0.0) + p * weight
        weights[cls] = weights.get(cls, 0.0) + weight

    entries: Dict[str, Dict[str, List]] = {}
    for cls, bucket in by_class.items():
        w = weights[cls] or 1.0
        actions = list(bucket.keys())
        probs = [bucket[a] / w for a in actions]
        s = sum(probs) or 1.0
        probs = [p / s for p in probs]
        entries[cls] = {
            "actions": actions,
            "probs": [round(p, 6) for p in probs],
        }

    out = {
        "format": COMPACT_TAG,
        "config": asdict(trainer.cfg),
        "infosets": entries,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(",", ":")))
    return len(entries)


def load(path: str | Path) -> Tuple[GameConfig, Dict[str, Dict[str, List]]]:
    data = json.loads(Path(path).read_text())
    fmt = data.get("format")
    if fmt not in (FORMAT_TAG, COMPACT_TAG):
        raise ValueError(f"unsupported strategy format: {fmt!r}")
    cfg = GameConfig(**data["config"])
    return cfg, data["infosets"]
