"""
Strategy file format and load/save helpers.

A strategy is just a JSON object:
    {
      "format": "cardpoker_cfr.v1",
      "config": { "starting_stack": 10, ... },
      "infosets": {
          "<infoset_key>": { "actions": ["F","C","Rmin"], "probs": [0.0,0.6,0.4] },
          ...
      }
    }
"""

from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from .cfr import CFRTrainer
from .game import GameConfig

FORMAT_TAG = "cardpoker_cfr.v1"


def save(trainer: CFRTrainer, path: str | Path) -> None:
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


def load(path: str | Path) -> Tuple[GameConfig, Dict[str, Dict[str, List]]]:
    data = json.loads(Path(path).read_text())
    if data.get("format") != FORMAT_TAG:
        raise ValueError(f"unsupported strategy format: {data.get('format')!r}")
    cfg = GameConfig(**data["config"])
    return cfg, data["infosets"]
