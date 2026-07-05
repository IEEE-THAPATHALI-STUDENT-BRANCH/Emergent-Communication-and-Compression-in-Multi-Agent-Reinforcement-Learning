"""Configuration helpers for Experiment 5."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON-compatible YAML config.

    The checked-in ``.yaml`` files are deliberately JSON-compatible so the
    experiment has no extra PyYAML dependency beyond the existing ML stack.
    """

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config_snapshot(cfg: Dict[str, Any], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "config_snapshot.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return path


def condition_from_config(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    if name not in cfg["conditions"]:
        available = ", ".join(sorted(cfg["conditions"]))
        raise KeyError(f"Unknown condition '{name}'. Available: {available}")
    condition = dict(cfg["conditions"][name])
    if condition.get("kind") == "duplex_curriculum":
        condition = dict(condition["stages"][-1])
        condition["kind"] = "duplex"
    return condition

