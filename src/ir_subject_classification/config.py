"""Configuration loading, inheritance, and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    parent = config.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        config = deep_merge(load_config(parent_path), config)
    validate_config(config)
    config["_config_path"] = str(path)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {"experiment", "data", "split", "features", "representations", "classifiers"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    val = float(config["split"].get("validation_size", 0.15))
    test = float(config["split"].get("test_size", 0.15))
    if not 0 < val < 1 or not 0 < test < 1 or val + test >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than one")
    target_columns = config["data"].get("target_columns", [])
    if not target_columns:
        raise ValueError("data.target_columns must be explicit and non-empty")


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(clean, stream, allow_unicode=True, sort_keys=False)

