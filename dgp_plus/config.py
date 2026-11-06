"""YAML configuration with simple inheritance.

A config lists parent files under `defaults:`. Parents merge first, then the file itself. Nested dicts merge key by
key; everything else is replaced. This keeps every dataset difference in a config file, never in a code branch.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict:
    """Load a config and its `defaults:` parents. Parent paths are relative to the file naming them."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged: dict = {}
    for parent in raw.pop("defaults", []) or []:
        merged = deep_merge(merged, load_config(path.parent / parent))
    return deep_merge(merged, raw)


def parse_overrides(items: list[str] | None) -> dict:
    """Turn ['dgp.M=4', 'training.max_epochs=2'] into a nested dict. Values are parsed as YAML scalars."""
    out: dict = {}
    for item in items or []:
        dotted, _, value = item.partition("=")
        node = out
        *parents, leaf = dotted.strip().split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = yaml.safe_load(value)
    return out


def config_hash(cfg: dict) -> str:
    """Short hash naming a run, so two configs never share a results folder."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
