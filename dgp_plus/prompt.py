"""The final dual-granularity prompt (paper Eq. 12).

The target keeps its raw text (fine grain). Each metapath adds one coarse line:

    - RTR-RSR: <summary> (neighbor mean: rating=1.67)
    - RSR: no neighbors
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEMPLATES = Path(__file__).parent / "templates"


def load_template(name: str) -> str:
    return (TEMPLATES / f"{name}.txt").read_text(encoding="utf-8")


def prompt_version() -> str:
    return (TEMPLATES / "VERSION").read_text(encoding="utf-8").strip()


@dataclass
class MetapathBlock:
    name: str
    num_neighbours: int
    text: str | None = None
    numeric: np.ndarray | None = None


def format_block(block: MetapathBlock, numeric_names: list[str], precision: int = 2) -> str:
    if block.num_neighbours == 0:
        return f"- {block.name}: no neighbors"
    line = f"- {block.name}: {block.text.strip()}"
    if block.numeric is not None:
        means = ", ".join(f"{n}={v:.{precision}f}" for n, v in zip(numeric_names, block.numeric))
        line += f" (neighbor mean: {means})"
    return line


def build_prompt(target_text: str, blocks: list[MetapathBlock], numeric_names: list[str], question: str,
                 template: str, precision: int = 2) -> str:
    lines = "\n".join(format_block(b, numeric_names, precision) for b in blocks)
    return (template.replace("{metapath_blocks}", lines).replace("{target_text}", target_text.strip())
            .replace("{question}", question))


def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    """First `max_tokens` tokens of the target's text. Long reviews would otherwise crowd out the metapath lines."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text if len(ids) <= max_tokens else tokenizer.decode(ids[:max_tokens])
