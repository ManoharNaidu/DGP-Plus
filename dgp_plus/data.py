"""Heterogeneous graphs and train/validation/test splits.

Relations stay separate, one sparse matrix each, and are never collapsed. A processed dataset is a folder with
relation_<R>.npz, numeric.npy, labels.npy, texts.jsonl, graph.json and split.json.

The split is fixed per dataset. The five experiment seeds vary only training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

UNLABELED = -1


@dataclass
class Graph:
    name: str
    relations: dict[str, sp.csr_matrix]      # relation name -> N x N symmetric matrix
    texts: list[str]
    numeric: np.ndarray                      # shape (N, d); d is 0 when a dataset has no numeric attributes
    numeric_names: list[str]
    labels: np.ndarray                       # 1 fraud, 0 benign, -1 unlabelled
    node_ids: list[str]

    def __post_init__(self):
        n = len(self.texts)
        for name, matrix in self.relations.items():
            if matrix.shape != (n, n):
                raise ValueError(f"relation {name} has shape {matrix.shape}, expected ({n}, {n})")
        if self.numeric.shape[0] != n or len(self.labels) != n:
            raise ValueError("texts, numeric and labels must all have N rows")

    @property
    def num_nodes(self) -> int:
        return len(self.texts)

    def summary(self) -> dict:
        edges = {name: int(matrix.nnz) for name, matrix in self.relations.items()}
        return {"name": self.name, "nodes": self.num_nodes, "edges": sum(edges.values()), "edge_types": len(edges),
                "edges_per_relation": edges, "frauds": int((self.labels == 1).sum()),
                "labelled": int((self.labels != UNLABELED).sum())}

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, matrix in self.relations.items():
            sp.save_npz(directory / f"relation_{name}.npz", matrix)
        np.save(directory / "numeric.npy", self.numeric)
        np.save(directory / "labels.npy", self.labels)
        with open(directory / "texts.jsonl", "w", encoding="utf-8") as f:
            for node_id, text in zip(self.node_ids, self.texts):
                f.write(json.dumps({"id": node_id, "text": text}, ensure_ascii=False) + "\n")
        meta = {"name": self.name, "relations": list(self.relations), "numeric_names": self.numeric_names,
                "summary": self.summary()}
        (directory / "graph.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_graph(directory: str | Path) -> Graph:
    directory = Path(directory)
    if not (directory / "graph.json").exists():
        raise FileNotFoundError(f"{directory} has no processed graph; see scripts/prepare_data.py")
    meta = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
    relations = {name: sp.load_npz(directory / f"relation_{name}.npz").tocsr() for name in meta["relations"]}
    rows = [json.loads(line) for line in open(directory / "texts.jsonl", encoding="utf-8")]
    return Graph(name=meta["name"], relations=relations, texts=[r["text"] for r in rows],
                 numeric=np.load(directory / "numeric.npy"), numeric_names=meta["numeric_names"],
                 labels=np.load(directory / "labels.npy"), node_ids=[r["id"] for r in rows])


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    seed: int

    def targets(self) -> np.ndarray:
        return np.concatenate([self.train, self.val, self.test])

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def save(self, path: str | Path) -> None:
        payload = {"seed": self.seed, "train": self.train.tolist(), "val": self.val.tolist(), "test": self.test.tolist()}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")


def load_split(directory: str | Path) -> Split:
    """Read split.json. A missing file is an error: silently drawing a new split would change every result."""
    path = Path(directory) / "split.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; create it once with scripts/prepare_data.py")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Split(np.array(payload["train"]), np.array(payload["val"]), np.array(payload["test"]), payload["seed"])


def _allocate(counts: dict[int, int], total: int) -> dict[int, int]:
    """Share `total` across classes in proportion, with largest-remainder rounding."""
    pool = sum(counts.values())
    exact = {c: total * n / pool for c, n in counts.items()}
    shares = {c: int(np.floor(x)) for c, x in exact.items()}
    for c in sorted(exact, key=lambda c: exact[c] - shares[c], reverse=True)[: total - sum(shares.values())]:
        shares[c] += 1
    return shares


def make_split(labels: np.ndarray, sizes: dict[str, int], seed: int = 0) -> Split:
    """Stratified split of the labelled nodes, drawn in the order test, validation, train."""
    labels = np.asarray(labels)
    labelled = np.flatnonzero(labels != UNLABELED)
    if sum(sizes[s] for s in ("train", "val", "test")) > len(labelled):
        raise ValueError(f"split needs more labelled nodes than the {len(labelled)} that exist")
    rng = np.random.default_rng(seed)
    remaining = {c: rng.permutation(labelled[labels[labelled] == c]) for c in (0, 1)}
    parts = {}
    for name in ("test", "val", "train"):
        shares = _allocate({c: len(v) for c, v in remaining.items()}, sizes[name])
        chosen = []
        for c, count in shares.items():
            chosen.append(remaining[c][:count])
            remaining[c] = remaining[c][count:]
        parts[name] = np.sort(np.concatenate(chosen))
    return Split(parts["train"], parts["val"], parts["test"], seed)
