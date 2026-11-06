"""A tiny synthetic review graph, processed on disk, with mock models. Everything runs on CPU in seconds."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dgp_plus.config import deep_merge, load_config  # noqa: E402
from dgp_plus.data import Graph, make_split  # noqa: E402
from dgp_plus.raw_data import relation_from_keys  # noqa: E402

BENIGN = ["the food was warm and the staff were friendly", "great location and a quiet comfortable room",
          "portions were generous and the service was quick"]
FRAUD = ["best place ever amazing amazing must visit now", "five stars best deal click my profile for coupons",
         "worst service ever do not go never again"]


def make_graph(nodes: int = 120, seed: int = 0) -> Graph:
    rng = np.random.default_rng(seed)
    users = rng.integers(0, 20, nodes)
    products = rng.integers(0, 6, nodes)
    labels = (users < 6).astype(np.int8)
    ratings = np.where(labels == 1, rng.choice([1.0, 5.0], nodes), rng.integers(2, 6, nodes).astype(float))
    texts = [str(rng.choice(FRAUD if y else BENIGN)) + f" review {i}" for i, y in enumerate(labels)]
    relations = {"RUR": relation_from_keys(users.tolist()),
                 "RSR": relation_from_keys(list(zip(products.tolist(), ratings.tolist())))}
    labels[-1] = -1
    return Graph("tiny", relations, texts, ratings.reshape(-1, 1).astype(np.float32), ["rating"], labels,
                 [f"n{i}" for i in range(nodes)])


@pytest.fixture
def tiny_config(tmp_path) -> dict:
    """Base config pointed at a tiny graph and mock models, in a temporary folder."""
    graph = make_graph()
    directory = tmp_path / "data" / "tiny"
    graph.save(directory)
    make_split(graph.labels, {"train": 40, "val": 30, "test": 40}, seed=0).save(directory / "split.json")
    base = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    return deep_merge(base, {
        "paths": {"data": str(tmp_path / "data"), "cache": str(tmp_path / "cache"), "results": str(tmp_path / "results")},
        "dataset": {"name": "tiny", "relations": ["RUR", "RSR"], "question": "Is this fraud?"},
        "embedder": {"backend": "hashing", "dim": 16}, "summarizer": {"backend": "mock"}, "classifier": {"backend": "mock"},
        "training": {"learning_rate": 0.05, "max_epochs": 4, "eval_batch_size": 16}})
