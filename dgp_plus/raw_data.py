"""Build processed graphs from raw files.

Two reviews are linked when they share a key, such as the same user. AmazonVideo and YelpChi match the node, edge and
fraud counts of the paper's Table 1 with the definitions below; scripts/prepare_data.py checks this.

AmazonVideo keeps helpful votes out of the numeric features because the label is computed from them.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Hashable, Sequence

import numpy as np
import scipy.sparse as sp

from dgp_plus.data import Graph

SECONDS_PER_DAY = 86400
SECONDS_PER_WEEK = 604800


def relation_from_keys(keys: Sequence[Hashable]) -> sp.csr_matrix:
    """Symmetric 0/1 adjacency linking every pair of nodes that share a key. A key of None links nothing."""
    groups: dict[Hashable, int] = {}
    rows, cols = [], []
    for node, key in enumerate(keys):
        if key is not None:
            rows.append(node)
            cols.append(groups.setdefault(key, len(groups)))
    incidence = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(keys), len(groups)))
    adjacency = (incidence @ incidence.T).tocsr()
    adjacency = (adjacency - sp.diags(adjacency.diagonal())).tocsr()
    adjacency.eliminate_zeros()
    adjacency.data = np.ones_like(adjacency.data)
    return adjacency


def amazon_label(helpful: list[int], min_votes: int = 1, threshold: float = 0.5) -> int:
    """Unhelpful reviews are fraud. Reviews with no votes count as benign."""
    helpful_votes, total_votes = helpful
    if total_votes < min_votes:
        return 0
    return 1 if helpful_votes / total_votes < threshold else 0


def build_amazonvideo(raw_path: str | Path, cfg: dict) -> Graph:
    """reviews_Amazon_Instant_Video_5.json.gz: review nodes, with RSR = same rating and same Saturday-aligned week."""
    with gzip.open(raw_path, "rt", encoding="utf-8") as f:
        reviews = [json.loads(line) for line in f if line.strip()]
    offset = cfg.get("week_offset_days", 5) * SECONDS_PER_DAY
    keys = {"RUR": [r["reviewerID"] for r in reviews], "RPR": [r["asin"] for r in reviews],
            "RSR": [(r["overall"], (int(r["unixReviewTime"]) + offset) // SECONDS_PER_WEEK) for r in reviews]}
    labels = np.array([amazon_label(r["helpful"], cfg["label"]["min_votes"], cfg["label"]["threshold"]) for r in reviews],
                      dtype=np.int8)
    return Graph("amazonvideo", {name: relation_from_keys(keys[name]) for name in cfg["relations"]},
                 [r.get("reviewText", "") for r in reviews],
                 np.array([[float(r["overall"])] for r in reviews], dtype=np.float32), ["rating"], labels,
                 [f"{r['reviewerID']}|{r['asin']}" for r in reviews])


def build_yelpchi(reviews_path: str | Path, cfg: dict) -> Graph:
    """reviews.jsonl with review_id, user_id, product_id, date, rating, text, label: 67,395 reviews with raw text.

    Do not substitute YelpChi.mat: it has 45,954 nodes and no text.
    """
    reviews = [json.loads(line) for line in open(reviews_path, encoding="utf-8") if line.strip()]
    keys = {"RUR": [r["user_id"] for r in reviews],
            "RSR": [(r["product_id"], float(r["rating"])) for r in reviews],
            "RTR": [(r["product_id"], r["date"][:7]) for r in reviews]}
    return Graph("yelpchi", {name: relation_from_keys(keys[name]) for name in cfg["relations"]},
                 [r["text"] for r in reviews], np.array([[float(r["rating"])] for r in reviews], dtype=np.float32),
                 ["rating"], np.array([int(r["label"]) for r in reviews], dtype=np.int8),
                 [r["review_id"] for r in reviews])
