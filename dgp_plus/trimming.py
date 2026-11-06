"""Metapath trimming with a Markov diffusion kernel (paper Eq. 2-9).

For a metapath P, the steps are:
    A_P = A_r1 ... A_rL          path counts (Eq. 3)
    T_P = D^-1 A_P               row-stochastic transition; zero-degree rows stay zero
    H   = (1/K) sum_{k=0..K} T_P^k X     diffused node features (Eq. 6-7)
    keep the M neighbours closest to the target in H (Eq. 8-9)

K sets both the longest metapath and the number of diffusion steps. Everything is applied as a chain of sparse
products, so A_P and the operator are never formed in memory.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp

Metapath = tuple[str, ...]
Relations = Mapping[str, sp.csr_matrix]


def enumerate_metapaths(relation_names: Sequence[str], max_hops: int) -> list[Metapath]:
    """Every relation sequence of length 1..max_hops, shortest first."""
    paths: list[Metapath] = []
    for length in range(1, max_hops + 1):
        paths.extend(itertools.product(relation_names, repeat=length))
    return paths


def metapath_name(path: Metapath) -> str:
    return "-".join(path)


def metapath_neighbours(relations: Relations, path: Metapath, targets: np.ndarray,
                        chunk_size: int = 2048) -> list[np.ndarray]:
    """Sorted neighbours of each target along a metapath (Eq. 4). A target is removed from its own list."""
    neighbours: list[np.ndarray] = []
    for start in range(0, len(targets), chunk_size):
        chunk = targets[start:start + chunk_size]
        rows = relations[path[0]][chunk]
        for relation in path[1:]:
            rows = rows @ relations[relation]
        rows = sp.csr_matrix(rows)
        for i, target in enumerate(chunk):
            low, high = rows.indptr[i], rows.indptr[i + 1]
            found = np.sort(rows.indices[low:high][rows.data[low:high] > 0])
            neighbours.append(found[found != target].astype(np.int64))
    return neighbours


def diffused_embeddings(relations: Relations, path: Metapath, features: np.ndarray, K: int) -> np.ndarray:
    """H = Z_P(K) X, literally as Eq. 6 writes it: (1/K) sum_{k=0..K} T^k X."""

    def apply_adjacency(x: np.ndarray) -> np.ndarray:
        for relation in reversed(path):
            x = relations[relation] @ x
        return x

    size = next(iter(relations.values())).shape[0]
    degree = apply_adjacency(np.ones((size, 1))).ravel()
    inverse_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    current = np.asarray(features, dtype=np.float64)
    total = current.copy()
    for _ in range(K):
        current = inverse_degree[:, None] * apply_adjacency(current)
        total += current
    return total / K


def top_m(candidates: np.ndarray, embeddings: np.ndarray, target: int, M: int) -> np.ndarray:
    """The M candidates closest to the target by L2 distance, closest first. Ties break by node id."""
    difference = embeddings[candidates] - embeddings[target]
    distances = np.sqrt(np.einsum("ij,ij->i", difference, difference, dtype=np.float64))
    return candidates[np.lexsort((candidates, distances))[:M]]


def trim_metapath(relations: Relations, path: Metapath, features: np.ndarray, targets: np.ndarray,
                  K: int, M: int) -> dict[int, np.ndarray]:
    """Trimmed neighbours of each target along one metapath."""
    targets = np.asarray(targets, dtype=np.int64)
    embeddings = diffused_embeddings(relations, path, features, K)
    neighbours = metapath_neighbours(relations, path, targets)
    return {int(v): top_m(found, embeddings, int(v), M) if len(found) else found
            for v, found in zip(targets, neighbours)}


def save_trim(path: str | Path, trim: dict[int, np.ndarray]) -> None:
    targets = np.array(sorted(trim), dtype=np.int64)
    lists = [trim[int(v)] for v in targets]
    np.savez(path, targets=targets, offsets=np.cumsum([0] + [len(x) for x in lists]),
             neighbours=np.concatenate(lists) if lists else np.zeros(0, dtype=np.int64))


def load_trim(path: str | Path) -> dict[int, np.ndarray]:
    with np.load(path) as stored:                   # read each array once: NpzFile re-reads on every access
        targets, offsets, flat = stored["targets"], stored["offsets"], stored["neighbours"]
    return {int(v): flat[offsets[i]:offsets[i + 1]] for i, v in enumerate(targets)}


class DebertaEmbedder:
    """Mean-pooled DeBERTaV3 embeddings for the diffusion features.

    Always float32: the checkpoint stores float16, which overflows in DeBERTa's attention and would silently corrupt
    neighbour selection.
    """

    def __init__(self, cfg: dict, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.device = torch, device
        self.batch_size, self.max_length = cfg["batch_size"], cfg["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg.get("revision"))
        self.model = AutoModel.from_pretrained(cfg["model_id"], revision=cfg.get("revision"),
                                               dtype=torch.float32).to(device).eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embeddings in input order. Texts are batched by length; padding is masked out of the mean."""
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = order[start:start + self.batch_size]
                encoded = self.tokenizer([texts[i] for i in batch], padding=True, truncation=True,
                                         max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                out[batch] = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).float().cpu().numpy()
        return out


class HashingEmbedder:
    """Bag-of-words embedder for tests, so they need no model download. Never used for results."""

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.dim = cfg.get("dim", 64)

    def embed(self, texts: list[str]) -> np.ndarray:
        import zlib

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                out[i, zlib.crc32(word.encode()) % self.dim] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def make_embedder(cfg: dict, device: str):
    return (HashingEmbedder if cfg.get("backend") == "hashing" else DebertaEmbedder)(cfg, device)
