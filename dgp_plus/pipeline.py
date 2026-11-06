"""From a graph to the final prompts: embed, trim, summarise, build.

The stages run in order and each one caches its output. `until` stops early, which is how summary generation is
sharded: build the trims once, then run each shard with until="node_summaries" and "metapath_summaries", then build
prompts in one process.
"""

from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dgp_plus.cache import TextStore, artifact_dir
from dgp_plus.data import Graph, Split, load_graph, load_split
from dgp_plus.prompt import MetapathBlock, build_prompt, load_template, prompt_version, truncate_to_tokens
from dgp_plus.summarise import (LazySummarizer, Shard, cap_source_tokens, metapath_summaries, node_summaries,
                                    numeric_mean)
from dgp_plus.trimming import (enumerate_metapaths, load_trim, make_embedder, metapath_name, save_trim,
                                   trim_metapath)

STAGES = ("trim", "node_summaries", "metapath_summaries", "prompts")


@dataclass
class PromptSet:
    prompts: dict[int, str]
    graph: Graph
    split: Split
    mean_tokens: float
    max_tokens: int
    truncated_targets: int


class WordTokenizer:
    """Counts words instead of model tokens, for tests with mock models."""

    def __call__(self, text: str, add_special_tokens: bool = False):
        return SimpleNamespace(input_ids=text.split())

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str], skip_special_tokens: bool = True) -> str:
        return " ".join(tokens)


def load_tokenizer(cfg: dict):
    if cfg["classifier"].get("backend") == "mock":
        return WordTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg["classifier"]["model_id"], revision=cfg["classifier"].get("revision"))


def _template_hash(name: str) -> str:
    return hashlib.sha1(load_template(name).encode()).hexdigest()[:12]


def embedding_path(graph: Graph, cfg: dict) -> Path:
    settings = {k: cfg["embedder"].get(k) for k in ("model_id", "revision", "max_length", "backend")}
    meta = {"dataset": graph.name, "nodes": graph.num_nodes, **settings}
    return artifact_dir(cfg["paths"]["cache"], "embeddings", meta) / "text_embeddings.npy"


def _diffusion_features(graph: Graph, cfg: dict, device: str) -> np.ndarray:
    """DeBERTaV3 text embeddings joined to the numeric features, unscaled."""
    embedder_cfg = cfg["embedder"]
    path = embedding_path(graph, cfg)
    if path.exists():
        embeddings = np.load(path)
    else:
        embeddings = make_embedder(embedder_cfg, device).embed(graph.texts)
        np.save(path, embeddings)
    return np.concatenate([np.asarray(embeddings, dtype=np.float64), np.asarray(graph.numeric, dtype=np.float64)], axis=1)


def trims_for(graph: Graph, split: Split, cfg: dict, device: str, log) -> tuple[dict[str, dict], Path]:
    """Trimmed neighbours per metapath, and the cache folder that names them."""
    dgp = cfg["dgp"]
    targets = split.targets()
    meta = {"dataset": graph.name, "split_seed": split.seed, "K": dgp["K"], "M": dgp["M"],
            "targets": hashlib.sha1(np.sort(targets).astype(np.int64).tobytes()).hexdigest()[:16],
            "embedder": cfg["embedder"].get("model_id", "hashing")}
    directory = artifact_dir(cfg["paths"]["cache"], "trims", meta)
    trims, features = {}, None
    for path in enumerate_metapaths(cfg["dataset"]["relations"], dgp["K"]):
        name = metapath_name(path)
        file = directory / f"{name}.npz"
        if not file.exists():
            log(f"trimming {name}")
            if features is None:
                features = _diffusion_features(graph, cfg, device)
            save_trim(file, trim_metapath(graph.relations, path, features, targets, dgp["K"], dgp["M"]))
        trims[name] = load_trim(file)
    return trims, directory


def _text_store(graph: Graph, cfg: dict, stage: str, extra: dict) -> TextStore:
    writer = cfg["summarizer"]
    meta = {"dataset": graph.name, "model": writer.get("model_id", "mock"), "revision": writer.get("revision"),
            "prompt_version": prompt_version(), **extra}
    return TextStore(artifact_dir(cfg["paths"]["cache"], stage, meta) / "summaries.jsonl")


def node_store(graph: Graph, cfg: dict) -> TextStore:
    """Cache key holds the template's content and the model revision, so editing either can never reuse stale text."""
    extra = {"template": _template_hash("node_summary"), "budget": cfg["dgp"]["B_node"]}
    if cfg["summarizer"].get("max_source_tokens"):
        extra["max_source_tokens"] = cfg["summarizer"]["max_source_tokens"]
    return _text_store(graph, cfg, "node_summaries", extra)


def metapath_store(graph: Graph, cfg: dict, nodes: TextStore, trim_dir: Path) -> TextStore:
    return _text_store(graph, cfg, "metapath_summaries",
                       {"template": _template_hash("metapath_summary"), "budget": cfg["dgp"]["B_meta"],
                        "node_store": nodes.path.parent.name, "trims": trim_dir.name})


def build_prompts(cfg: dict, device: str = "cpu", until: str = "prompts", shard: Shard | None = None, log=print) -> PromptSet | None:
    """Run the stages up to `until`. Returns the prompts only at the last stage."""
    if until not in STAGES:
        raise ValueError(f"until must be one of {STAGES}")
    if shard and until not in ("node_summaries", "metapath_summaries"):
        raise ValueError("a shard may only fill summary stores")
    data_dir = Path(cfg["paths"]["data"]) / cfg["dataset"]["name"]
    graph, split = load_graph(data_dir), load_split(data_dir)
    dgp = cfg["dgp"]
    trims, trim_dir = trims_for(graph, split, cfg, device, log)
    if until == "trim":
        return None

    tokenizer = load_tokenizer(cfg)
    writer = LazySummarizer(cfg["summarizer"], device)
    targets = [int(v) for v in split.targets()]
    used = sorted({int(u) for trim in trims.values() for found in trim.values() for u in found})
    texts = graph.texts
    cap = cfg["summarizer"].get("max_source_tokens")
    if cap:
        texts, cut = cap_source_tokens(graph.texts, used, tokenizer, cap)
        log(f"source cap: {cut} of {len(used)} nodes cut to their first {cap} tokens")
    log(f"node summaries: {len(used)} nodes needed")
    nodes = node_store(graph, cfg)
    node_text = node_summaries(texts, used, writer, load_template("node_summary"), dgp["B_node"], nodes, shard)
    if until == "node_summaries":
        return None

    requests = [(v, name, [int(u) for u in trim[v]]) for name, trim in trims.items() for v in targets]
    log(f"metapath summaries: {sum(1 for r in requests if r[2])} requests")
    paths = metapath_store(graph, cfg, nodes, trim_dir)
    path_text = metapath_summaries(requests, node_text, writer, load_template("metapath_summary"), dgp["B_meta"],
                                   paths, shard)
    if until == "metapath_summaries":
        return None
    del writer
    gc.collect()

    template = load_template("final_prompt")
    question = cfg["dataset"]["question"]
    prompts, truncated = {}, 0
    for v in targets:
        target_text = truncate_to_tokens(tokenizer, graph.texts[v], dgp["max_target_tokens"])
        truncated += target_text != graph.texts[v]
        blocks = [MetapathBlock(name, len(trim[v]), path_text.get((v, name)),
                                numeric_mean(graph.numeric, trim[v]) if dgp["numeric_summary"] else None)
                  for name, trim in trims.items()]
        prompts[v] = build_prompt(target_text, blocks, graph.numeric_names, question, template, dgp["numeric_precision"])
    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts.values()]
    return PromptSet(prompts, graph, split, float(np.mean(lengths)), int(np.max(lengths)), truncated)
