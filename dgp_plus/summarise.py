"""Node and metapath summaries, numeric means, and the frozen summary writer.

    node level      s_u    = Summarize(raw text of u; B_node)                    (Eq. 5)
    metapath level  S_P(v) = Summarize(s_u of the trimmed neighbours; B_meta)    (Eq. 10)
    numeric         a_P(v) = mean of the trimmed neighbours' numeric features    (Eq. 11)

Only trimmed neighbours are summarised; targets keep their raw text. The writer sees text only, never labels.

Large jobs are split by hash into shards. Each shard process generates only its slice of keys and appends to the same
store, so stopped shards resume and nothing is written twice.
"""

from __future__ import annotations

import hashlib

import numpy as np

from dgp_plus.cache import TextStore

Shard = tuple[int, int]                        # (index, count)
NODE_FLUSH, METAPATH_FLUSH = 64, 32            # summaries generated between writes to the store


def in_shard(key: str, shard: Shard | None) -> bool:
    return shard is None or int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % shard[1] == shard[0]


def fill_template(template: str, text: str, budget: int) -> str:
    return template.replace("{budget}", str(budget)).replace("{text}", text)


class ChatSummarizer:
    """Frozen Gemma-2-9B with greedy decoding, so summaries are deterministic and cacheable."""

    def __init__(self, cfg: dict, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.cfg, self.device = torch, cfg, device
        self.name = cfg["model_id"]
        self.batch_size = cfg["batch_size"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, revision=cfg.get("revision"))
        self.tokenizer.padding_side = "left"   # decoder-only batched generation
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        attention = {"attn_implementation": cfg["attn_implementation"]} if cfg.get("attn_implementation") else {}
        self.model = AutoModelForCausalLM.from_pretrained(self.name, revision=cfg.get("revision"), dtype=dtype, **attention)
        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _render(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True)

    def _batches(self, rendered: list[str]) -> list[list[str]]:
        """Consecutive batches. With max_batch_tokens_sq set, a batch also closes before prompts x longest^2 exceeds it:
        eager attention (Gemma) needs memory in that proportion, and 16 long reviews once asked for 15 GiB at once."""
        cap = self.cfg.get("max_batch_tokens_sq")
        if not cap:
            return [rendered[i:i + self.batch_size] for i in range(0, len(rendered), self.batch_size)]
        lengths = [len(ids) for ids in self.tokenizer(rendered, add_special_tokens=False)["input_ids"]]
        batches, current, longest = [], [], 0
        for text, length in zip(rendered, lengths):
            if current and (len(current) >= self.batch_size or (len(current) + 1) * max(longest, length) ** 2 > cap):
                batches.append(current)
                current, longest = [], 0
            current.append(text)
            longest = max(longest, length)
        return batches + [current] if current else batches

    def summarize(self, prompts: list[str], budget: int) -> list[str]:
        rendered = [self._render(p) for p in prompts]
        limit = self.cfg["max_input_tokens"]
        max_new_tokens = self.cfg["max_new_tokens_factor"] * budget + self.cfg["max_new_tokens_extra"]
        special = None
        outputs = []
        for batch in self._batches(rendered):
            # Default special tokens: changing this would change the text of every cached summary.
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True)
            longest = int(encoded["attention_mask"].sum(dim=1).max())
            if longest > limit:                # never cut here: that would remove the assistant header
                raise ValueError(f"summarizer input of {longest} tokens exceeds max_input_tokens={limit}")
            encoded = encoded.to(self.device)
            with self.torch.no_grad():
                generated = self.model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                                pad_token_id=self.tokenizer.pad_token_id)
            new_tokens = generated[:, encoded["input_ids"].shape[1]:]
            if special is None:
                special = self.torch.tensor(self.tokenizer.all_special_ids, device=new_tokens.device)
            for row in new_tokens:
                row = row[~self.torch.isin(row, special)]
                outputs.append(self.tokenizer.decode(row, skip_special_tokens=True).strip())
        return outputs


class MockSummarizer:
    """Keeps the first B words of the text. For tests only."""

    name = "mock-first-words"
    tokenizer = None

    def __init__(self, cfg: dict | None = None, device: str = "cpu"):
        pass

    def summarize(self, prompts: list[str], budget: int) -> list[str]:
        return [" ".join(p.split("\n\n", 1)[-1].split()[:budget]) for p in prompts]


class LazySummarizer:
    """Loads the model on the first real request, so a run with complete caches never spends GPU memory on it."""

    def __init__(self, cfg: dict, device: str):
        self.cfg, self.device, self.model = cfg, device, None

    def summarize(self, prompts: list[str], budget: int) -> list[str]:
        if self.model is None:
            kind = MockSummarizer if self.cfg.get("backend") == "mock" else ChatSummarizer
            self.model = kind(self.cfg, self.device)
        return self.model.summarize(prompts, budget)


def cap_source_tokens(texts: list[str], nodes: list[int], tokenizer, limit: int) -> tuple[list[str], int]:
    """Copy of `texts` with listed nodes cut to their first `limit` tokens, and how many were cut.

    A recorded deviation, used only where a config sets summarizer.max_source_tokens: FLAG's Amazon graph joins all of
    a user's reviews into one text, up to 321,281 tokens. Only source text is cut, so the instruction stays whole.
    """
    capped, cut = list(texts), 0
    for node in sorted(set(nodes)):
        ids = tokenizer(texts[node], add_special_tokens=False).input_ids
        if len(ids) > limit:
            capped[node] = tokenizer.decode(ids[:limit], skip_special_tokens=True)
            cut += 1
    return capped, cut


def _fill(store: TextStore, items: dict[str, str], summarizer, template: str, budget: int, flush: int,
          shard: Shard | None) -> None:
    """Generate what is missing from the store, in this shard's slice, writing as results arrive."""
    todo = [(k, t) for k, t in items.items() if k not in store and in_shard(k, shard)]
    for start in range(0, len(todo), flush):
        batch = todo[start:start + flush]
        summaries = summarizer.summarize([fill_template(template, text, budget) for _, text in batch], budget)
        store.add_many([(key, summary) for (key, _), summary in zip(batch, summaries)])


def node_summaries(texts: list[str], nodes: list[int], summarizer, template: str, budget: int, store: TextStore,
                   shard: Shard | None = None) -> dict[int, str]:
    """Summaries by node id. In a shard, this holds only what is in the store so far."""
    unique = sorted(set(nodes))
    _fill(store, {str(u): texts[u] for u in unique}, summarizer, template, budget, NODE_FLUSH, shard)
    return {u: store[str(u)] for u in unique if str(u) in store}


def metapath_key(target: int, metapath: str, neighbours: list[int]) -> str:
    digest = hashlib.sha1(",".join(map(str, neighbours)).encode()).hexdigest()[:12]
    return f"{target}|{metapath}|{digest}"


def metapath_summaries(requests: list[tuple[int, str, list[int]]], node_text: dict[int, str], summarizer, template: str,
                       budget: int, store: TextStore, shard: Shard | None = None) -> dict[tuple[int, str], str]:
    """requests: (target, metapath name, trimmed neighbours, closest first). Empty neighbourhoods are skipped."""
    items = {metapath_key(v, name, found): "\n".join(node_text[u] for u in found)
             for v, name, found in requests if found and in_shard(metapath_key(v, name, found), shard)}
    _fill(store, items, summarizer, template, budget, METAPATH_FLUSH, shard)
    keys = {(v, name): metapath_key(v, name, found) for v, name, found in requests if found}
    return {pair: store[key] for pair, key in keys.items() if key in store}


def numeric_mean(numeric: np.ndarray, neighbours: np.ndarray) -> np.ndarray | None:
    """Mean of the trimmed neighbours' features, target excluded. None when there are no neighbours."""
    return numeric[neighbours].astype(np.float64).mean(axis=0) if len(neighbours) else None
