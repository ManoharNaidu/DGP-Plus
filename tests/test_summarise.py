import numpy as np

from dgp_plus.cache import TextStore, artifact_dir
from dgp_plus.pipeline import WordTokenizer
from dgp_plus.summarise import (MockSummarizer, cap_source_tokens, fill_template, in_shard, metapath_key,
                                    metapath_summaries, node_summaries, numeric_mean)

TEMPLATE = "Summarize the text within {budget} tokens.\n\n{text}"


class Counting(MockSummarizer):
    calls = 0

    def summarize(self, prompts, budget):
        Counting.calls += len(prompts)
        return super().summarize(prompts, budget)


def test_template_fills_both_fields():
    assert fill_template(TEMPLATE, "hello", 10) == "Summarize the text within 10 tokens.\n\nhello"


def test_store_survives_a_half_written_line(tmp_path):
    store = TextStore(tmp_path / "s.jsonl")
    store.add_many([("a", "one")])
    with open(tmp_path / "s.jsonl", "a") as f:
        f.write('{"key": "b", "te')
    assert len(TextStore(tmp_path / "s.jsonl")) == 1


def test_cache_key_changes_with_content(tmp_path):
    assert artifact_dir(tmp_path, "s", {"a": 1}) != artifact_dir(tmp_path, "s", {"a": 2})
    assert artifact_dir(tmp_path, "s", {"a": 1}) == artifact_dir(tmp_path, "s", {"a": 1})


def test_shards_cover_every_key_once():
    keys = [str(i) for i in range(500)]
    owners = [sum(in_shard(k, (i, 4)) for i in range(4)) for k in keys]
    assert owners == [1] * 500


def test_shards_fill_one_store_and_a_rerun_generates_nothing(tmp_path):
    store = TextStore(tmp_path / "s.jsonl")
    texts = [f"word{i} and more" for i in range(40)]
    Counting.calls = 0
    for index in range(3):
        node_summaries(texts, list(range(40)), Counting(), TEMPLATE, 2, store, (index, 3))
    assert len(store) == 40 and Counting.calls == 40
    assert len(node_summaries(texts, list(range(40)), Counting(), TEMPLATE, 2, store)) == 40
    assert Counting.calls == 40


def test_metapath_summary_uses_neighbour_order_and_skips_empty(tmp_path):
    store = TextStore(tmp_path / "s.jsonl")
    text = metapath_summaries([(1, "A", [2, 3]), (1, "B", [])], {2: "two", 3: "three"}, MockSummarizer(), TEMPLATE, 5, store)
    assert text == {(1, "A"): "two three"} and metapath_key(1, "A", [2, 3]) in store
    assert metapath_key(1, "A", [3, 2]) != metapath_key(1, "A", [2, 3])


def test_numeric_mean():
    numeric = np.array([[1.0], [3.0], [100.0]])
    assert numeric_mean(numeric, np.array([0, 1])).tolist() == [2.0]
    assert numeric_mean(numeric, np.zeros(0, dtype=int)) is None


def test_cap_cuts_only_listed_long_nodes():
    texts = ["a b c d e", "a b", "a b c d e"]
    capped, cut = cap_source_tokens(texts, [0, 1], WordTokenizer(), 3)
    assert capped == ["a b c", "a b", "a b c d e"] and cut == 1 and texts[0] == "a b c d e"
