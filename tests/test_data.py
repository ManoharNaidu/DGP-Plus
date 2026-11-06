import numpy as np
import pytest

from conftest import make_graph
from dgp_plus.data import load_graph, load_split, make_split
from dgp_plus.raw_data import amazon_label, relation_from_keys


def test_graph_round_trip(tmp_path):
    graph = make_graph()
    graph.save(tmp_path)
    loaded = load_graph(tmp_path)
    assert loaded.texts == graph.texts and loaded.summary() == graph.summary()
    assert (loaded.relations["RUR"] != graph.relations["RUR"]).nnz == 0


def test_relation_links_shared_keys_without_self_loops():
    matrix = relation_from_keys(["a", "a", "b", None]).toarray()
    assert matrix.tolist() == [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def test_split_is_stratified_disjoint_and_repeatable():
    labels = np.array([1] * 20 + [0] * 80 + [-1] * 5)
    first = make_split(labels, {"train": 10, "val": 10, "test": 30}, seed=3)
    second = make_split(labels, {"train": 10, "val": 10, "test": 30}, seed=3)
    assert (first.test == second.test).all()
    assert len(set(first.train) | set(first.val) | set(first.test)) == 50
    assert labels[first.test].sum() == 6                  # 20% of 30
    assert (labels[first.targets()] != -1).all()


def test_missing_split_is_an_error_not_a_new_draw(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path)


@pytest.mark.parametrize("votes, label", [([0, 0], 0), ([1, 4], 1), ([2, 4], 0), ([4, 4], 0)])
def test_amazon_label(votes, label):
    assert amazon_label(votes) == label
