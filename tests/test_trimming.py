import numpy as np
import scipy.sparse as sp

from dgp_plus.trimming import (diffused_embeddings, enumerate_metapaths, load_trim, metapath_name, metapath_neighbours,
                                   save_trim, top_m, trim_metapath)

CHAIN = sp.csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=float))


def dense_diffusion(adjacency: np.ndarray, features: np.ndarray, K: int) -> np.ndarray:
    """Eq. 6 written out with dense matrices, as a reference."""
    degree = adjacency.sum(axis=1)
    transition = np.diag(np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)) @ adjacency
    power, total = np.eye(len(adjacency)), np.zeros_like(adjacency)
    for _ in range(K + 1):
        total += power
        power = power @ transition
    return total / K @ features


def test_metapaths_count_and_order():
    paths = enumerate_metapaths(["A", "B", "C"], 2)
    assert len(paths) == 12 and paths[0] == ("A",) and paths[3] == ("A", "A")
    assert metapath_name(("A", "B")) == "A-B"


def test_neighbours_exclude_the_target():
    found = metapath_neighbours({"X": CHAIN}, ("X", "X"), np.array([1]))
    assert found[0].tolist() == [3]


def test_diffusion_matches_dense_formula():
    features = np.random.default_rng(0).normal(size=(4, 3))
    for K in (1, 2, 3):
        expected = dense_diffusion(CHAIN.toarray(), features, K)
        assert np.allclose(diffused_embeddings({"X": CHAIN}, ("X",), features, K), expected)


def test_zero_degree_nodes_keep_a_zero_row():
    empty = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float))
    features = np.ones((3, 1))
    assert diffused_embeddings({"X": empty}, ("X",), features, 2)[2, 0] == 1 / 2      # only the k = 0 term remains


def test_top_m_is_closest_first_with_id_tie_break():
    embeddings = np.array([[0.0], [2.0], [1.0], [1.0]])
    assert top_m(np.array([1, 3, 2]), embeddings, 0, 2).tolist() == [2, 3]


def test_trim_keeps_at_most_m_and_handles_empty():
    relation = sp.csr_matrix(np.array([[0, 1, 1, 1, 0], [1, 0, 1, 1, 0], [1, 1, 0, 1, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=float))
    trim = trim_metapath({"X": relation}, ("X",), np.arange(5.0).reshape(-1, 1), np.array([0, 4]), K=2, M=2)
    assert len(trim[0]) == 2 and len(trim[4]) == 0


def test_trim_file_round_trip(tmp_path):
    trim = {3: np.array([1, 2]), 7: np.zeros(0, dtype=np.int64)}
    save_trim(tmp_path / "t.npz", trim)
    loaded = load_trim(tmp_path / "t.npz")
    assert loaded[3].tolist() == [1, 2] and len(loaded[7]) == 0
