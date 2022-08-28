import numpy as np
import pandas as pd
import pytest
from priority_queues.commons import DTYPE_INF_PY
from priority_queues.shortest_path import *
from scipy.sparse import coo_array, csr_matrix
from scipy.sparse.csgraph import dijkstra


@pytest.fixture
def braess():
    """Braess-like graph"""
    edges_df = pd.DataFrame(
        data={
            "source": [0, 0, 1, 1, 2],
            "target": [1, 2, 2, 3, 3],
            "weight": [1.0, 2.0, 0.0, 2.0, 1.0],
        }
    )
    return edges_df


def test_check_edges_01():
    """Parallel edges."""

    edges_df = pd.DataFrame(
        data={
            "source": [0, 0],
            "target": [1, 1],
            "weight": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match=r"no parallel edges"):
        sp = ShortestPath(
            edges_df,
            check_edges=True,
        )


def test_check_edges_02():
    """Loops."""

    edges_df = pd.DataFrame(
        data={
            "source": [0, 1],
            "target": [1, 1],
            "weight": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match=r"no loop"):
        sp = ShortestPath(
            edges_df,
            check_edges=True,
        )


def test_check_edges_03():
    """Negative weights."""

    edges_df = pd.DataFrame(
        data={
            "source": [0, 0],
            "target": [1, 2],
            "weight": [1.0, -2.0],
        }
    )
    with pytest.raises(ValueError, match=r"nonnegative"):
        sp = ShortestPath(
            edges_df,
            check_edges=True,
        )


def test_check_edges_04(braess):

    edges_df = braess

    with pytest.raises(TypeError, match=r"pandas DataFrame"):
        sp = ShortestPath("yeaaahhh!!!", check_edges=True)
    with pytest.raises(KeyError, match=r"not found in graph edges dataframe"):
        sp = ShortestPath(edges_df, source="tail", check_edges=True)
    with pytest.raises(KeyError, match=r"not found in graph edges dataframe"):
        sp = ShortestPath(edges_df, target="head", check_edges=True)
    with pytest.raises(KeyError, match=r"not found in graph edges dataframe"):
        sp = ShortestPath(edges_df, weight="cost", check_edges=True)
    with pytest.raises(ValueError, match=r"missing value"):
        sp = ShortestPath(edges_df.replace(0, np.nan), check_edges=True)
    with pytest.raises(TypeError, match=r"should be of integer type"):
        sp = ShortestPath(edges_df.astype({"source": float}), check_edges=True)
    with pytest.raises(TypeError, match=r"should be of numeric type"):
        sp = ShortestPath(edges_df.astype({"weight": str}), check_edges=True)


def test_check_edges_05(braess):

    edges_df = braess
    sp = ShortestPath(edges_df, orientation="one-to-all", check_edges=True)
    assert (sp._indices == [1, 2, 2, 3, 3]).all()
    assert (sp._indptr == [0, 2, 4, 5, 5]).all()
    assert (sp._edge_weights == [1.0, 2.0, 0.0, 2.0, 1.0]).all()


def test_run_01(braess):

    edges_df = braess
    sp = ShortestPath(edges_df, orientation="one-to-all", check_edges=False)
    path_lengths = sp.run(vertex_idx=0)
    path_lengths_ref = pd.Series([0.0, 1.0, 1.0, 2.0])
    path_lengths_ref.index.name = "vertex_idx"
    path_lengths_ref.name = "path_length"
    pd.testing.assert_series_equal(path_lengths, path_lengths_ref)


def test_run_02(random_seed=124, n=1000):

    source = np.random.randint(0, int(n / 5), n)
    target = np.random.randint(0, int(n / 5), n)
    weight = np.random.rand(n)
    edges_df = pd.DataFrame(data={"source": source, "target": target, "weight": weight})
    edges_df.drop_duplicates(subset=["source", "target"], inplace=True)
    edges_df = edges_df.loc[edges_df["source"] != edges_df["target"]]
    edges_df.reset_index(drop=True, inplace=True)

    # SciPy
    vertex_count = edges_df[["source", "target"]].max().max() + 1
    data = edges_df["weight"].values
    row = edges_df["source"].values
    col = edges_df["target"].values
    graph_coo = coo_array((data, (row, col)), shape=(vertex_count, vertex_count))
    graph_csr = graph_coo.tocsr()
    dist_matrix_ref = dijkstra(
        csgraph=graph_csr, directed=True, indices=0, return_predecessors=False
    )
    dist_matrix_ref = np.where(
        dist_matrix_ref > DTYPE_INF_PY, DTYPE_INF_PY, dist_matrix_ref
    )

    # In-house

    # without grapth permutation
    sp = ShortestPath(
        edges_df, orientation="one-to-all", check_edges=True, permute=False
    )
    path_lengths = sp.run(vertex_idx=0)
    dist_matrix = path_lengths.values

    np.testing.assert_array_almost_equal(dist_matrix, dist_matrix_ref, decimal=8)

    # with graph permutation
    sp = ShortestPath(
        edges_df, orientation="one-to-all", check_edges=True, permute=True
    )
    path_lengths = sp.run(vertex_idx=0)
    dist_matrix = path_lengths.values

    np.testing.assert_array_almost_equal(dist_matrix, dist_matrix_ref, decimal=8)
