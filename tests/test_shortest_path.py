import pandas as pd
import pytest
from priority_queues.shortest_path import *


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
