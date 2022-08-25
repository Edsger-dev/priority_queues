import pandas as pd
import pytest
from priority_queues.shortest_path import *

# @pytest.fixture
# def braess():
#     """ Braess-like graph
#     """
#     edges_df = pd.DataFrame(
#         data={
#             "source": [0, 0, 1, 1, 2],
#             "target": [1, 2, 2, 3, 3],
#             "weight": [1.0, 2.0, 0.0, 2.0, 1.0],
#         }
#     )
#     return edges_df


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
