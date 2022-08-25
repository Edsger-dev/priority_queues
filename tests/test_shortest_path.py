import pandas as pd
from priority_queues.shortest_path import *

# import pytest


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
