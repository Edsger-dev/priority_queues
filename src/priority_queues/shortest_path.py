# TODO : argument permute_graph bool?
# QUESTION : maybe there can be a loop in the graph?

import numpy as np
import pandas as pd
from scipy.sparse import coo_array, csc_matrix, csr_matrix

# )
from priority_queues.commons import DTYPE_PY, Timer
from priority_queues.dijkstra import path_length_from


def convert_sorted_graph_to_csr(edges_df, source, target, weight, vertex_count):
    """Compute the CSR representation of the node-node adjacency matrix of the
    graph.

    input
    =====
    * pd.DataFrame edges_df :  edge dataframe
    * str source : name of the source column
    * str target : name of the target column
    * str weight : name of the weight column
    * int vertex_count : number of edges

    output
    ======
    * TODO
    """

    data = edges_df[weight].values
    row = edges_df[source].values
    col = edges_df[target].values
    graph_coo = coo_array((data, (row, col)), shape=(vertex_count, vertex_count))
    graph_csr = graph_coo.tocsr()

    return graph_csr


def convert_sorted_graph_to_csc(edges_df, source, target, weight, vertex_count):
    """Compute the CSC representation of the node-node adjacency matrix of the
    graph.

    input
    =====
    * pd.DataFrame edges_df :  edge dataframe
    * str source : name of the source column
    * str target : name of the target column
    * str weight : name of the weight column
    * int vertex_count : number of edges

    output
    ======
    * TODO
    """

    data = edges_df[weight].values
    row = edges_df[source].values
    col = edges_df[target].values
    graph_coo = coo_array((data, (row, col)), shape=(vertex_count, vertex_count))

    return csc_mat.indptr.astype(np.intp)


class ShortestPath:
    def __init__(
        self,
        edges_df,
        source="source",
        target="target",
        weight="weight",
        orientation="one-to-all",
        check_edges=True,
    ):
        """TODO : check if we need to deep copy edges_df"""
        self.time = {}

        t = Timer()
        t.start()
        # load the edges
        if check_edges:
            self._check_edges(edges_df, source, target, weight)
        self._edges = edges_df[[source, target, weight]]
        self.n_edges = len(self._edges)
        t.stop()
        self.time["load the edges"] = t.interval

        # reindex the vertices
        t = Timer()
        t.start()
        self._vertices = self._permute_graph(source, target)
        self.n_vertices = len(self._vertices)
        t.stop()
        self.time["reindex the vertices"] = t.interval

        # cast the weight types
        t = Timer()
        t.start()
        self._edges[weight] = self._edges[weight].astype(DTYPE_PY)
        t.stop()
        self.time["cast the types"] = t.interval

        # convert to CSR/CSC
        t = Timer()
        t.start()
        self._check_orientation(orientation)
        self._orientation = orientation
        if self._orientation == "one-to-all":
            graph_csr = convert_sorted_graph_to_csr(
                self._edges, source, target, weight, self.n_vertices
            )
            self._indices = graph_csr.indices.astype(np.intp)
            self._indptr = graph_csr.indptr.astype(np.intp)
            self._edge_weights = graph_csr.data
        else:
            graph_csc = convert_sorted_graph_to_csc(
                self._edges, source, target, weight, self.n_vertices
            )
            self._indices = graph_csc.indices.astype(np.intp)
            self._indptr = graph_csc.indptr.astype(np.intp)
            self._edge_weights = graph_csc.data
            raise NotImplementedError("one-to_all shortest path not implemented yet")
        t.stop()
        self.time["convert to CSR/CSC"] = t.interval

    def _check_edges(self, edges_df, source, target, weight):

        if type(edges_df) != pd.core.frame.DataFrame:
            raise TypeError("edges_df should be a pandas DataFrame")

        if source not in edges_df:
            raise KeyError(
                f"edge source column '{source}'  not found in graph edges dataframe"
            )

        if target not in edges_df:
            raise KeyError(
                f"edge target column '{target}' not found in graph edges dataframe"
            )

        if weight not in edges_df:
            raise KeyError(
                f"edge weight column '{weight}' not found in graph edges dataframe"
            )

        if edges_df[[source, target, weight]].isna().any().any():
            raise ValueError(
                " ".join(
                    [
                        f"edges_df[[{source}, {target}, {weight}]] ",
                        "should not have any missing value",
                    ]
                )
            )

        for col in [source, target]:
            if not pd.api.types.is_integer_dtype(edges_df[col].dtype):
                raise TypeError(f"edges_df['{col}'] should be of integer type")

        if not pd.api.types.is_numeric_dtype(edges_df[weight].dtype):
            raise TypeError(f"edges_df['{weight}'] should be of numeric type")

        if edges_df[weight].min() < 0.0:
            raise ValueError(f"edges_df['{weight}'] should be nonnegative")

        if not np.isfinite(edges_df[weight]).all():
            raise ValueError(f"edges_df['{weight}'] should be finite")

        # the graph must be a simple directed graphs
        if edges_df.duplicated(subset=[source, target]).any():
            raise ValueError("there should be no parallel edges in the graph")
        if (edges_df[source] == edges_df[target]).any():
            raise ValueError("there should be no loop in the graph")

    def _permute_graph(self, source, target):
        """Create a vertex table and reindex the vertices."""

        vertices = pd.DataFrame(
            data={
                "vert_idx": np.union1d(
                    self._edges[source].values, self._edges[target].values
                )
            }
        )
        vertices["vert_idx_new"] = vertices.index
        vertices.index.name = "index"

        self._edges = pd.merge(
            self._edges,
            vertices[["vert_idx", "vert_idx_new"]],
            left_on=source,
            right_on="vert_idx",
            how="left",
        )
        self._edges.drop([source, "vert_idx"], axis=1, inplace=True)
        self._edges.rename(columns={"vert_idx_new": source}, inplace=True)

        self._edges = pd.merge(
            self._edges,
            vertices[["vert_idx", "vert_idx_new"]],
            left_on=target,
            right_on="vert_idx",
            how="left",
        )
        self._edges.drop([target, "vert_idx"], axis=1, inplace=True)
        self._edges.rename(columns={"vert_idx_new": target}, inplace=True)

        vertices.rename(columns={"vert_idx": "vert_idx_old"}, inplace=True)
        vertices.reset_index(drop=True, inplace=True)
        vertices.sort_values(by="vert_idx_new", inplace=True)

        vertices.index.name = "index"
        self._edges.index.name = "index"

        return vertices

    def _check_orientation(self, orientation):
        if orientation not in ["one-to-all", "all-to-one"]:
            raise ValueError(
                f"orientation should be either 'one-to-all' or 'all-to-one'"
            )

    def run(self, vertex_idx):

        # check the source/target vertex
        t = Timer()
        t.start()
        if vertex_idx not in self._vertices.vert_idx_old.values:
            raise ValueError(f"vertex {vertex_idx} not found in graph")
        vertex_new = self._vertices.loc[
            self._vertices.vert_idx_old == vertex_idx, "vert_idx_new"
        ]
        t.stop()
        self.time["check the source/target vertex"] = t.interval

        # compute path length
        t = Timer()
        t.start()
        if self._orientation == "one-to-all":
            path_lengths = path_length_from(
                self._indices,
                self._indptr,
                self._edge_weights,
                vertex_new,
                self.n_vertices,
                n_jobs=-1,
            )
        t.stop()
        self.time["compute path length"] = t.interval

        # reorder results
        t = Timer()
        t.start()
        self._vertices["path_length"] = path_lengths
        path_lengths_df = self._vertices[["vert_idx_old", "path_length"]].sort_values(
            by="vert_idx_old"
        )
        path_lengths_df.set_index("vert_idx_old", drop=True, inplace=True)
        path_lengths_df.index.name = "vertex_idx"
        path_lengths_series = path_lengths_df.path_length
        t.stop()
        self.time["reorder results"] = t.interval

        return path_lengths_series
