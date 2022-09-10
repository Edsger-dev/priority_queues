"""
single library
No check for solution

https://askubuntu.com/questions/1052644/prevent-other-processes-for-performance
sudo nice -n -20 /home/francois/miniconda3/envs/algo/bin/python perf_01.py -n USA
"""

from argparse import ArgumentParser
import os
from time import perf_counter

import graph_tool as gt
from graph_tool import topology
from igraph import Graph
import networkit as nk
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_array, csr_matrix
from scipy.sparse.csgraph import dijkstra

from priority_queues.shortest_path import ShortestPath

parser = ArgumentParser(description="Command line interface to perf_01.py")
parser.add_argument(
    "-n",
    "--network",
    dest="network_name",
    help='network name, must be "NY", "BAY", "COL", "FLA", "NW", "NE", "CAL", "LKS", "E", "W", "CTR", "USA"',
    metavar="TXT",
    type=str,
    required=True,
)
parser.add_argument(
    "-l",
    "--library",
    dest="library_name",
    help='library name, must be "SP" (SciPy), "PQ" (priority_queues), "IG" (iGraph), "GT" (graph-tools), "NK" (NetworkKit), "NX" (NetworkX)',
    metavar="TXT",
    type=str,
    required=True,
)
args = parser.parse_args()
reg = args.network_name.upper()
assert reg in [
    "NY",
    "BAY",
    "COL",
    "FLA",
    "NW",
    "NE",
    "CAL",
    "LKS",
    "E",
    "W",
    "CTR",
    "USA",
]
lib = args.library_name.upper()
assert lib in ["SP", "PQ", "IG", "GT", "NK", "NX"]

# load the edges as a dataframe

NETWORK_FILE_PATH = (
    f"/home/francois/Workspace/posts_priority_queue/data/{reg}/{reg}.parquet"
)
IDX_FROM = 1000

edges_df = pd.read_parquet(NETWORK_FILE_PATH)
edges_df.rename(
    columns={"id_from": "source", "id_to": "target", "tt": "weight"}, inplace=True
)
vertex_count = edges_df[["source", "target"]].max().max() + 1
print(f"{len(edges_df)} edges and {vertex_count} vertices")

if lib == "SP":

    # SciPy
    # =====

    start = perf_counter()

    data = edges_df["weight"].values
    row = edges_df["source"].values
    col = edges_df["target"].values
    graph_coo = coo_array((data, (row, col)), shape=(vertex_count, vertex_count))
    graph_csr = graph_coo.tocsr()

    end = perf_counter()
    elapsed_time = end - start
    print(f"SciPy Prepare the data - Elapsed time: {elapsed_time:6.2f} s")

    start = perf_counter()

    dist_matrix = dijkstra(
        csgraph=graph_csr, directed=True, indices=IDX_FROM, return_predecessors=False
    )

    end = perf_counter()
    elapsed_time = end - start
    print(f"SciPy Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

elif lib == "PQ":

    # priority_queues
    # ===============

    start = perf_counter()
    sp = ShortestPath(
        edges_df, orientation="one-to-all", check_edges=False, permute=False
    )
    end = perf_counter()
    elapsed_time = end - start
    print(f"PQ Prepare the data - Elapsed time: {elapsed_time:6.2f} s")

    start = perf_counter()
    dist_matrix = sp.run(vertex_idx=IDX_FROM, return_inf=True, return_Series=False)
    end = perf_counter()
    elapsed_time = end - start
    print(f"PQ Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

    time_df = sp.get_timings()
    # print(time_df)


elif lib == "IG":

    # iGraph
    # ======

    start = perf_counter()
    g = Graph.DataFrame(edges_df, directed=True)
    end = perf_counter()
    elapsed_time = end - start
    print(f"iG Load the graph - Elapsed time: {elapsed_time:6.2f} s")

    start = perf_counter()

    distances = g.distances(source=IDX_FROM, target=None, weights="weight", mode="out")
    dist_matrix = np.asarray(distances[0])

    end = perf_counter()
    elapsed_time = end - start
    print(f"iG Dijkstra - Elapsed time: {elapsed_time:6.2f} s")


elif lib == "GT":

    # Graph-tools
    # ===========

    start = perf_counter()

    # create the graph
    g = gt.Graph(directed=True)

    # create the vertices
    g.add_vertex(vertex_count)

    # create the edges
    g.add_edge_list(edges_df[["source", "target"]].values)

    # edge property for the travel time
    eprop_t = g.new_edge_property("float")
    g.edge_properties["t"] = eprop_t  # internal property
    g.edge_properties["t"].a = edges_df["weight"].values

    end = perf_counter()
    elapsed_time = end - start
    print(f"GT Load the graph - Elapsed time: {elapsed_time:6.2f} s")

    start = perf_counter()
    dist = topology.shortest_distance(
        g,
        source=g.vertex(IDX_FROM),
        weights=g.ep.t,
        negative_weights=False,
        directed=True,
    )
    dist_matrix = dist.a
    end = perf_counter()
    elapsed_time = end - start
    print(f"GT Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

elif lib == "NK":

    # NetworkKit
    # ==========

    start = perf_counter()

    nk_file_format = nk.graphio.Format.NetworkitBinary
    networkit_file_path = f"/home/francois/Workspace/posts_priority_queue/data/{reg}/{reg}.NetworkitBinary"

    if os.path.exists(networkit_file_path):

        g = nk.graphio.readGraph(networkit_file_path, nk_file_format)

    else:

        g = nk.Graph(n=vertex_count, weighted=True, directed=True, edgesIndexed=False)

        for row in edges_df.itertuples():
            g.addEdge(row.source, row.target, w=row.weight)

        nk.graphio.writeGraph(g, networkit_file_path, nk_file_format)

    dijkstra = nk.distance.Dijkstra(
        g, IDX_FROM, storePaths=False, storeNodesSortedByDistance=False
    )

    end = perf_counter()
    elapsed_time = end - start
    print(f"NK Load the graph - Elapsed time: {elapsed_time:6.2f} s")

    start = perf_counter()
    dijkstra.run()

    dist_matrix = dijkstra.getDistances(asarray=True)
    dist_matrix = np.asarray(dist_matrix)
    dist_matrix = np.where(dist_matrix >= 1.79769313e308, np.inf, dist_matrix)

    end = perf_counter()
    elapsed_time = end - start
    print(f"NK Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

elif lib == "NX":

    start = perf_counter()

    graph = nx.from_pandas_edgelist(
        edges_df,
        source="source",
        target="target",
        edge_attr="weight",
        create_using=nx.DiGraph,
    )

    # print(graph.edges(data=True))

    end = perf_counter()
    elapsed_time = end - start
    print(f"nx load graph - Elapsed time: {elapsed_time:6.2f} s")

    start = perf_counter()

    distance_dict = nx.single_source_dijkstra_path_length(
        G=graph, source=IDX_FROM, weight="weight"
    )
    dist_matrix = np.inf * np.ones(vertex_count, dtype=np.float64)
    dist_matrix[list(distance_dict.keys())] = list(distance_dict.values())

    end = perf_counter()
    elapsed_time = end - start
    print(f"nx Dijkstra - Elapsed time: {elapsed_time:6.2f} s")