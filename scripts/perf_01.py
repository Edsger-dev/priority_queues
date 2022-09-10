"""
https://askubuntu.com/questions/1052644/prevent-other-processes-for-performance
sudo nice -n -20 /home/francois/miniconda3/envs/algo/bin/python perf_01.py -n USA
"""


from argparse import ArgumentParser
import gc
import os
from time import perf_counter

import graph_tool as gt
from graph_tool import topology
from igraph import Graph
import networkit as nk
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

dist_matrix_ref = dijkstra(
    csgraph=graph_csr, directed=True, indices=IDX_FROM, return_predecessors=False
)

end = perf_counter()
elapsed_time = end - start
print(f"SciPy Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

del data, row, col, graph_coo, graph_csr
gc.collect()

# iGraph
# ======

start = perf_counter()
g = Graph.DataFrame(edges_df, directed=True)
end = perf_counter()
elapsed_time = end - start
print(f"iG Load the graph - Elapsed time: {elapsed_time:6.2f} s")

start = perf_counter()

distances = g.distances(source=IDX_FROM, target=None, weights="weight", mode="out")
dist_matrix_ig = np.asarray(distances[0])

end = perf_counter()
elapsed_time = end - start
print(f"iG Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

assert np.allclose(
    dist_matrix_ig, dist_matrix_ref, rtol=1e-05, atol=1e-08, equal_nan=True
)

del g, dist_matrix_ig

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
    g, source=g.vertex(IDX_FROM), weights=g.ep.t, negative_weights=False, directed=True
)
dist_matrix_gt = dist.a
end = perf_counter()
elapsed_time = end - start
print(f"GT Dijkstra - Elapsed time: {elapsed_time:6.2f} s")


assert np.allclose(
    dist_matrix_gt, dist_matrix_ref, rtol=1e-05, atol=1e-08, equal_nan=True
)

del g, eprop_t, dist, dist_matrix_gt
gc.collect()

# NetworkKit
# ==========

start = perf_counter()

nk_file_format = nk.graphio.Format.NetworkitBinary
networkit_file_path = (
    f"/home/francois/Workspace/posts_priority_queue/data/{reg}/{reg}.NetworkitBinary"
)

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


# Run
start = perf_counter()
dijkstra.run()

dist_matrix = dijkstra.getDistances(asarray=True)
dist_matrix_nk = np.asarray(dist_matrix)
dist_matrix_nk = np.where(dist_matrix_nk >= 1.79769313e308, np.inf, dist_matrix_nk)

end = perf_counter()
elapsed_time = end - start
print(f"NK Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

assert np.allclose(
    dist_matrix_nk, dist_matrix_ref, rtol=1e-05, atol=1e-08, equal_nan=True
)

del g, dijkstra, dist_matrix, dist_matrix_nk
gc.collect()

# priority_queues
# ===============

start = perf_counter()
sp = ShortestPath(edges_df, orientation="one-to-all", check_edges=False, permute=False)
end = perf_counter()
elapsed_time = end - start
print(f"PQ Prepare the data - Elapsed time: {elapsed_time:6.2f} s")

start = perf_counter()
dist_matrix_pq = sp.run(vertex_idx=IDX_FROM, return_inf=True, return_Series=False)
end = perf_counter()
elapsed_time = end - start
print(f"PQ Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

time_df = sp.get_timings()
# print(time_df)


assert np.allclose(
    dist_matrix_pq, dist_matrix_ref, rtol=1e-05, atol=1e-08, equal_nan=True
)


del sp, dist_matrix_pq
gc.collect()
