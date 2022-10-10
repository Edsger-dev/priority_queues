"""scikit-network

https://scikit-network.readthedocs.io/en/latest/reference/path.html#shortest-path
"""

from argparse import ArgumentParser
from time import perf_counter

from sknetwork.path import get_distances
import numpy as np
import pandas as pd
from scipy.sparse import coo_array, csr_matrix

from priority_queues.shortest_path import ShortestPath, convert_sorted_graph_to_csr


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

network_file_path = (
    f"/home/francois/Data/Disk_1/DIMACS_road_networks/{reg}/USA-road-t.{reg}.gr.parquet"
)
IDX_FROM = 1000

edges_df = pd.read_parquet(network_file_path)
edges_df.rename(
    columns={"id_from": "source", "id_to": "target", "tt": "weight"}, inplace=True
)
vertex_count = edges_df[["source", "target"]].max().max() + 1
print(f"{len(edges_df)} edges and {vertex_count} vertices")

# scikit-network
# ==============

start = perf_counter()
graph_csr = convert_sorted_graph_to_csr(
    edges_df, "source", "target", "weight", vertex_count
)
end = perf_counter()
elapsed_time = end - start
print(f"SKN Prepare the data - Elapsed time: {elapsed_time:6.2f} s")


start = perf_counter()
dist_matrix_ref = get_distances(
    adjacency=graph_csr,
    sources=IDX_FROM,
    method="D",
    return_predecessors=False,
    unweighted=False,
    n_jobs=-1,
)
end = perf_counter()
elapsed_time = end - start
print(f"SKN Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

# In-house
# ========

# without graph permutation
# return_inf=True
start = perf_counter()
sp = ShortestPath(edges_df, orientation="one-to-all", check_edges=False, permute=False)
end = perf_counter()
elapsed_time = end - start
print(f"PQ Prepare the data - Elapsed time: {elapsed_time:6.2f} s")

start = perf_counter()
dist_matrix = sp.run(vertex_idx=IDX_FROM, return_inf=True, return_Series=False)
end = perf_counter()
elapsed_time = end - start
print(f"PQ Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

assert np.allclose(dist_matrix, dist_matrix_ref, rtol=1e-05, atol=1e-08, equal_nan=True)

print("done")
