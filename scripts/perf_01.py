from argparse import ArgumentParser, ArgumentTypeError
from time import perf_counter

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
    f"/home/francois/Workspace/posts_priority_queue//data/{reg}/{reg}.parquet"
)
IDX_FROM = 1000

edges_df = pd.read_parquet(NETWORK_FILE_PATH)
edges_df.rename(
    columns={"id_from": "source", "id_to": "target", "tt": "weight"}, inplace=True
)

print(edges_df.head(3))
# print(edges_df.min())
# print(edges_df.max())
# print(edges_df.isna().any())

# SciPy

start = perf_counter()

vertex_count = edges_df[["source", "target"]].max().max() + 1
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

# In-house
# without graph permutation
# return_inf=True
start = perf_counter()
sp = ShortestPath(edges_df, orientation="one-to-all", check_edges=False, permute=False)
end = perf_counter()
elapsed_time = end - start
print(f"PQ Prepare the data - Elapsed time: {elapsed_time:6.2f} s")

start = perf_counter()
path_lengths = sp.run(vertex_idx=IDX_FROM, return_inf=True)
dist_matrix = path_lengths.values
end = perf_counter()
elapsed_time = end - start
print(f"PQ Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

time_df = sp.get_timings()
print(time_df)

print(f"{len(edges_df)} edges and {vertex_count} vertices")
assert np.allclose(dist_matrix, dist_matrix_ref, rtol=1e-05, atol=1e-08, equal_nan=True)
