from argparse import ArgumentParser
from time import perf_counter

import networkit as nk
import numpy as np
import pandas as pd
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


# NetworkKit
# ==========

start = perf_counter()

g = nk.Graph(n=vertex_count, weighted=True, directed=True, edgesIndexed=False)

for row in edges_df.itertuples():
    g.addEdge(row.source, row.target, w=row.weight)
# print(g.numberOfNodes())
# print(g.numberOfEdges())

end = perf_counter()
elapsed_time = end - start
print(f"NK Load the graph - Elapsed time: {elapsed_time:6.2f} s")

start = perf_counter()

dijkstra = nk.distance.Dijkstra(
    g, IDX_FROM, storePaths=False, storeNodesSortedByDistance=False
)

end = perf_counter()
elapsed_time = end - start
print(f"NK Dijkstra init - Elapsed time: {elapsed_time:6.2f} s")

# Run
start = perf_counter()
dijkstra.run()


dist_matrix = dijkstra.getDistances(asarray=True)
dist_matrix_ref = np.asarray(dist_matrix)
dist_matrix_ref = np.where(dist_matrix_ref >= 1.79769313e308, np.inf, dist_matrix_ref)

end = perf_counter()
elapsed_time = end - start
print(f"NK Dijkstra - Elapsed time: {elapsed_time:6.2f} s")


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

# time_df = sp.get_timings()
# print(time_df)

assert np.allclose(dist_matrix, dist_matrix_ref, rtol=1e-05, atol=1e-08, equal_nan=True)


print("done")
