from argparse import ArgumentParser
from time import perf_counter

import networkx as nx
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

# NetworkX
# ========

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
dist_matrix_ref = np.inf * np.ones(vertex_count, dtype=np.float64)
dist_matrix_ref[list(distance_dict.keys())] = list(distance_dict.values())

end = perf_counter()
elapsed_time = end - start
print(f"nx Dijkstra - Elapsed time: {elapsed_time:6.2f} s")


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
