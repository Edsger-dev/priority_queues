from time import perf_counter

import numpy as np
import pandas as pd
from priority_queues.shortest_path import ShortestPath
from scipy.sparse import coo_array, csr_matrix
from scipy.sparse.csgraph import dijkstra

NETWORK_FILE_PATH = "/home/francois/Workspace/posts_priority_queue/data/USA/USA.parquet"

edges_df = pd.read_parquet(NETWORK_FILE_PATH)
edges_df.rename(
    columns={"id_from": "source", "id_to": "target", "tt": "weight"}, inplace=True
)

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
    csgraph=graph_csr, directed=True, indices=0, return_predecessors=False
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

path_lengths = sp.run(vertex_idx=0, return_inf=True)
dist_matrix = path_lengths.values

end = perf_counter()
elapsed_time = end - start
print(f"PQ Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

# not really understanding what happens when computing the difference of inf
assert np.testing.assert_array_almost_equal(dist_matrix, dist_matrix_ref, decimal=2)
