
from argparse import ArgumentParser
import gc
import os
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import coo_array, csr_matrix
from scipy.sparse.csgraph import dijkstra

from priority_queues.shortest_path import ShortestPath


n = 100000

np.random.seed(124)
source = np.random.randint(0, int(n / 5), n)
target = np.random.randint(0, int(n / 5), n)
weight = np.random.rand(n)
edges_df = pd.DataFrame(data={"source": source, "target": target, "weight": weight})
edges_df.drop_duplicates(subset=["source", "target"], inplace=True)
edges_df = edges_df.loc[edges_df["source"] != edges_df["target"]]
edges_df.reset_index(drop=True, inplace=True)
vertex_count = edges_df[["source", "target"]].max().max() + 1
IDX_FROM = 10

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

print(dist_matrix_ref[:100])

del data, row, col, graph_coo, graph_csr
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

print(dist_matrix_pq[:100])


time_df = sp.get_timings()
# print(time_df)


assert np.allclose(
    dist_matrix_pq, dist_matrix_ref, rtol=1e-05, atol=1e-08, equal_nan=True
)

del sp, dist_matrix_pq
gc.collect()