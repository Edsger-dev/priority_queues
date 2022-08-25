"""A small SSSP test case.
"""

import numpy as np
import pandas as pd
from priority_queues.shortest_path import ShortestPath
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

result_ref = [0.0, 1.0, 2.0, 2.0]

# SciPy

graph = np.array(
    [[0, 1, 2, 0], [0, 0, 0, 1], [0, 0, 0, 3], [0, 0, 0, 0]], dtype=np.float64
)

graph = csr_matrix(graph)

dist_matrix = dijkstra(
    csgraph=graph, directed=False, indices=0, return_predecessors=False
)

assert (dist_matrix == result_ref).all()

# in-house

edges_df = pd.DataFrame(
    data={"tail": [0, 0, 1, 2], "head": [1, 2, 3, 3], "weight": [1.0, 2.0, 1.0, 3.0]}
)

sp = ShortestPath(
    edges_df,
    source="tail",
    target="head",
    weight="weight",
    orientation="one-to-all",
    check_edges=True,
)
path_lengths = sp.run(vertex_idx=0)

assert (path_lengths.values == result_ref).all()

# timings
timings = pd.DataFrame.from_dict(sp.time, orient="index", columns=["elapsed_time_s"])
print(timings)

print("done")
