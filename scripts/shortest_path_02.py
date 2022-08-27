"""SciPy spasre coo_array input.
"""

import pandas as pd
from scipy.sparse import coo_array, csr_matrix

edges_df = pd.DataFrame(
    data={"tail": [2, 1, 0, 0], "head": [3, 3, 2, 1], "weight": [3.0, 1.0, 2.0, 1.0]}
)
vertex_count = edges_df[["tail", "head"]].max().max() + 1
data = edges_df["weight"].values
row = edges_df["tail"].values
col = edges_df["head"].values
graph_coo = coo_array((data, (row, col)), shape=(vertex_count, vertex_count))
graph_csr = graph_coo.tocsr()

print(graph_csr)
