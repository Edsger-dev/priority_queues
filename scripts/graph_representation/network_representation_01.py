import os
from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import coo_array

from graph import loop_CSR, loop_FSV
from priority_queues.shortest_path import convert_sorted_graph_to_csr

DATA_DIR = "/home/francois/Data/Disk_1/"

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
reg = args.network_name

# network name check
regions_usa = [
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
regions_eur = ["osm-bawu", "osm-ger", "osm-eur"]
regions_all = regions_usa + regions_eur
assert reg in regions_all

continent = None
if reg in regions_usa:
    continent = "usa"
else:
    continent = "eur"

if continent == "usa":
    network_file_path = os.path.join(
        DATA_DIR, f"DIMACS_road_networks/{reg}/USA-road-t.{reg}.gr.parquet"
    )
else:
    network_file_path = os.path.join(DATA_DIR, f"OSMR/{reg}/{reg}.gr.parquet")


edges_df = pd.read_parquet(network_file_path)
edges_df.rename(
    columns={"id_from": "source", "id_to": "target", "tt": "weight"}, inplace=True
)
vertex_count = edges_df[["source", "target"]].max().max() + 1
print(f"{len(edges_df)} edges and {vertex_count} vertices")

# CSR
start = perf_counter()

data = edges_df["weight"].values
row = edges_df["source"].values
col = edges_df["target"].values
graph_coo = coo_array((data, (row, col)), shape=(vertex_count, vertex_count))
graph_csr = graph_coo.tocsr()

end = perf_counter()
elapsed_time = end - start
print(f"convert to CSR - Elapsed time: {elapsed_time:6.2f} s")

start = perf_counter()
loop_CSR(
    graph_csr.indptr.astype(np.intp),
    graph_csr.indices.astype(np.intp),
    graph_csr.data,
    vertex_count,
)
end = perf_counter()
elapsed_time = end - start
print(f"CSR loop - Elapsed time: {elapsed_time:6.2f} s")

# adjacency vectors
start = perf_counter()
loop_FSV(
    graph_csr.indptr.astype(np.intp),
    graph_csr.indices.astype(np.intp),
    graph_csr.data,
    vertex_count,
)
end = perf_counter()
elapsed_time = end - start
print(f"FSV whole process - Elapsed time: {elapsed_time:6.2f} s")
