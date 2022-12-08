from argparse import ArgumentParser
from time import perf_counter

import dyntapy
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
    f"/home/francois/Data/Disk_1/DIMACS_road_networks/{reg}/USA-road-t.{reg}.gr.parquet"
)
IDX_FROM = 1000

edges_df = pd.read_parquet(NETWORK_FILE_PATH)
edges_df.rename(
    columns={"id_from": "source", "id_to": "target", "tt": "weight"}, inplace=True
)
vertex_count = edges_df[["source", "target"]].max().max() + 1
print(f"{len(edges_df)} edges and {vertex_count} vertices")

# Dyntapy
# =======

# about twice as fast as igraph

start = perf_counter()

edges_df.rename(
    columns={"source": "from_node_id", "target": "to_node_id", "weight": "cost"},
    inplace=True,
)
edges_df["link_id"] = edges_df.index
edges_df["capacity"] = 10000
edges_df["free_speed"] = 13.888
edges_df["length"] = edges_df["cost"] * edges_df["free_speed"]
edges_df["lanes"] = 1
graph = nx.from_pandas_edgelist(
    edges_df,
    source="from_node_id",
    target="to_node_id",
    edge_attr=[
        "from_node_id",
        "to_node_id",
        "link_id",
        "cost",
        "length",
        "capacity",
        "free_speed",
        "lanes",
    ],
    create_using=nx.DiGraph,
)
d = dict(zip(np.arange(vertex_count), np.arange(vertex_count)))
nx.set_node_attributes(graph, d, name="node_id")
d = dict(zip(np.arange(vertex_count), np.ones(vertex_count, dtype=float)))
nx.set_node_attributes(graph, d, name="x_coord")
nx.set_node_attributes(graph, d, name="y_coord")
network = dyntapy.graph_utils.build_network(graph)
costs = network.links.cost
out_links = network.nodes.out_links
is_centroid = network.nodes.is_centroid


# edges_df["lanes"] = 1
# edges_df["capacity"] = 10000
# edges_df["free_speed"] = 13.888
# edges_df["length"] = edges_df["weight"] * edges_df["free_speed"]
# edges_df["link_id"] = edges_df.index

# edges_df.rename(
#     columns={"source": "from_node_id", "target": "to_node_id"}, inplace=True
# )

# graph = nx.from_pandas_edgelist(
#     edges_df,
#     source="from_node_id",
#     target="to_node_id",
#     edge_attr=["length", "capacity", "free_speed", "lanes", "link_id"],
#     create_using=nx.DiGraph,
# )
# assert graph.edges(data=True)

# d = dict(zip(np.arange(vertex_count), np.arange(vertex_count)))
# nx.set_node_attributes(graph, d, name="node_id")

# network = dyntapy.supply_data.build_network(graph, u_turns=False)

end = perf_counter()
elapsed_time = end - start
print(f"DT load graph - Elapsed time: {elapsed_time:6.2f} s")

start = perf_counter()

# distance_dict = nx.single_source_dijkstra_path_length(
#     G=graph, source=IDX_FROM, weight="weight"
# )
# dist_matrix_ref = np.inf * np.ones(vertex_count, dtype=np.float64)
# dist_matrix_ref[list(distance_dict.keys())] = list(distance_dict.values())

end = perf_counter()
elapsed_time = end - start
print(f"DT Dijkstra - Elapsed time: {elapsed_time:6.2f} s")


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


# assert np.allclose(dist_matrix, dist_matrix_ref, rtol=1e-05, atol=1e-08, equal_nan=True)


print("done")
