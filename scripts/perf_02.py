"""
single library
No check for solution
"""

import os
from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import pandas as pd
from priority_queues.shortest_path import ShortestPath, convert_sorted_graph_to_csr
from scipy.sparse import coo_array
from scipy.sparse.csgraph import dijkstra

DATA_DIR = "/home/francois/Data/Disk_1/"


parser = ArgumentParser(description="Command line interface to perf_02.py")
parser.add_argument(
    "-n",
    "--network",
    dest="network_name",
    help='network name, must be "NY", "BAY", "COL", "FLA", "NW", "NE", "CAL", "LKS", "E", "W", "CTR", "USA"',
    metavar="TXT",
    type=str,
    required=True,
)
parser.add_argument(
    "-l",
    "--library",
    dest="library_name",
    help='library name, must be "SP" (SciPy), "PQ" (priority_queues), "IG" (iGraph), "GT" (graph-tools), "NK" (NetworkKit), "NX" (NetworkX), "SK" (scikit network)',
    metavar="TXT",
    type=str,
    required=True,
)
parser.add_argument(
    "-f",
    "--from",
    dest="idx_from",
    help="source vertex index",
    metavar="INT",
    type=int,
    required=False,
    default=1000,
)
parser.add_argument(
    "-t",
    "--heap_type",
    dest="heap_type",
    help="heap type in the priority_queues library : 'fib' or 'bin'",
    metavar="TXT",
    type=str,
    required=False,
    default="bin",
)
args = parser.parse_args()
reg = args.network_name
idx_from = args.idx_from
heap_type = args.heap_type


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

# lib name check
lib = args.library_name.upper()
assert lib in ["SP", "PQ", "IG", "GT", "NK", "NX", "SK"]

if continent == "usa":
    network_file_path = os.path.join(
        DATA_DIR, f"DIMACS_road_networks/{reg}/USA-road-t.{reg}.gr.parquet"
    )
else:
    network_file_path = os.path.join(DATA_DIR, f"OSMR/{reg}/{reg}.gr.parquet")

# shortest path
if lib == "SP":

    # SciPy
    # =====

    edges_df = pd.read_parquet(network_file_path)
    edges_df.rename(
        columns={"id_from": "source", "id_to": "target", "tt": "weight"}, inplace=True
    )
    vertex_count = edges_df[["source", "target"]].max().max() + 1
    print(f"{len(edges_df)} edges and {vertex_count} vertices")

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

    dist_matrix = dijkstra(
        csgraph=graph_csr, directed=True, indices=idx_from, return_predecessors=False
    )

    end = perf_counter()
    elapsed_time = end - start
    print(f"SciPy Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

elif lib == "PQ":

    # priority_queues
    # ===============

    edges_df = pd.read_parquet(network_file_path)
    edges_df.rename(
        columns={"id_from": "source", "id_to": "target", "tt": "weight"}, inplace=True
    )
    vertex_count = edges_df[["source", "target"]].max().max() + 1
    print(f"{len(edges_df)} edges and {vertex_count} vertices")

    start = perf_counter()
    sp = ShortestPath(
        edges_df,
        orientation="one-to-all",
        check_edges=False,
        permute=False,
        heap_type=heap_type,
    )
    end = perf_counter()
    elapsed_time = end - start
    print(f"PQ Prepare the data - Elapsed time: {elapsed_time:6.2f} s")

    start = perf_counter()
    dist_matrix = sp.run(vertex_idx=idx_from, return_inf=True, return_Series=False)
    end = perf_counter()
    elapsed_time = end - start
    print(f"PQ Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

    # time_df = sp.get_timings()
    # print(time_df)

elif lib == "IG":

    # iGraph
    # ======

    from igraph import Graph

    edges_df = pd.read_parquet(network_file_path)
    edges_df.rename(
        columns={"id_from": "source", "id_to": "target", "tt": "weight"}, inplace=True
    )
    vertex_count = edges_df[["source", "target"]].max().max() + 1
    print(f"{len(edges_df)} edges and {vertex_count} vertices")

    start = perf_counter()
    g = Graph.DataFrame(edges_df, directed=True)
    end = perf_counter()
    elapsed_time = end - start
    print(f"iG Load the graph - Elapsed time: {elapsed_time:6.2f} s")

    start = perf_counter()

    distances = g.distances(source=idx_from, target=None, weights="weight", mode="out")
    dist_matrix = np.asarray(distances[0])

    end = perf_counter()
    elapsed_time = end - start
    print(f"iG Dijkstra - Elapsed time: {elapsed_time:6.2f} s")


elif lib == "GT":

    # Graph-tools
    # ===========

    import graph_tool as gt
    from graph_tool import topology

    edges_df = pd.read_parquet(network_file_path)
    edges_df.rename(
        columns={"id_from": "source", "id_to": "target", "tt": "weight"}, inplace=True
    )
    vertex_count = edges_df[["source", "target"]].max().max() + 1
    print(f"{len(edges_df)} edges and {vertex_count} vertices")

    start = perf_counter()

    # create the graph
    g = gt.Graph(directed=True)

    # create the vertices
    g.add_vertex(vertex_count)

    # create the edges
    g.add_edge_list(edges_df[["source", "target"]].values)

    # edge property for the travel time
    eprop_t = g.new_edge_property("float")
    g.edge_properties["t"] = eprop_t  # internal property
    g.edge_properties["t"].a = edges_df["weight"].values

    end = perf_counter()
    elapsed_time = end - start
    print(f"GT Load the graph - Elapsed time: {elapsed_time:6.2f} s")

    start = perf_counter()
    dist = topology.shortest_distance(
        g,
        source=g.vertex(idx_from),
        weights=g.ep.t,
        negative_weights=False,
        directed=True,
    )
    dist_matrix = dist.a
    end = perf_counter()
    elapsed_time = end - start
    print(f"GT Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

elif lib == "NK":

    # NetworkKit
    # ==========

    import networkit as nk

    start = perf_counter()

    nk_file_format = nk.graphio.Format.NetworkitBinary
    if continent == "usa":
        networkit_file_path = os.path.join(
            DATA_DIR, f"DIMACS_road_networks/{reg}/USA-road-t.{reg}.gr.NetworkitBinary"
        )
    else:
        networkit_file_path = os.path.join(
            DATA_DIR, f"OSMR/{reg}/{reg}.gr.NetworkitBinary"
        )

    if os.path.exists(networkit_file_path):

        g = nk.graphio.readGraph(networkit_file_path, nk_file_format)

    else:

        edges_df = pd.read_parquet(network_file_path)
        edges_df.rename(
            columns={"id_from": "source", "id_to": "target", "tt": "weight"},
            inplace=True,
        )
        vertex_count = edges_df[["source", "target"]].max().max() + 1
        print(f"{len(edges_df)} edges and {vertex_count} vertices")

        g = nk.Graph(n=vertex_count, weighted=True, directed=True, edgesIndexed=False)

        for row in edges_df.itertuples():
            g.addEdge(row.source, row.target, w=row.weight)

        nk.graphio.writeGraph(g, networkit_file_path, nk_file_format)

    dijkstra = nk.distance.Dijkstra(
        g, idx_from, storePaths=False, storeNodesSortedByDistance=False
    )

    end = perf_counter()
    elapsed_time = end - start
    print(f"NK Load the graph - Elapsed time: {elapsed_time:6.2f} s")

    start = perf_counter()
    dijkstra.run()

    dist_matrix = dijkstra.getDistances(asarray=True)
    dist_matrix = np.asarray(dist_matrix)
    dist_matrix = np.where(dist_matrix >= 1.79769313e308, np.inf, dist_matrix)

    end = perf_counter()
    elapsed_time = end - start
    print(f"NK Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

elif lib == "NX":

    # NetworkX
    # ========

    import networkx as nx

    edges_df = pd.read_parquet(network_file_path)
    edges_df.rename(
        columns={"id_from": "source", "id_to": "target", "tt": "weight"}, inplace=True
    )
    vertex_count = edges_df[["source", "target"]].max().max() + 1
    print(f"{len(edges_df)} edges and {vertex_count} vertices")

    start = perf_counter()

    graph = nx.from_pandas_edgelist(
        edges_df,
        source="source",
        target="target",
        edge_attr="weight",
        create_using=nx.DiGraph,
    )

    end = perf_counter()
    elapsed_time = end - start
    print(f"nx load graph - Elapsed time: {elapsed_time:6.2f} s")

    assert graph.edges(data=True)

    start = perf_counter()

    distance_dict = nx.single_source_dijkstra_path_length(
        G=graph, source=idx_from, weight="weight"
    )
    dist_matrix = np.inf * np.ones(vertex_count, dtype=np.float64)
    dist_matrix[list(distance_dict.keys())] = list(distance_dict.values())

    end = perf_counter()
    elapsed_time = end - start
    print(f"nx Dijkstra - Elapsed time: {elapsed_time:6.2f} s")

elif lib == "SK":

    # scikit-network
    # ==============

    from sknetwork.path import get_distances

    edges_df = pd.read_parquet(network_file_path)
    edges_df.rename(
        columns={"id_from": "source", "id_to": "target", "tt": "weight"}, inplace=True
    )
    vertex_count = edges_df[["source", "target"]].max().max() + 1
    print(f"{len(edges_df)} edges and {vertex_count} vertices")

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
        sources=idx_from,
        method="D",
        return_predecessors=False,
        unweighted=False,
        n_jobs=-1,
    )
    end = perf_counter()
    elapsed_time = end - start
    print(f"SKN Dijkstra - Elapsed time: {elapsed_time:6.2f} s")
