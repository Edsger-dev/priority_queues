import os
import sys
from argparse import ArgumentParser
from time import perf_counter

import graph_tool as gt
import networkit as nk
import numpy as np
import pandas as pd
from graph_tool import topology
from igraph import Graph
from loguru import logger
from priority_queues.shortest_path import ShortestPath, convert_sorted_graph_to_csr
from scipy.sparse import coo_array
from scipy.sparse.csgraph import dijkstra
from sknetwork.path import get_distances

logger.remove()
fmt = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |"
    + " <level>{message}</level>"
)
logger.add(sys.stderr, format=fmt)

parser = ArgumentParser(description="Command line interface to perf_02.py")
parser.add_argument(
    "-d",
    "--data_dir",
    dest="data_dir",
    help="data dir path",
    metavar="TXT",
    type=str,
    required=False,
    default="/home/francois/Data/Disk_1/",
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
    "-r",
    "--repeat",
    dest="repeat",
    help="repeat",
    metavar="INT",
    type=int,
    required=False,
    default=4,
)

args = parser.parse_args()
data_dir = args.data_dir
idx_from = args.idx_from
repeat = args.repeat

# list of regions
regions = [
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

results = []


for reg in regions:

    logger.info(f"region : {reg}")

    # locate the parquet file
    network_file_path = os.path.join(
        data_dir, f"DIMACS_road_networks/{reg}/USA-road-t.{reg}.gr.parquet"
    )
    logger.info(f"network file path : {network_file_path}")

    # load the network into a Pandas dataframe
    edges_df = pd.read_parquet(network_file_path)
    edges_df.rename(
        columns={"id_from": "source", "id_to": "target", "tt": "weight"}, inplace=True
    )
    vertex_count = edges_df[["source", "target"]].max().max() + 1
    logger.info(f"{len(edges_df)} edges and {vertex_count} vertices")

    # SciPy
    # =====

    logger.info("SciPy Prepare the data")

    data = edges_df["weight"].values
    row = edges_df["source"].values
    col = edges_df["target"].values
    graph_coo = coo_array((data, (row, col)), shape=(vertex_count, vertex_count))
    graph_csr = graph_coo.tocsr()

    for i in range(repeat):

        d = {}

        start = perf_counter()

        dist_matrix = dijkstra(
            csgraph=graph_csr,
            directed=True,
            indices=idx_from,
            return_predecessors=False,
        )

        end = perf_counter()
        elapsed_time = end - start
        logger.info(
            f"SciPy Dijkstra {i+1}/{repeat} - Elapsed time: {elapsed_time:8.4f} s"
        )

        d = {
            "library": "SciPy",
            "network": reg,
            "trial": i,
            "elapsed_time": elapsed_time,
        }
        results.append(d)

    # iGraph
    # ======

    logger.info("iGraph Prepare the data")
    g = Graph.DataFrame(edges_df, directed=True)

    for i in range(repeat):

        d = {}

        start = perf_counter()

        distances = g.distances(
            source=idx_from, target=None, weights="weight", mode="out"
        )
        dist_matrix = np.asarray(distances[0])

        end = perf_counter()
        elapsed_time = end - start
        logger.info(f"iG Dijkstra {i+1}/{repeat} - Elapsed time: {elapsed_time:8.4f} s")

        d = {
            "library": "iGraph",
            "network": reg,
            "trial": i,
            "elapsed_time": elapsed_time,
        }
        results.append(d)

    # Graph-tool
    # ==========

    logger.info("graph-tool Prepare the data")

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

    for i in range(repeat):

        d = {}

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
        logger.info(f"GT Dijkstra {i+1}/{repeat} - Elapsed time: {elapsed_time:8.4f} s")

        d = {
            "library": "graph-tool",
            "network": reg,
            "trial": i,
            "elapsed_time": elapsed_time,
        }
        results.append(d)

    # NetworKit
    # =========

    logger.info("NetworKit Prepare the data")

    nk_file_format = nk.graphio.Format.NetworkitBinary
    networkit_file_path = os.path.join(
        data_dir, f"DIMACS_road_networks/{reg}/USA-road-t.{reg}.gr.NetworkitBinary"
    )

    if os.path.exists(networkit_file_path):

        g = nk.graphio.readGraph(networkit_file_path, nk_file_format)

    else:

        g = nk.Graph(n=vertex_count, weighted=True, directed=True, edgesIndexed=False)

        for row in edges_df.itertuples():
            g.addEdge(row.source, row.target, w=row.weight)

        nk.graphio.writeGraph(g, networkit_file_path, nk_file_format)

    nk_dijkstra = nk.distance.Dijkstra(
        g, idx_from, storePaths=False, storeNodesSortedByDistance=False
    )

    for i in range(repeat):

        d = {}

        start = perf_counter()
        nk_dijkstra.run()

        dist_matrix = nk_dijkstra.getDistances(asarray=True)
        dist_matrix = np.asarray(dist_matrix)
        dist_matrix = np.where(dist_matrix >= 1.79769313e308, np.inf, dist_matrix)

        end = perf_counter()
        elapsed_time = end - start
        logger.info(f"NK Dijkstra {i+1}/{repeat} - Elapsed time: {elapsed_time:8.4f} s")

        d = {
            "library": "NetworKit",
            "network": reg,
            "trial": i,
            "elapsed_time": elapsed_time,
        }
        results.append(d)

    # scikit-network
    # ==============

    logger.info("scikit-network Prepare the data")
    graph_csr = convert_sorted_graph_to_csr(
        edges_df, "source", "target", "weight", vertex_count
    )

    for i in range(repeat):

        d = {}

        start = perf_counter()
        dist_matrix_ref = get_distances(
            adjacency=graph_csr,
            sources=idx_from,
            method="D",
            return_predecessors=False,
            unweighted=False,
            n_jobs=1,
        )
        end = perf_counter()
        elapsed_time = end - start
        logger.info(
            f"SKN Dijkstra {i+1}/{repeat} - Elapsed time: {elapsed_time:8.4f} s"
        )

        d = {
            "library": "scikit-network",
            "network": reg,
            "trial": i,
            "elapsed_time": elapsed_time,
        }
        results.append(d)

    # priority_queues
    # ===============

    logger.info("priority_queues Prepare the data")
    sp = ShortestPath(
        edges_df,
        orientation="one-to-all",
        check_edges=False,
        permute=False,
        heap_type="4ary",
    )

    for i in range(repeat):

        d = {}

        start = perf_counter()
        dist_matrix = sp.run(vertex_idx=idx_from, return_inf=True, return_Series=False)
        end = perf_counter()
        elapsed_time = end - start
        logger.info(f"PQ Dijkstra {i+1}/{repeat} - Elapsed time: {elapsed_time:8.4f} s")

        d = {
            "library": "priority_queues",
            "network": reg,
            "trial": i,
            "elapsed_time": elapsed_time,
        }
        results.append(d)

df = pd.DataFrame.from_records(results)
df.to_csv("results.csv")
