"""

https://i11www.iti.kit.edu/resources/roadgraphs.php
"""

import os


import numpy as np
import pandas as pd

DATA_DIR_ROOT_PATH = "/home/francois/Data/Disk_1/"  # root data dir

data_dir_path = os.path.join(DATA_DIR_ROOT_PATH, "OSMR")
names = [
    "osm-bawu",
    "osm-ger",
    "osm-eur",
]

network_dir_paths = {}
for name in names:
    network_dir_path = os.path.join(data_dir_path, name)
    network_dir_paths[name] = network_dir_path

graph_file_paths = {}
coord_file_paths = {}
for name in names:
    network_dir_path = network_dir_paths[name]
    graph_file_paths[name] = os.path.join(network_dir_path, name + ".gr")
    coord_file_paths[name] = os.path.join(network_dir_path, name + ".co")


def read_travel_time_graph(file_path):

    # read the header
    with open(file_path) as f:
        lines = f.readlines(10_000)
    header_size = 0
    for line in lines:
        header_size += 1
        if line.startswith("p"):
            # we read the edge count from the header
            edge_count = int(line.split(" ")[-1])
        elif line.startswith("a"):
            header_size -= 1
            break

    # read the data
    df = pd.read_csv(
        file_path,
        sep=" ",
        names=["a", "source", "target", "weight"],
        usecols=["source", "target", "weight"],
        skiprows=header_size,
        dtype={"source": np.uintc, "target": np.uintc, "weight": np.uintc},
    )

    assert len(df) == edge_count

    # data preparation and assertions
    assert not df.isna().any().any()  # no missing values
    df.weight = df.weight.astype(float)  # convert to float type
    df.weight *= 0.01  # convert to seconds
    assert df.weight.min() >= 0.0  # make sure travel times are non-negative
    df = (
        df.groupby(by=["source", "target"], sort=False).min().reset_index()
    )  # remove parallel edges and keep the one with shortest weight
    df = df[df["source"] != df["target"]]  # remove loops
    df[["source", "target"]] -= 1  # switch to 0-based indices

    return df


parquet_gr_file_paths = {}
for name in names:

    file_path = graph_file_paths[name]
    parquet_gr_file_path = file_path + ".parquet"
    parquet_gr_file_paths[name] = parquet_gr_file_path
    edges_df = read_travel_time_graph(file_path)
    if not os.path.exists(parquet_gr_file_path):
        edges_df = read_travel_time_graph(file_path)
        edges_df.to_parquet(parquet_gr_file_path)

print("done")
