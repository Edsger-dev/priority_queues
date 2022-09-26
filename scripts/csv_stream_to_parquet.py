import pandas as pd

import pyarrow as pa
import pyarrow.csv
import pyarrow.parquet as pq

# csv_file_path = "/home/francois/Data/Disk_1/OSMR/osm-bawu/osm-bawu.gr"
# parquet_file_path = "/home/francois/Data/Disk_1/OSMR/osm-bawu/osm-bawu.gr.parquet"
# csv_file_path = "/home/francois/Data/Disk_1/OSMR/osm-ger/osm-ger.gr"
# parquet_file_path = "/home/francois/Data/Disk_1/OSMR/osm-ger/osm-ger.gr.parquet"
csv_file_path = "/home/francois/Data/Disk_1/OSMR/osm-eur/osm-eur.gr"
parquet_file_path = "/home/francois/Data/Disk_1/OSMR/osm-eur/osm-eur.gr.parquet"

# read_options = pyarrow.csv.ReadOptions(
#     skip_rows=1, column_names=["a", "source", "target", "weight"], block_size=10000000
# )
# parse_options = pyarrow.csv.ParseOptions(delimiter=" ")
# convert_options = pyarrow.csv.ConvertOptions(
#     column_types={
#         "a": pa.string(),
#         "source": pa.int64(),
#         "target": pa.int64(),
#         "weight": pa.float64(),
#     }
# )

# writer = None
# counter = 0
# with pyarrow.csv.open_csv(
#     csv_file_path,
#     read_options=read_options,
#     parse_options=parse_options,
#     convert_options=convert_options,
# ) as reader:
#     for chunk in reader:
#         print(f"chunk #{counter}")
#         counter += 1
#         if chunk is None:
#             break
#         table = pa.Table.from_batches([chunk])
#         table = table.remove_column(0)
#         if writer is None:
#             writer = pq.ParquetWriter(parquet_file_path, table.schema, compression=None)
#         writer.write_table(table)
# writer.close()


# read the header
with open(csv_file_path) as f:
    lines = f.readlines(10000)
header_size = 0
for line in lines:
    header_size += 1
    if line.startswith("p"):
        # we read the edge count from the header
        edge_count = int(line.split(" ")[-1])
    elif line.startswith("a"):
        header_size -= 1
        break
print(f"edge_count : {edge_count}")

# =======================
# ISSUE WITH (Int(bitWidth=64, isSigned=false))
# NOK with Hyper

# (tableau3)  ✘ francois@francois-P64KV7  ~/Data/Disk_1/OSMR  parq osm-bawu/osm-bawu.gr.parquet --schema

#  # Schema
#  <pyarrow._parquet.ParquetSchema object at 0x7f010a9e9100>
# required group field_id=-1 schema {
#   optional int64 field_id=-1 source (Int(bitWidth=64, isSigned=false));
#   optional int64 field_id=-1 target (Int(bitWidth=64, isSigned=false));
#   optional double field_id=-1 weight;
#   optional int64 field_id=-1 __index_level_0__;
# }
schema = pa.schema(
    [
        ("source", pa.int64()),
        ("target", pa.int64()),
        ("weight", pa.float64()),
    ]
)
# =======================

chunksize = 100_000_000
writer = None
counter = 0
for df in pd.read_csv(
    csv_file_path,
    sep=" ",
    names=["a", "source", "target", "weight"],
    usecols=["source", "target", "weight"],
    skiprows=header_size,
    chunksize=chunksize,
):

    print(f"chunk #{counter}")
    counter += 1

    # data preparation and assertions
    assert not df.isna().any().any()  # no missing values
    df.weight = df.weight.astype(float)  # convert to float type
    df.weight *= 0.01  # convert to seconds
    assert df.weight.min() >= 0.0  # make sure travel times are non-negative
    df = (
        df.groupby(["source", "target"], sort=False).min().reset_index()
    )  # remove parallel edges and keep the one with shortest weight
    df = df[df["source"] != df["target"]]  # remove loops
    df[["source", "target"]] -= 1  # switch to 0-based indices

    table = pa.Table.from_pandas(
        df, schema=schema  #  columns=["source", "target", "weight"],
    )
    if writer is None:
        writer = pq.ParquetWriter(parquet_file_path, table.schema, compression=None)

    writer.write_table(table)

writer.close()
