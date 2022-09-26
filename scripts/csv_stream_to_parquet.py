import pyarrow as pa
import pyarrow.csv
import pyarrow.parquet as pq

csv_file_path = "/home/francois/Data/Disk_1/OSMR/osm-eur/osm-eur.gr"
parquet_file_path = "/home/francois/Data/Disk_1/OSMR/osm-eur/osm-eur.gr.parquet"

read_options = pyarrow.csv.ReadOptions(
    skip_rows=1, column_names=["a", "source", "target", "weight"], block_size=10000000
)
parse_options = pyarrow.csv.ParseOptions(delimiter=" ")
convert_options = pyarrow.csv.ConvertOptions(
    column_types={
        "a": pa.string(),
        "source": pa.uint32(),
        "target": pa.uint32(),
        "weight": pa.uint64(),
    }
)

writer = None
counter = 0
with pyarrow.csv.open_csv(
    csv_file_path,
    read_options=read_options,
    parse_options=parse_options,
    convert_options=convert_options,
) as reader:
    for chunk in reader:
        print(f"chunk #{i}")
        i += 1
        if chunk is None:
            break
        table = pa.Table.from_batches([chunk])
        table = table.remove_column(0)
        if writer is None:
            writer = pq.ParquetWriter(parquet_file_path, table.schema, compression=None)
        writer.write_table(table)
writer.close()
