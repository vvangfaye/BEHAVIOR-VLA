"""Mac-only, read-only OSS footer inspection; request JSON arrives on stdin."""
from concurrent.futures import ThreadPoolExecutor
import configparser
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import threading

import oss2
import pyarrow.parquet as pq


def main():
    request = json.load(sys.stdin)
    if request["source_bucket"] != "traffic-model":
        raise ValueError("Unexpected source bucket")
    config = configparser.ConfigParser()
    config.read(Path.home() / ".ossutilconfig")
    credentials = config["Credentials"]
    endpoint = credentials["endpoint"]
    if not endpoint.startswith("https://"):
        endpoint = "https://" + endpoint.removeprefix("http://")
    lock, transferred = threading.Lock(), [0]
    limit = 64 * 1024**2
    count = [0]

    def inspect(item):
        source = item["source"]
        bucket = oss2.Bucket(oss2.Auth(credentials["accesskeyid"], credentials["accesskeysecret"]),
                             endpoint, request["source_bucket"])
        def read(start, size):
            with lock:
                if transferred[0] + size > limit:
                    raise ValueError("Footer budget exceeded")
                transferred[0] += size
            response = bucket.get_object(source["key"], byte_range=(start, start + size - 1),
                                         headers={"If-Match": '"' + source["etag"] + '"'})
            expected = f"bytes {start}-{start+size-1}/{source['bytes']}"
            if response.headers.get("Content-Range") != expected:
                raise ValueError("Unbounded or changed object response")
            data = response.read(size + 1)
            if len(data) != size:
                raise ValueError("Incorrect footer range length")
            return data
        try:
            tail = read(source["bytes"] - 8, 8)
            if tail[4:] != b"PAR1":
                raise ValueError("Not a Parquet object")
            size = struct.unpack("<I", tail[:4])[0]
            if not 0 < size < 1024**2:
                raise ValueError("Unexpected footer size")
            footer = read(source["bytes"] - 8 - size, size)
            metadata = pq.read_metadata(io.BytesIO(b"PAR1" + footer + tail))
            selected, columns = [], {}
            for i in range(metadata.num_row_groups):
                group = metadata.row_group(i)
                index = next((group.column(j) for j in range(group.num_columns)
                              if group.column(j).path_in_schema == "index"), None)
                statistics = index.statistics if index is not None else None
                if statistics is not None and statistics.has_min_max:
                    if not any(window["dataset_index_start"] <= statistics.max and
                               window["dataset_index_end_exclusive"] > statistics.min
                               for window in item["windows"]):
                        continue
                selected.append(i)
                for j in range(group.num_columns):
                    column = group.column(j)
                    if any(column.path_in_schema == name or column.path_in_schema.startswith(name + ".")
                           for name in request["required_columns"]):
                        columns[column.path_in_schema] = columns.get(column.path_in_schema, 0) + column.total_compressed_size
            if not selected:
                raise ValueError("No row group overlaps requested dataset indices")
            result = {"relative_path": source["relative_path"], "source_etag": source["etag"],
                      "source_bytes": source["bytes"], "footer_sha256": hashlib.sha256(footer).hexdigest(),
                      "footer_bytes": size, "rows": metadata.num_rows,
                      "row_group_count": metadata.num_row_groups, "selected_row_groups": selected,
                      "projected_compressed_bytes": columns, "candidate_windows": len(item["windows"])}
        except Exception as exc:
            result = {"relative_path": source["relative_path"], "error_type": type(exc).__name__,
                      "error_code": getattr(exc, "code", None)}
        with lock:
            count[0] += 1
            if count[0] % 50 == 0:
                print(json.dumps({"footers_inspected": count[0]}), file=sys.stderr, flush=True)
        return result

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(inspect, request["objects"]))
    failures = [r for r in results if "error_type" in r]
    json.dump({"complete": not failures, "object_count": len(results), "footer_range_bytes": transferred[0],
               "projected_column_bytes": sum(sum(r.get("projected_compressed_bytes", {}).values()) for r in results),
               "raw_columns_downloaded": False, "failures": failures, "objects": results}, sys.stdout)
    print()


if __name__ == "__main__":
    main()
