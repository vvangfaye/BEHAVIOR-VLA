"""Exercise real PyArrow random reads against the restricted range wrapper."""
import io
import json
import os
import re
import time

import pyarrow as pa
import pyarrow.parquet as pq
from worker import RangeSource


def main():
    table = pa.table({"index": list(range(40)), "action": [[float(i)] * 23 for i in range(40)],
                      "unused": [os.urandom(65536) for _ in range(40)]})
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="NONE", row_group_size=40)
    data, calls = buffer.getvalue(), []
    used = [0]
    def reserve(size):
        used[0] += size
        if used[0] > 256 * 1024:
            raise ValueError("Test byte budget exceeded")
    class Response(io.BytesIO):
        status = 206
        def __init__(self, start, end):
            super().__init__(data[start:end + 1])
            self.headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}
    class Opener:
        def open(self, request, timeout):
            assert request.get_header("If-match") == '"test-etag"'
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
            calls.append((start, end))
            return Response(start, end)
    stream = RangeSource({"bytes": len(data), "get_url": "https://example.invalid/test.parquet", "etag": "test-etag"}, reserve, time.monotonic() + 10)
    stream.opener = Opener()
    parquet = pq.ParquetFile(stream, pre_buffer=False)
    for i in range(parquet.metadata.num_row_groups):
        group = parquet.metadata.row_group(i)
        for j in range(group.num_columns):
            column = group.column(j)
            if column.path_in_schema.startswith("action") or column.path_in_schema == "index":
                start = column.dictionary_page_offset if column.has_dictionary_page else column.data_page_offset
                stream.allowed.append((start, start + column.total_compressed_size))
    actual = parquet.read(columns=["index", "action"], use_threads=False)
    assert actual.equals(table.select(["index", "action"]))
    unused = parquet.metadata.row_group(0).column(2)
    stream.seek(unused.dictionary_page_offset if unused.has_dictionary_page else unused.data_page_offset)
    try:
        stream.read(2 * 1024**2)
    except ValueError as exc:
        assert str(exc) == "Attempt to read unrequested Parquet columns"
    else:
        raise AssertionError("Unrequested column read was allowed")
    print(json.dumps({"pyarrow_projection_matches": True, "unrequested_column_blocked": True,
                      "range_bytes": used[0], "object_bytes": len(data), "range_requests": len(calls)}))


if __name__ == "__main__":
    main()
