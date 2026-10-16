"""Bounded, resumable HTTP ranges for large public artifacts."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import time
import urllib.request


def download(url, destination, size, workers=12, headers=None, use_proxy=False):
    destination = Path(destination)
    parts = destination.with_name(destination.name + ".parts")
    parts.mkdir(exist_ok=True)
    chunk_size = 64 * 1024 * 1024
    ranges = [(n, start, min(start + chunk_size, size))
              for n, start in enumerate(range(0, size, chunk_size))]
    # Preserve a prefix previously fetched by a sequential downloader.
    if destination.exists():
        with destination.open("rb") as source:
            for n, start, end in ranges:
                if destination.stat().st_size < end:
                    break
                part = parts / str(n)
                if not part.exists():
                    part.write_bytes(source.read(end - start))
                else:
                    source.seek(end)

    def fetch(info):
        n, start, end = info
        part = parts / str(n)
        for attempt in range(5):
            done = part.stat().st_size if part.exists() else 0
            if done == end - start:
                return
            if done > end - start:
                raise RuntimeError("Oversized partial range")
            req_headers = dict(headers() if callable(headers) else (headers or {}))
            req_headers["Range"] = f"bytes={start + done}-{end - 1}"
            try:
                req = urllib.request.Request(url, headers=req_headers)
                opener = (urllib.request.build_opener() if use_proxy else
                          urllib.request.build_opener(urllib.request.ProxyHandler({})))
                with opener.open(req, timeout=90) as response:
                    content_range = response.headers.get("Content-Range", "")
                    expected = f"bytes {start + done}-{end - 1}/{size}"
                    if response.status != 206 or content_range != expected:
                        raise RuntimeError("Server did not honor the requested bounded range")
                    with part.open("ab") as output:
                        left = end - start - done
                        while left:
                            buf = response.read(min(left, 1024 * 1024))
                            if not buf:
                                raise RuntimeError("Truncated range")
                            output.write(buf)
                            left -= len(buf)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(fetch, ranges), 1):
            if i % 16 == 0 or i == len(ranges):
                print("ranges", destination.name[:24], i, "/", len(ranges), flush=True)
    assembled = destination.with_name(destination.name + ".assembled")
    with assembled.open("wb") as output:
        for n, _, _ in ranges:
            with (parts / str(n)).open("rb") as source:
                shutil.copyfileobj(source, output, 4 * 1024 * 1024)
    if assembled.stat().st_size != size:
        raise RuntimeError("Assembled artifact size mismatch")
    os.replace(assembled, destination)
    shutil.rmtree(parts)
