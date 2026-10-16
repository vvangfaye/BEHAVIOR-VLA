"""Validate the full archived environment using OSS's published CRC64."""
import hashlib
import json
from pathlib import Path
import oss2

ROOT = Path(__file__).resolve().parent
source = json.loads((ROOT / "EVAL_ARCHIVE_SOURCE.private.json").read_text())
archive = ROOT / "runtime/downloads/b1k-eval.tar.gz"
sha = hashlib.sha256()
crc = oss2.utils.Crc64()
with archive.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        sha.update(block)
        crc.update(block)
if archive.stat().st_size != source["bytes"] or str(crc.crc) != source["crc64"]:
    raise RuntimeError("Environment archive does not match the source object's size and CRC64")
result = {"sha256": sha.hexdigest(), "crc64": str(crc.crc), "etag": source["etag"],
          "bytes": archive.stat().st_size, "matches_source_object": True,
          "mtime_ns": archive.stat().st_mtime_ns}
(ROOT / "EVAL_ARCHIVE_VERIFIED.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
