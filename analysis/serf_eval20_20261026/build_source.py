from pathlib import Path
import ast
import gzip
import hashlib
import io
import json
import tarfile

HERE=Path(__file__).resolve().parent
base=HERE/"runtime"
manifest={}
with (HERE/"source.tar.gz").open("wb") as raw, gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as gz,tarfile.open(fileobj=gz,mode="w") as tar:
    for f in sorted(base.rglob("*")):
        if not f.is_file() or "__pycache__" in f.parts:continue
        b=f.read_bytes();name=str(f.relative_to(base))
        if f.suffix==".py":ast.parse(b,filename=name)
        manifest[name]={"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b)}
        entry=tarfile.TarInfo(name);entry.size=len(b);entry.mode=0o644;entry.mtime=0
        tar.addfile(entry,io.BytesIO(b))
(HERE/"source_manifest.json").write_text(json.dumps(manifest,indent=2))
receipt={"source_sha256":hashlib.sha256((HERE/"source.tar.gz").read_bytes()).hexdigest(),
         "bytes":(HERE/"source.tar.gz").stat().st_size,"files":len(manifest)}
(HERE/"source_receipt.json").write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
