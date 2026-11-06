#!/usr/bin/env python3
"""Directory archive to AOSS, phase 1 = copy + full verification, NO deletion (2026-11-06; user: move behavior_1k to AOSS,
"可以直接用我的凭据"). Runs inside a cluster job with AFS mounted; credentials only in the environment.
  1. inventory: every entry under SRC - regular files (rel, size, mtime_ns, ino, nlink, sha256), symlinks (target),
     empty dirs, special files (skipped, listed);
  2. ads-cli sync SRC -> s3://<bucket>.<host>/<PREFIX>/data/ (no delete flags);
  3. verification: list the prefix; every regular file must exist with the same size and its object, read back in full
     over HTTPS (SigV4 GET), must have the same SHA256 as the source; the source identity is re-checked afterwards;
  4. only if everything matches: PUT manifest.json.gz, VERIFIED.json, COMPLETE.json (binds both SHA256s), GET COMPLETE back.
Release (deleting the AFS source) is a separate, later step. Usage: dir_archive_phase1.py SRC PREFIX [--threads 32]"""
import concurrent.futures as cf
import gzip
import hashlib
import json
import os
import stat
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aoss_sigv4 as S  # noqa: E402


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_file(p, bs=8 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    src, prefix = sys.argv[1].rstrip("/"), sys.argv[2].strip("/")
    threads = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 32
    t0 = time.time()
    files, links, empty_dirs, special = [], [], [], []
    for dp, dn, fn in os.walk(src):
        rel_dp = os.path.relpath(dp, src)
        if not dn and not fn:
            empty_dirs.append(rel_dp)
        for d in list(dn):
            p = os.path.join(dp, d)
            if os.path.islink(p):
                links.append({"rel": os.path.relpath(p, src), "target": os.readlink(p)})
                dn.remove(d)
        for f in fn:
            p = os.path.join(dp, f)
            st = os.lstat(p)
            rel = os.path.relpath(p, src)
            if stat.S_ISLNK(st.st_mode):
                links.append({"rel": rel, "target": os.readlink(p)})
            elif stat.S_ISREG(st.st_mode):
                files.append({"rel": rel, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "ino": st.st_ino,
                              "dev": st.st_dev, "nlink": st.st_nlink})
            else:
                special.append(rel)
    print("[%s] inventory: %d files %.1f GB, %d symlinks, %d empty dirs, %d special; walk %.0f s" % (
        now(), len(files), sum(x["size"] for x in files) / 1e9, len(links), len(empty_dirs), len(special), time.time() - t0), flush=True)
    t1 = time.time()
    with cf.ThreadPoolExecutor(threads) as ex:
        for x, h in zip(files, ex.map(lambda x: sha_file(os.path.join(src, x["rel"])), files)):
            x["sha256"] = h
    print("[%s] hashed in %.0f s" % (now(), time.time() - t1), flush=True)

    # 2) ads-cli sync (credentials from the environment; no delete flags)
    t2 = time.time()
    dst = "s3://%s.%s/%s/data/" % (S.BUCKET, S.HOST, prefix)
    r = subprocess.run(["/tmp/ads-cli", "--https", "--threads", "64", "--listers", "16", "sync", src + "/", dst],
                       capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-3:])
    for secret in (os.environ.get("AWS_ACCESS_KEY_ID", "?"), os.environ.get("AWS_SECRET_ACCESS_KEY", "?")):
        tail = tail.replace(secret, "<redacted>")
    print("[%s] ads-cli sync rc=%d in %.0f s: %s" % (now(), r.returncode, time.time() - t2, tail), flush=True)
    if r.returncode != 0:
        raise SystemExit("ads-cli failed; nothing is marked complete")

    # 3) verification: presence + size from the listing, full read-back SHA256 of every object
    t3 = time.time()
    listed = dict(S.list_keys(prefix + "/data/"))
    missing = [x["rel"] for x in files if listed.get(prefix + "/data/" + x["rel"]) != x["size"]]
    print("[%s] listed %d objects; missing or wrong size: %d" % (now(), len(listed), len(missing)), flush=True)
    bad = []

    def check(x):
        h, n = S.get_sha256(prefix + "/data/" + x["rel"])
        return x["rel"] if (h != x["sha256"] or n != x["size"]) else None

    with cf.ThreadPoolExecutor(threads) as ex:
        for res in ex.map(check, [x for x in files if x["rel"] not in set(missing)]):
            if res:
                bad.append(res)
    changed = []
    for x in files:
        st = os.lstat(os.path.join(src, x["rel"]))
        if (st.st_size, st.st_mtime_ns, st.st_ino) != (x["size"], x["mtime_ns"], x["ino"]):
            changed.append(x["rel"])
    print("[%s] read-back done in %.0f s: sha/size mismatches %d, source changed during archive %d" % (
        now(), time.time() - t3, len(bad), len(changed)), flush=True)
    if missing or bad or changed:
        print("NOT COMPLETE: missing %s bad %s changed %s" % (missing[:5], bad[:5], changed[:5]), flush=True)
        raise SystemExit(2)

    # 4) manifest, VERIFIED, COMPLETE
    man = {"source": src, "prefix": prefix + "/data/", "created_utc": now(), "files": files, "symlinks": links,
           "empty_dirs": empty_dirs, "special_skipped": special}
    mbody = gzip.compress(json.dumps(man).encode())
    S.put_bytes(prefix + "/manifest.json.gz", mbody)
    ver = {"source": src, "utc": now(), "files": len(files), "bytes": sum(x["size"] for x in files),
           "method": "SigV4 HTTPS GET full read-back of every object, SHA256 vs source", "mismatches": 0, "missing": 0,
           "source_changed": 0, "manifest_sha256": hashlib.sha256(mbody).hexdigest()}
    vbody = json.dumps(ver).encode()
    S.put_bytes(prefix + "/VERIFIED.json", vbody)
    cbody = json.dumps({"source": src, "utc": now(), "manifest_sha256": ver["manifest_sha256"],
                        "verified_sha256": hashlib.sha256(vbody).hexdigest(), "release": "pending user confirmation"}).encode()
    S.put_bytes(prefix + "/COMPLETE.json", cbody)
    assert S.get_bytes(prefix + "/COMPLETE.json") == cbody and S.get_bytes(prefix + "/manifest.json.gz") == mbody
    print("[%s] COMPLETE: %d files %.1f GB verified; total %.0f min" % (now(), len(files), ver["bytes"] / 1e9, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
