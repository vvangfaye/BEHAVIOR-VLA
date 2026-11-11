#!/usr/bin/env python3
"""AFS -> AOSS cold archive for finished RL runs (2026-11-06; user: "把一些数据放在 aoss 上"), single-job design.

Cluster jobs have no AOSS credentials, so the job only uses presigned URLs created locally ahead of time:
  plan (local, credentials): per unit (= run tag) a pool of object SLOTS {prefix}/units/<unit>/blob/<i> sized from an
       upper bound of the unit's bytes, plus manifest/VERIFIED/COMPLETE/RELEASED objects; PUT+GET URLs for all of them go
       into a private plan object {prefix}/plan_with_urls.json (never logged); a redacted copy is kept locally.
  job  (in the cluster, AFS mounted, nothing written to AFS): per unit
       1. enumerate regular files under .../checkpoints/global_step_*/** (nlink > 1 -> protected, kept; others skipped);
       2. per file: record (dev, ino, nlink, size, mtime_ns, blocks); read sequentially in <= part_size parts, hash the
          whole file and each part, PUT each part into the next slot (files in parallel);
       3. re-check every file's identity (unchanged); full read-back over HTTPS GET of every slot: part SHA256 and
          whole-file SHA256 must equal the source hashes;
       4. PUT manifest (files, identities, SHA256, part->slot keys), VERIFIED (read-back report) and COMPLETE (binds
          manifest SHA + VERIFIED SHA), GET COMPLETE back;
       5. release: identity re-check, unlink exactly the archived files, remove emptied checkpoint dirs, PUT RELEASED.
       Any failure leaves the unit's source untouched. States: inventoried -> uploaded -> verified -> complete -> source_released.
  restore (local, credentials): COMPLETE -> VERIFIED/manifest SHA check -> download slots -> SHA256 -> NEW directory.
Logs, tensorboard and everything outside checkpoints/global_step_* stay on AFS. AFS quota space only returns after a
directory-quota "校验" in the console (AFS soft delete)."""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request

ROOT = "/mnt/afs/260010165/rlinf_runs_20261104"
CK = re.compile(r"(^|/)checkpoints/global_step_\d+(/|$)")
IDKEYS = ("dev", "ino", "size", "mtime_ns")


def ident(st):
    return {"dev": st.st_dev, "ino": st.st_ino, "nlink": st.st_nlink, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "blocks": st.st_blocks}


def same(a, b):
    return all(a[k] == b[k] for k in IDKEYS)


def _req(url, data=None, method="GET", timeout=900, tries=4):
    for i in range(tries):
        try:
            r = urllib.request.Request(url, data=data, method=method)
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return resp.read() if method == "GET" else resp.status
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print("  retry %s after %s" % (method, type(e).__name__), flush=True)
            time.sleep(10 * (i + 1))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------ plan (local, credentials)
def plan(units_bytes, archive_id, part_size, out, bucket_prefix="afs_archive_20261106", extra_slots=24, external=False):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import aoss_io

    s3 = aoss_io.local()
    if external:  # local end-to-end test: presign against the external endpoint (cluster jobs use the internal one)
        def pres(m, k, seconds=3600):
            return s3.generate_presigned_url(m, Params={"Bucket": aoss_io.BUCKET, "Key": k}, ExpiresIn=seconds)
    else:
        def pres(m, k, seconds=3 * 86400):
            return aoss_io.presign(m, k, seconds=seconds)
    pre = "%s/%s" % (bucket_prefix, archive_id)
    units = {}
    for u, nbytes in units_bytes.items():
        n = int(math.ceil(nbytes / part_size)) + extra_slots
        units[u] = {"slots": [{"key": "%s/units/%s/blob/%05d" % (pre, u, i)} for i in range(n)],
                    "objects": {k: {"key": "%s/units/%s/%s" % (pre, u, k)}
                                for k in ("manifest.json", "VERIFIED.json", "COMPLETE.json", "RELEASED.json")}}
        for o in units[u]["slots"] + list(units[u]["objects"].values()):
            o["put"], o["get"] = pres("put_object", o["key"]), pres("get_object", o["key"])
    pl = {"archive_id": archive_id, "prefix": pre, "bucket": aoss_io.BUCKET, "root": ROOT, "part_size": part_size,
          "created_utc": now(), "units": units}
    s3.put_object(Bucket=aoss_io.BUCKET, Key="%s/plan_with_urls.json" % pre, Body=json.dumps(pl).encode())
    red = json.loads(json.dumps(pl))
    for u in red["units"].values():
        for o in u["slots"] + list(u["objects"].values()):
            o.pop("put"), o.pop("get")
    with open(out, "w") as f:
        json.dump(red, f, indent=1)
    print("plan: %d units, %d slots; plan object %s/plan_with_urls.json" % (len(units), sum(len(u["slots"]) for u in units.values()), pre))
    return "%s/plan_with_urls.json" % pre


# ------------------------------------------------------------------ job
def archive_unit(root, unit, up, part_size, release, workers):
    base = os.path.join(root, unit)
    files, protected, skipped = [], [], []
    for dp, dn, fn in os.walk(base):
        dn.sort()
        if not CK.search(os.path.relpath(dp, base) + "/"):
            continue
        for f in sorted(fn):
            p = os.path.join(dp, f)
            st = os.lstat(p)
            rel = os.path.relpath(p, base)
            if not os.path.isfile(p) or os.path.islink(p):
                skipped.append(rel)
            elif st.st_nlink > 1:
                protected.append(rel)
            else:
                files.append({"rel": rel, "id": ident(st)})
    # assign slots
    k = 0
    for x in files:
        x["parts"] = []
        for off in range(0, max(x["id"]["size"], 1), part_size):
            if k >= len(up["slots"]):
                raise RuntimeError("not enough slots (%d)" % len(up["slots"]))
            x["parts"].append({"slot": k, "key": up["slots"][k]["key"], "offset": off,
                               "length": min(part_size, x["id"]["size"] - off)})
            k += 1

    def push(x):  # sequential read of one file: whole-file + part hashes, PUT each part
        whole = hashlib.sha256()
        with open(os.path.join(base, x["rel"]), "rb") as f:
            for part in x["parts"]:
                f.seek(part["offset"])
                data = f.read(part["length"])
                if len(data) != part["length"]:
                    raise RuntimeError("short read %s" % x["rel"])
                whole.update(data)
                part["sha256"] = hashlib.sha256(data).hexdigest()
                _req(up["slots"][part["slot"]]["put"], data=data, method="PUT")
        x["sha256"] = whole.hexdigest()
        return x["rel"]

    st = {"unit": unit, "files": len(files), "bytes": sum(x["id"]["size"] for x in files), "protected": protected,
          "skipped": skipped, "slots_used": k, "state": "inventoried"}
    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(push, files))
    for x in files:
        if not same(ident(os.lstat(os.path.join(base, x["rel"]))), x["id"]):
            raise RuntimeError("source changed during upload: %s" % x["rel"])
    st["state"] = "uploaded"
    rep = []
    for x in files:  # full read-back of every part from the object store
        whole = hashlib.sha256()
        for part in x["parts"]:
            data = _req(up["slots"][part["slot"]]["get"])
            h = hashlib.sha256(data).hexdigest()
            if len(data) != part["length"] or h != part["sha256"]:
                raise RuntimeError("read-back mismatch %s" % part["key"])
            whole.update(data)
            rep.append({"key": part["key"], "length": len(data), "sha256": h})
        if whole.hexdigest() != x["sha256"]:
            raise RuntimeError("whole-file read-back SHA256 mismatch %s" % x["rel"])
    man = {"unit": unit, "root": root, "created_utc": now(), "part_size": part_size, "files": files,
           "protected_kept": protected, "skipped_kept": skipped}
    mbody = json.dumps(man).encode()
    _req(up["objects"]["manifest.json"]["put"], data=mbody, method="PUT")
    vbody = json.dumps({"unit": unit, "utc": now(), "method": "presigned HTTPS GET, full read-back of every part",
                        "manifest_sha256": hashlib.sha256(mbody).hexdigest(), "files": len(files), "parts": len(rep),
                        "bytes": st["bytes"], "errors": 0, "parts_detail": rep}).encode()
    _req(up["objects"]["VERIFIED.json"]["put"], data=vbody, method="PUT")
    st["state"] = "verified"
    cbody = json.dumps({"unit": unit, "utc": now(), "manifest_sha256": hashlib.sha256(mbody).hexdigest(),
                        "verified_sha256": hashlib.sha256(vbody).hexdigest()}).encode()
    _req(up["objects"]["COMPLETE.json"]["put"], data=cbody, method="PUT")
    if _req(up["objects"]["COMPLETE.json"]["get"]) != cbody or _req(up["objects"]["manifest.json"]["get"]) != mbody:
        raise RuntimeError("COMPLETE/manifest read-back mismatch")
    st["state"] = "complete"
    if release:
        freed, res = 0, []
        for x in files:
            p = os.path.join(base, x["rel"])
            cur = ident(os.lstat(p))
            if not same(cur, x["id"]) or cur["nlink"] != 1:
                res.append([x["rel"], "kept_identity_changed"])
                continue
            os.unlink(p)
            freed += x["id"]["size"]
            res.append([x["rel"], "unlinked"])
        for dp, _dn, _fn in sorted(os.walk(base), key=lambda t: -len(t[0])):
            if CK.search(os.path.relpath(dp, base) + "/") and not os.listdir(dp):
                os.rmdir(dp)
        _req(up["objects"]["RELEASED.json"]["put"], data=json.dumps({"unit": unit, "utc": now(), "unlinked_bytes": freed,
                                                                      "results": res}).encode(), method="PUT")
        st["state"], st["unlinked_gb"] = "source_released", round(freed / 1e9, 2)
    return st


def job(plan_url, release=True, workers=4, root_override=""):
    pl = json.loads(_req(plan_url).decode())
    root = root_override or pl["root"]
    summary = {}
    for unit, up in pl["units"].items():
        t0 = time.time()
        if not os.path.isdir(os.path.join(root, unit)):
            summary[unit] = {"unit": unit, "state": "missing"}
            continue
        try:
            st = archive_unit(root, unit, up, pl["part_size"], release, workers)
        except Exception as e:  # noqa: BLE001  (the unit keeps its source)
            st = {"unit": unit, "state": "failed_source_kept", "error": "%s: %s" % (type(e).__name__, str(e)[:300])}
        st["minutes"] = round((time.time() - t0) / 60, 1)
        summary[unit] = st
        print("unit " + json.dumps(st), flush=True)
    print("SUMMARY " + json.dumps({u: [s.get("state"), s.get("unlinked_gb"), s.get("bytes")] for u, s in summary.items()}), flush=True)
    return summary


# ------------------------------------------------------------------ restore (local, credentials)
def restore(prefix, unit, dest):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import aoss_io

    s3 = aoss_io.local()

    def get(k):
        return s3.get_object(Bucket=aoss_io.BUCKET, Key="%s/units/%s/%s" % (prefix, unit, k))["Body"].read()

    comp = json.loads(get("COMPLETE.json"))
    mbody, vbody = get("manifest.json"), get("VERIFIED.json")
    assert hashlib.sha256(mbody).hexdigest() == comp["manifest_sha256"], "manifest does not match COMPLETE"
    assert hashlib.sha256(vbody).hexdigest() == comp["verified_sha256"], "VERIFIED does not match COMPLETE"
    man = json.loads(mbody)
    assert not os.path.exists(dest) or not os.listdir(dest), "restore into a NEW empty directory"
    for x in man["files"]:
        out = os.path.join(dest, x["rel"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        whole = hashlib.sha256()
        with open(out, "wb") as w:
            for part in x["parts"]:
                body = s3.get_object(Bucket=aoss_io.BUCKET, Key=part["key"])["Body"].read()
                assert hashlib.sha256(body).hexdigest() == part["sha256"]
                whole.update(body)
                w.write(body)
        assert whole.hexdigest() == x["sha256"], "restored SHA mismatch %s" % x["rel"]
    print("restored %d files of %s into %s" % (len(man["files"]), unit, dest))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    b = sp.add_parser("plan")
    b.add_argument("--units-bytes", required=True, help="JSON {unit: upper bound of checkpoint bytes}")
    b.add_argument("--archive-id", required=True)
    b.add_argument("--part-size", type=int, default=1 << 30)
    b.add_argument("--out", required=True)
    b.add_argument("--bucket-prefix", default="afs_archive_20261106")
    b.add_argument("--external", action="store_true")
    c = sp.add_parser("job")
    c.add_argument("--plan-url-env", default="B1K_ARCHIVE_PLAN_URL")
    c.add_argument("--no-release", action="store_true")
    c.add_argument("--root-override", default="")
    d = sp.add_parser("restore")
    d.add_argument("--prefix", required=True)
    d.add_argument("--unit", required=True)
    d.add_argument("--dest", required=True)
    x = ap.parse_args()
    if x.cmd == "plan":
        print(plan(json.loads(x.units_bytes), x.archive_id, x.part_size, x.out, x.bucket_prefix, external=x.external))
    elif x.cmd == "job":
        job(os.environ[x.plan_url_env], release=not x.no_release, root_override=x.root_override)
    elif x.cmd == "restore":
        restore(x.prefix, x.unit, x.dest)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
