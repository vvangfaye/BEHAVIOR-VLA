#!/bin/bash
# AFS cleanup (2026-11-06 ~11:10, approved by the user): /260010165 hit its 2.2 TB quota and pirl_repro_libero_long_p1b
# crashed writing its step-25 checkpoint. Deletes ONLY intermediate actor dcp_checkpoint dirs (optimizer/resume state) of
# FINISHED RL runs under /mnt/afs/260010165/rlinf_runs_20261104: per checkpoints/ directory the newest COMPLETE dcp (has
# .metadata) is kept, older ones and newer incomplete ones are deleted. Model weights (model_state_dict), logs, eval
# outputs, running jobs (EXCLUDE) and everything outside rlinf_runs_20261104 are never touched. Prints/uploads a manifest.
set -u
LOG=/tmp/afs_cleanup.log
ROOT=/mnt/afs/260010165/rlinf_runs_20261104
EXCLUDE="robocasa_progress_ppo libero_seed2_mgr_agent libero_seed2_mgr_rule_random afs_cleanup_20261106"
{
echo "== $(date) start"
python3 - "$ROOT" $EXCLUDE <<'PY'
import json
import os
import re
import shutil
import sys

root = sys.argv[1]
exclude = set(sys.argv[2:])


def du(p):
    t = 0
    for dp, _dn, fn in os.walk(p):
        for f in fn:
            try:
                t += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return t


tags = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
before = {t: du(os.path.join(root, t)) for t in tags}
print("per-run size before (GB):", json.dumps({t: round(v / 1e9, 1) for t, v in before.items()}), flush=True)
print("total rlinf_runs_20261104 before: %.1f GB" % (sum(before.values()) / 1e9), flush=True)
pat = re.compile(r"^(.*/checkpoints)/global_step_(\d+)/actor/dcp_checkpoint$")
groups = {}
for t in tags:
    if t in exclude:
        continue
    for dp, dn, _fn in os.walk(os.path.join(root, t)):
        if dp.endswith("/actor") and "dcp_checkpoint" in dn:
            p = os.path.join(dp, "dcp_checkpoint")
            m = pat.match(p)
            if m:
                groups.setdefault(m.group(1), []).append((int(m.group(2)), p))
freed, deleted = 0, []
for ck, lst in sorted(groups.items()):
    lst.sort()
    complete = [(s, p) for s, p in lst if os.path.exists(os.path.join(p, ".metadata"))]
    keep = complete[-1][1] if complete else None
    for _s, p in lst:
        if p == keep:
            continue
        assert p.startswith(root + "/") and p.endswith("/actor/dcp_checkpoint")
        assert not any(p.startswith(os.path.join(root, e) + "/") for e in exclude)
        sz = du(p)
        shutil.rmtree(p)
        freed += sz
        deleted.append((p, round(sz / 1e9, 2)))
        print("deleted %.2f GB %s" % (sz / 1e9, p), flush=True)
    print("kept:", keep, flush=True)
print("deleted %d dcp dirs, freed %.1f GB" % (len(deleted), freed / 1e9), flush=True)
json.dump({"deleted": deleted, "freed_gb": round(freed / 1e9, 1)}, open("/tmp/afs_cleanup_manifest.json", "w"))
PY
echo "== $(date) done"
} > $LOG 2>&1
cat /tmp/afs_cleanup_manifest.json >> $LOG 2>/dev/null
curl -s -m 120 -X PUT --upload-file $LOG "$B1K_PUT_LOG" >/dev/null
cat $LOG
