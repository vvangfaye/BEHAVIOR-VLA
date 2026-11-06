#!/bin/bash
# Low-frequency status watch (submission_policy.md): every 20 min, two `jobs list` calls (RUNNING, STARTING) for the
# whole account instead of per-job describes; logs come from the AOSS mirror (object storage, not the ACP API).
# Prints one line per tag when its summary changes. Usage: [INITIAL_SLEEP=s] monitor_light2.sh [TAG...]
# v2 (11/6 00:0x): tags are re-read every cycle from monitor_tags.txt (one per line; CLI tags if the file is missing),
# and one 'account-queued' line is printed whenever the account's queued (STARTING) jobs change - same two list calls.
# Progress: training success per iteration (x100) from the tensorboard mirror, or from the console mirror when no
# tensorboard has been uploaded yet ("con:" prefix; multi-stage jobs mirror the console of the current stage).
cd "$(dirname "$0")"
PY=/mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/python
declare -A last
sleep "${INITIAL_SLEEP:-0}"   # keeps >= 20 min between list calls when the monitor is restarted
while true; do
    states=$(for st in RUNNING STARTING; do
        timeout 120 /home/faye/.local/bin/sco acp jobs list --workspace-name=share-space --state=$st --page-size=500 -o json 2>/dev/null \
        | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); js=d if isinstance(d,list) else (d.get('training_jobs') or [])
except Exception: js=[]
for j in js:
    n=j.get('display_name','')
    own=(j.get('ownership') or {}).get('user_name')=='260010165'
    if n.startswith('b1k-'): print(n, '$st')
    elif own and '$st'=='STARTING': print(n, 'ACCOUNT_STARTING')"; done)
    q=$(echo "$states" | awk '$2=="STARTING" || $2=="ACCOUNT_STARTING" {print $1}' | sort | tr '\n' ' ')
    qn=$(echo "$states" | awk '$2=="STARTING" || $2=="ACCOUNT_STARTING"' | grep -c .)
    if [ "${lastq:-x}" != "$q" ]; then echo "$(date +%H:%M) account-queued $qn: ${q:-none}"; lastq="$q"; fi
    TAGS=("$@"); [ -f monitor_tags.txt ] && mapfile -t TAGS < <(grep -v '^ *$' monitor_tags.txt)
    for tag in "${TAGS[@]}"; do
        N=$(python3 -c "import json;print(json.load(open('runs/$tag.json'))['name'])" 2>/dev/null) || continue
        js=$(echo "$states" | awk -v n="$N" '$1==n {print $2}' | head -1); js=${js:-ENDED}
        timeout 150 uv run --offline --with boto3 python fetch_run.py "$tag" >/dev/null 2>&1 || true
        D=cluster_runs/$tag
        ms=$(python3 -c "import json;print(json.load(open('$D/status.json'))['state'])" 2>/dev/null || echo "-")
        it=$($PY tb_last.py "$D" 2>/dev/null)
        if [ "${it:-0}" = "0" ] && [ -f "$D/console.log" ]; then
            c=$(grep -a -o "success_once=[0-9.]*" "$D/console.log" | awk -F= '{n++; s = s sprintf(" %.0f", 100 * $2)} END {if (n) printf "con:%d%s", n, s}')
            [ -n "$c" ] && it=$c
        fi
        stg=$(python3 -c "import json; s=json.load(open('$D/tb/stages_status.json')); print('stage', s.get('current'), {k: v.get('rc') for k, v in s['results'].items()})" 2>/dev/null)
        ne=$(grep -a -cE "Traceback|killed due to memory|OOM kill" "$D/console.log" 2>/dev/null)
        line="$js/$ms | it=${it:-0} | ${stg:-} | errs=${ne:-0}"
        if [ "${last[$tag]:-}" != "$line" ]; then echo "$(date +%H:%M) $tag $line"; last[$tag]="$line"; fi
    done
    sleep 1200
done
