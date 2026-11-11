#!/bin/bash
# Poll cluster RLinf runs (tags as arguments) every 5 min; print one line per run whenever its summary changes.
# Summary: ACP job state, mirrored bootstrap state, last Global Step, rollout-epoch progress, eval counters, error count.
cd "$(dirname "$0")"
declare -A last
while true; do
    for tag in "$@"; do
        N=$(python3 -c "import json;print(json.load(open('runs/$tag.json'))['name'])" 2>/dev/null) || continue
        st=$(timeout 40 /home/faye/.local/bin/sco acp jobs describe "$N" --workspace-name=share-space -o json 2>/dev/null \
             | python3 -c "import json,sys; d=json.load(sys.stdin); s=d.get('state'); print(s.get('state') if isinstance(s,dict) else s)" 2>/dev/null || echo "?")
        timeout 150 uv run --offline --with boto3 python fetch_run.py "$tag" >/dev/null 2>&1 || true
        D=cluster_runs/$tag
        ms=$(python3 -c "import json;print(json.load(open('$D/status.json'))['state'])" 2>/dev/null || echo "-")
        gs=$(/mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/python tb_last.py "$D" 2>/dev/null | awk '{if ($1 > 0) {printf "it=%d succ%%:", $1; for (i = 2; i <= NF; i++) printf " %s", $i}}')
        ro=$(tr '\r' '\n' < "$D/console.log" 2>/dev/null | grep -a "Rollout Epochs" | tail -1 | grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<" | head -1)
        lib=$(grep -aoE "\[libero eval\] task_id=[0-9]+, trial_id=[0-9]+, success=[A-Za-z]+" "$D/console.log" 2>/dev/null | sort -u | awk '{n++; if($0~/True/) s++} END {if(n) printf "libero %d/%d", s, n}')
        cal=$(grep -ac "\[calvin eval\]" "$D/console.log" 2>/dev/null)
        ne=$(grep -a -cE "Traceback|killed due to memory|OOM kill|Error:" "$D/console.log" 2>/dev/null)
        line="$st/$ms | ${gs:-gs-} | ro=${ro:--} | ${lib:-} cal=${cal:-0} | errs=${ne:-0}"
        if [ "${last[$tag]:-}" != "$line" ]; then
            echo "$(date +%H:%M) $tag $line"
            last[$tag]="$line"
        fi
    done
    sleep 300
done
