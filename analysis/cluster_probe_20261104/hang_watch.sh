#!/bin/bash
# Hang watchdog (2026-11-05, after libero_seed2_arv2 hung for 20+ min behind a Ray OOM kill): every 10 min re-fetch the
# AOSS mirrors of the given cluster jobs (object storage only, no ACP API calls) and print a line when a job's console
# has not grown for >= 50 min (the piRL actor phase alone is ~35 min without console output) or when a new Ray OOM kill appears. Usage: hang_watch.sh TAG [TAG...]  (output -> log)
cd "$(dirname "$0")"
declare -A last_lines last_change oom_seen
while true; do
    for tag in "$@"; do
        timeout 150 uv run --offline --with boto3 python fetch_run.py "$tag" >/dev/null 2>&1 || true
        f=cluster_runs/$tag/console.log
        [ -f "$f" ] || continue
        n=$(wc -l < "$f"); now=$(date +%s)
        if [ "${last_lines[$tag]:-}" != "$n" ]; then last_lines[$tag]=$n; last_change[$tag]=$now; fi
        idle=$(( (now - ${last_change[$tag]:-$now}) / 60 ))
        if [ "$idle" -ge 50 ]; then
            mem=$(tar -xzOf "cluster_runs/$tag/tensorboard.tgz" mem.log 2>/dev/null | tail -n 1 | cut -c1-120)
            echo "$(date +%H:%M) HANG? $tag console unchanged for ${idle} min (lines $n) | $mem"
        fi
        k=$(grep -a -c -E "killed due to memory pressure|killed due to the node running low on memory" "$f")
        if [ "$k" != "${oom_seen[$tag]:-0}" ]; then
            oom_seen[$tag]=$k
            [ "$k" -gt 0 ] && echo "$(date +%H:%M) OOM-KILL $tag ($k lines in console)"
        fi
    done
    sleep 600
done
