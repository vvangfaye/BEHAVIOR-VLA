#!/usr/bin/env bash
# Download and extract the OmniGibson data bundle needed for closed-loop eval.
#
# Usage:
#   ./scripts/setup_eval_assets.sh
#   DATA_PATH=/somewhere/omnigibson-data ./scripts/setup_eval_assets.sh
#
# The zips are cached on NAS so a Pod restart only costs re-extraction, while the
# extracted tree lives on tmpfs because scene loading is read-latency bound.
#
# The encryption key is not fetched here: its URL is deliberately obfuscated in
# omnigibson.utils.asset_utils.download_key, so the eval env runs that function
# itself (see scripts/setup_eval_env.sh consumers).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${DATA_PATH:-/dev/shm/wangfaye/omnigibson-data}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/BEHAVIOR-1K/.downloads/hf-assets}"
LINK_PATH="$REPO_ROOT/BEHAVIOR-1K/omnigibson-data"
BASE_URL="${BASE_URL:-https://hf-mirror.com/datasets/behavior-1k/zipped-datasets/resolve/main}"
# hf-mirror serves ~1.5 MB/s per connection but ~13 MB/s across 16, so the 29 GB
# core-assets zip needs range requests to finish in under an hour.
PARTS="${PARTS:-16}"

# dataset dir name : zip file name
ASSETS=(
    "omnigibson-robot-assets:omnigibson-robot-assets-3.8.2.zip"
    "2026-challenge-task-instances:2026-challenge-task-instances.zip"
    "behavior-1k-assets:behavior-1k-assets-3.9.0.zip"
)

log() { echo "[$(date '+%F %T')] $*"; }

remote_size() {
    curl -sIL --max-time 60 "$1" \
        | awk 'tolower($1)=="content-length:"{n=$2} END{print n+0}' | tr -dc '0-9'
}

fetch_parallel() {
    # Resumable chunked download: each part is sized up front, so a part whose
    # file already matches its expected length is skipped on re-run.
    local url="$1" out="$2" total="$3"
    local chunk=$(( (total + PARTS - 1) / PARTS ))
    local pending=()

    for ((i = 0; i < PARTS; i++)); do
        local start=$((i * chunk)) end=$((i * chunk + chunk - 1))
        (( end >= total )) && end=$((total - 1))
        local want=$((end - start + 1))
        local part="$out.part_$(printf %02d "$i")"
        local have=0
        [[ -f "$part" ]] && have="$(stat -c %s "$part")"
        (( have == want )) && continue
        rm -f "$part"
        pending+=("$start-$end"$'\t'"$part")
    done

    if (( ${#pending[@]} )); then
        log "  fetching ${#pending[@]}/$PARTS parts"
        # curl 7.61 on this image predates --retry-all-errors.
        printf '%s\n' "${pending[@]}" | xargs -P "$PARTS" -d '\n' -I{} bash -c '
            IFS=$'"'"'\t'"'"' read -r range part <<< "{}"
            curl -sL --retry 5 --retry-delay 2 --retry-connrefused --max-time 3600 \
                -r "$range" -o "$part" "'"$url"'"
        '
    fi

    for ((i = 0; i < PARTS; i++)); do
        [[ -f "$out.part_$(printf %02d "$i")" ]] \
            || { echo "part $i missing after download" >&2; return 1; }
    done
    cat "$out".part_* > "$out"
    local got
    got="$(stat -c %s "$out")"
    if (( got != total )); then
        echo "size mismatch for $out: got $got want $total" >&2
        return 1
    fi
    rm -f "$out".part_*
}

mkdir -p "$CACHE_DIR" "$DATA_PATH"

for entry in "${ASSETS[@]}"; do
    name="${entry%%:*}"
    zip="${entry#*:}"
    target="$DATA_PATH/$name"
    archive="$CACHE_DIR/$zip"

    if [[ -e "$target/VERSION" || ( "$name" == "2026-challenge-task-instances" && -f "$target/metadata/task.jsonl" ) ]]; then
        log "$name already extracted"
        continue
    fi

    log "$name"
    total="$(remote_size "$BASE_URL/$zip")"
    [[ -n "$total" && "$total" -gt 0 ]] || { echo "could not size $zip" >&2; exit 1; }

    if [[ -f "$archive" ]] && (( $(stat -c %s "$archive") == total )); then
        log "  cached archive $zip"
    else
        log "  downloading $zip ($(( total / 1073741824 )).$(( total % 1073741824 * 10 / 1073741824 )) GB)"
        fetch_parallel "$BASE_URL/$zip" "$archive" "$total"
    fi

    log "  verifying archive"
    unzip -tq "$archive" >/dev/null || { echo "corrupt archive: $archive" >&2; exit 1; }

    log "  extracting to $target"
    mkdir -p "$target"
    unzip -q -o "$archive" -d "$target"
done

# Keep the layout the README and eval_b1k.sh expect, without copying 30 GB onto NAS.
if [[ ! -e "$LINK_PATH" || -L "$LINK_PATH" ]]; then
    ln -sfn "$DATA_PATH" "$LINK_PATH"
    log "linked $LINK_PATH -> $DATA_PATH"
fi

log "installed under $DATA_PATH:"
cat "$DATA_PATH/behavior-1k-assets/VERSION" 2>/dev/null | sed 's/^/  behavior-1k-assets /'
cat "$DATA_PATH/omnigibson-robot-assets/VERSION" 2>/dev/null | sed 's/^/  omnigibson-robot-assets /'
test -f "$DATA_PATH/2026-challenge-task-instances/metadata/task.jsonl" \
    && echo "  2026-challenge-task-instances ok"
test -f "$DATA_PATH/omnigibson.key" \
    && echo "  omnigibson.key present" \
    || echo "  omnigibson.key MISSING (run download_behavior_1k_assets from the eval env)"
