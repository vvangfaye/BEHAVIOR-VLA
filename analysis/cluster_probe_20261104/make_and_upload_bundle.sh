#!/bin/bash
# Package RLinf (code + 3 venvs) + uv python + glvnd libs + caches + models + RoboCasa assets and upload to AOSS
# (aoss://260010165/rlinf_bundle_20261104/). Cluster jobs cannot reach HF/GitHub (probe 2026-11-04), so everything is
# staged through AOSS. Upload order = what the first cluster runs need (CALVIN/LIBERO before RoboCasa).
set -euo pipefail
B=/mnt/data1/faye/rlinf_bundle_20261104
HERE="$(cd "$(dirname "$0")" && pwd)"
M="/mnt/data1/faye/code/Behavior-1k Challenge/analysis/libero_feasibility_20261103/models"
mkdir -p "$B/sys_extra/lib" "$B/sys_extra/home/.libero" "$B/sys_extra/home/.cache"
log() { echo "$(date '+%F %T') $*"; }
# 1. small system extras: glvnd dispatch libs (missing from the cluster images), LIBERO config, openpi cache
for f in libEGL.so.1 libEGL.so.1.1.0 libGLdispatch.so.0 libGLdispatch.so.0.0.0 libOpenGL.so.0 libOpenGL.so.0.0.0 libGL.so.1 libGL.so.1.7.0 libGLX.so.0 libGLX.so.0.0.0; do
    cp -P /lib/x86_64-linux-gnu/$f "$B/sys_extra/lib/"; done
cp ~/.libero/config.yaml "$B/sys_extra/home/.libero/"
cp -r /mnt/data1/faye/dotfiles/.cache/openpi "$B/sys_extra/home/.cache/"
tar -C "$B" -czf "$B/sys_extra.tar.gz" sys_extra; log "sys_extra $(du -sh $B/sys_extra.tar.gz | cut -f1)"
# 2. uv-managed python (the venvs' base interpreter)
tar -C /home/faye/.local/share/uv/python -czf "$B/uv_python.tar.gz" cpython-3.11.14-linux-x86_64-gnu; log "uv_python $(du -sh $B/uv_python.tar.gz | cut -f1)"
# 3. RLinf code + venvs + local configs (hardlinks between venvs are kept inside the tar)
tar -C /mnt/data1/faye/code/libero_rlinf_20261103 -I "pigz -p 48" -cf "$B/rlinf_code_venvs.tar.gz" RLinf configs; log "code_venvs $(du -sh $B/rlinf_code_venvs.tar.gz | cut -f1)"
# 4. LIBERO + CALVIN models (no compression)
tar -C "$M" -cf "$B/models_libero_calvin.tar" RLinf-Pi05-LIBERO-SFT RLinf-Pi05-CALVIN-ABC-D-SFT; log "models_libero_calvin $(du -sh $B/models_libero_calvin.tar | cut -f1)"
( cd "$B" && sha256sum sys_extra.tar.gz uv_python.tar.gz rlinf_code_venvs.tar.gz models_libero_calvin.tar > SHA256SUMS_part1 )
cd "$HERE"
for f in SHA256SUMS_part1 sys_extra.tar.gz uv_python.tar.gz rlinf_code_venvs.tar.gz models_libero_calvin.tar; do
    log "upload $f"; uv run --offline --with boto3 python upload.py "$B/$f" "rlinf_bundle_20261104/$f"; done
log "PART1 UPLOADED"
# 5. RoboCasa model + assets
tar -C "$M" -cf "$B/models_robocasa.tar" RLinf-pi0-robocasa-pretrain-human300; log "models_robocasa $(du -sh $B/models_robocasa.tar | cut -f1)"
tar -C /mnt/data1/faye/code/libero_rlinf_20261103 -cf "$B/robocasa_assets.tar" robocasa_assets; log "robocasa_assets $(du -sh $B/robocasa_assets.tar | cut -f1)"
( cd "$B" && sha256sum models_robocasa.tar robocasa_assets.tar > SHA256SUMS_part2 )
cd "$HERE"
for f in SHA256SUMS_part2 models_robocasa.tar robocasa_assets.tar; do
    log "upload $f"; uv run --offline --with boto3 python upload.py "$B/$f" "rlinf_bundle_20261104/$f"; done
log "PART2 UPLOADED"
