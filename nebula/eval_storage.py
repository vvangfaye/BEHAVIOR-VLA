#!/usr/bin/env python3
"""Estimate inference scratch from actual inputs and restore only the tokenizer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import zipfile
GIB = 1024**3
ARCHIVES = ('behavior-1k-assets-3.9.0.zip', 'omnigibson-robot-assets-3.8.2.zip', '2026-challenge-task-instances.zip')
TOKENIZER = 'big_vision/paligemma_tokenizer.model'


def estimate(asset_root, checkpoint, *, raw_bytes=0, result_bytes=0, reserve_gib=16):
    if type(raw_bytes) is not int or raw_bytes < 0 or type(result_bytes) is not int or result_bytes < 0 or type(reserve_gib) is not int or reserve_gib < 8:
        raise ValueError('Invalid storage budget')
    asset_root, checkpoint = Path(asset_root), Path(checkpoint)
    expanded = {}
    for name in ARCHIVES:
        with zipfile.ZipFile(asset_root/name) as archive:
            expanded[name] = sum(info.file_size for info in archive.infolist())
    # Inference loads the trained checkpoint directly. Base checkpoint weights
    # from the old OpenPI cache archive are unnecessary for create_trained_policy.
    model_bytes = sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file())
    required = sum(expanded.values()) + model_bytes + raw_bytes + result_bytes + reserve_gib*GIB
    return dict(schema_version=1, asset_expanded_bytes=expanded, checkpoint_bytes=model_bytes,
                selected_raw_bytes=raw_bytes, maximum_result_bytes=result_bytes, reserve_bytes=reserve_gib*GIB,
                required_disk_bytes=required, cache_mode='tokenizer_only', asset_copy_mode='direct_verified_zip')


def check_space(path, report, *, disk_usage=shutil.disk_usage):
    available = disk_usage(path).free
    evidence = {**report, 'runtime_parent':str(path), 'available_disk_bytes':available}
    print(json.dumps({'storage_preflight':evidence}), flush=True)
    if available < report['required_disk_bytes']:
        raise ValueError(f"{path} has {available/GIB:.1f} GiB free; actual inputs and reserve need {report['required_disk_bytes']/GIB:.1f} GiB")
    return evidence


def restore_tokenizer(archive, destination):
    destination = Path(destination)
    output = destination/TOKENIZER
    if output.exists():raise ValueError('Tokenizer cache requires a fresh destination')
    # Streaming also works for compressed archives, stops after the tokenizer,
    # and avoids materializing unrelated pretrained weights.
    with tarfile.open(archive, mode='r|*') as source:
        for member in source:
            if member.name.lstrip('./') != TOKENIZER:
                continue
            if not member.isfile() or member.size <= 0:
                raise ValueError('Tokenizer must be a nonempty regular file')
            output.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream, output.open('xb') as target:
                shutil.copyfileobj(stream, target)
            if output.stat().st_size != member.size:
                raise ValueError('Incomplete tokenizer extraction')
            return output
    raise ValueError('OpenPI cache archive lacks the required tokenizer')


def main():
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    check=commands.add_parser('check')
    for name in ['asset-root','checkpoint','runtime-parent']:check.add_argument('--'+name,type=Path,required=True)
    check.add_argument('--raw-bytes',type=int,default=0);check.add_argument('--result-bytes',type=int,default=0);check.add_argument('--reserve-gib',type=int,default=16)
    restore=commands.add_parser('tokenizer')
    restore.add_argument('--archive',type=Path,required=True);restore.add_argument('--destination',type=Path,required=True)
    a=p.parse_args()
    if a.command=='check':check_space(a.runtime_parent,estimate(a.asset_root,a.checkpoint,raw_bytes=a.raw_bytes,result_bytes=a.result_bytes,reserve_gib=a.reserve_gib))
    else:print(f'Restored tokenizer: {restore_tokenizer(a.archive,a.destination)}',flush=True)


if __name__=='__main__':main()
