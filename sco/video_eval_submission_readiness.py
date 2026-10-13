"""Prevent freezing a formal package while its policy image is still a draft."""
import json
from pathlib import Path
import re


def image_packaging_readiness(root):
    root = Path(root)
    bundle = root / 'serving_bundle'
    required = ('IMAGE_BUILT.json', 'CPU_IMPORT_CHECK.json',
                'IMAGE_CONTENT_VERIFIED.json', 'IMAGE_PUBLISHED.json',
                'IMAGE_GPU_VERIFIED.json')
    missing = [name for name in required if not (bundle / name).exists()]
    if missing:
        return {'state': 'awaiting_verified_policy_image', 'missing_evidence': missing}
    built, cpu, content, published, gpu = [json.loads((bundle / name).read_text()) for name in required]
    ready = json.loads((root / 'preparation_ready.json').read_text())
    image_id = built['image_id']
    reference = published.get('immutable_reference', '')
    if not re.fullmatch(r'docker\.io/faye0138/behavior-pi05@sha256:[0-9a-f]{64}', reference):
        raise ValueError('Invalid immutable policy image reference')
    if not (built.get('platform') == 'linux/amd64' and cpu.get('exit_code') == 0
            and content.get('verified') is True and content.get('image_id') == image_id
            and content.get('environment_files') == 45571 and content.get('checkpoint_files') == 24
            and published.get('image_id') == image_id and published.get('manifest_verified') is True
            and published.get('is_private') is True
            and gpu.get('verified') is True and gpu.get('image_id') == image_id
            and gpu.get('immutable_reference') == reference
            and gpu.get('policy_rpc_verified') is True
            and gpu.get('image_paths_unshadowed') is True
            and re.fullmatch(r'[0-9a-f]{64}', gpu.get('mountinfo_sha256', ''))
            and gpu.get('single_24gb_compatible') is True):
        raise ValueError('Policy image build, content, registry and GPU evidence do not agree')
    for field in ('source_manifest_sha256', 'checkpoint_ready_sha256'):
        if any(record.get(field) != ready[field] for record in (built, content, published, gpu)):
            raise ValueError('Policy image source identity mismatch')
    readme = (root / 'SUBMISSION_README.md').read_text()
    if reference not in readme or 'remain pending actual image build' in readme:
        return {'state': 'awaiting_final_image_readme', 'immutable_reference': reference}
    return {'state': 'ready', 'immutable_reference': reference}
