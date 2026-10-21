"""Separate strict-FP32 control for the completed default-arithmetic probe."""
import hashlib
import json
from pathlib import Path

import torch

import native_tangents as probe


HERE=Path(__file__).resolve().parent
probe.ROOT=HERE/'strict_fp32_probe'
load_default=probe.load_base


def load_strict(*args,**kwargs):
    if kwargs.get('precision')!='fp32':
        raise ValueError('Strict control is FP32 only')
    base=load_default(*args,**kwargs)
    # The model constructor enables TF32, so override after construction.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    settings=dict(matmul_precision=torch.get_float32_matmul_precision(),
                  cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  torch_version=torch.__version__)
    if settings['matmul_precision']!='highest' or settings['cuda_matmul_allow_tf32']:
        raise RuntimeError('Strict arithmetic configuration was not applied')
    (probe.ROOT/'precision_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    return base


if __name__=='__main__':
    original=HERE/'native_tangent_result.json'
    if not original.exists() or not json.loads(original.read_text())['complete']:
        raise RuntimeError('Original probe must finish before strict control')
    probe.load_base=load_strict
    probe.main()
    (probe.ROOT/'strict_receipt.json').write_text(json.dumps(dict(
        base_result_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        result_sha256=hashlib.sha256((probe.ROOT/'native_tangent_result.json').read_bytes()).hexdigest(),
        settings_sha256=hashlib.sha256((probe.ROOT/'precision_settings.json').read_bytes()).hexdigest(),
        arithmetic='FP32 matrices, TF32 disabled after model construction'),indent=2)+'\n')
