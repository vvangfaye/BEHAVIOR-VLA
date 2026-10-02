"""Optional frozen SigLIP features. A pinned revision is part of the data contract."""
from __future__ import annotations
import re
import numpy as np
import torch


class SiglipFeatures:
    def __init__(self, model_id: str, revision: str, device="cpu"):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Use a resolved 40-character Hugging Face commit, not main")
        from transformers import AutoModel, AutoProcessor, __version__ as transformers_version
        self.model = AutoModel.from_pretrained(model_id, revision=revision, trust_remote_code=False).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        self.device = device
        self.feature_id = f"siglip-camera-mean+text-v1:transformers-{transformers_version}:{model_id}@{revision}"
        self._text_cache = {}
        for parameter in self.model.parameters(): parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode_queries(self, texts):
        key = tuple(texts)
        if key not in self._text_cache:
            inputs = self.processor(text=list(texts), padding="max_length", truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
            value = self.model.get_text_features(**inputs)
            if not isinstance(value, torch.Tensor):
                value = getattr(value, "pooler_output", None)
            if not isinstance(value, torch.Tensor):
                raise TypeError("Unsupported transformers get_text_features output; pin a compatible version")
            self._text_cache[key] = value.float().cpu().numpy()
        return self._text_cache[key].copy()

    @torch.inference_mode()
    def encode_images(self, images: tuple[np.ndarray, ...]):
        inputs = self.processor(images=list(images), return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device)
        tokens = self.model.vision_model(pixel_values=pixels).last_hidden_state
        return tokens.mean(dim=1, keepdim=True).float().cpu().numpy()
