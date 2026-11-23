"""Native visual-language LoRA baseline, with no numerical injection."""
import json
from pathlib import Path

import numpy as np
import torch

from runtime import EAVRuntime


class VisualBaselineRuntime(EAVRuntime):
    def __init__(self, cache, model_path, language_checkpoint, graph_checkpoint):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.cache = Path(cache)
        self.protocol = json.loads((self.cache / 'prepared.json').read_text())
        self.logs = np.load(self.cache / 'band_logpower.npy', mmap_mode='r')
        self.weights = np.load(self.cache / 'topology_geometry.npz')['weights']
        self.limits = np.load(self.cache / 'measurements.npz')['limits']
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        base = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16,
            device_map={'': 'cuda'}, attn_implementation='sdpa', local_files_only=True)
        self.model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32,
            lora_dropout=0., target_modules=['q_proj', 'v_proj'], bias='none', task_type='CAUSAL_LM'))
        self.model.eval().requires_grad_(False)
        self.numeric = torch.nn.Identity()
        self.context = {}
        self.language_source = {'merger': {}, 'native_model': str(model_path),
            'initialization': 'Native checkpoint, fresh Q/V LoRA; no restored task adaptation'}

    def batch(self, *args, **kwargs):
        batch = super().batch(*args, **kwargs)
        self.context.clear()
        return batch
