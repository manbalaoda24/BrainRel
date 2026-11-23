"""Released native model with fresh language LoRA and video-only inputs."""
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import VideoMetadata

from common import ROOT, PROJECT, evaluator, save
from eval_video import PROMPT
from eval_erv import load_frame_reader
from joint_train import REGEX


class VideoOnlyRuntime:
    def __init__(self, out, native_model):
        assert native_model in ['oneemo', 'qwen25vl3b']
        self.out = out
        self.source = (ROOT / 'open_emotion_baselines_v1/assets/qwen25vl3b/config.json'
            if native_model == 'qwen25vl3b' else
            Path(os.environ.get('FACED_ONEEMO_MODEL', str(PROJECT / 'oneemo/weights'))) / 'config.json')
        self.processor = AutoProcessor.from_pretrained(self.source.parent, local_files_only=True)
        base = AutoModelForImageTextToText.from_pretrained(self.source.parent, dtype=torch.bfloat16,
            device_map={'': 'cuda'}, attn_implementation='sdpa', local_files_only=True)
        self.model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'] if native_model == 'qwen25vl3b' else REGEX,
            bias='none', task_type='CAUSAL_LM'))
        self.model.eval().requires_grad_(False)
        self.model.config.use_cache = False
        self.numeric = torch.nn.Identity()
        self.metric = evaluator()
        self.frames = {}
        self.reader = load_frame_reader()

    def batch(self, record, joint=False, answer=False):
        key = record['video_id']
        if key not in self.frames:
            self.frames[key] = self.reader(ROOT / f'stimulus_videos_v1/videos/{key}.mp4')
            save(self.out / f'video_{key}_sampling.json', self.frames[key][1])
        frames, metadata = self.frames[key]
        options = dict(videos=[frames], videos_kwargs={'do_sample_frames': False},
            video_metadata=[VideoMetadata(total_num_frames=8, fps=8 / metadata['duration_seconds'],
                duration=metadata['duration_seconds'], frames_indices=list(range(8)))])
        messages = [dict(role='user', content=[dict(type='video'), dict(type='text', text=PROMPT)])]
        prefix = self.processor.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        prefix_batch = self.processor(text=[prefix], return_tensors='pt', **options)
        count = prefix_batch['input_ids'].shape[1]
        if answer:
            full = self.processor.apply_chat_template(messages + [dict(role='assistant',
                content=json.dumps(record['target']))], tokenize=False,
                add_generation_prompt=False, enable_thinking=False)
            batch = self.processor(text=[full], return_tensors='pt', **options)
            assert torch.equal(batch['input_ids'][:, :count], prefix_batch['input_ids'])
            labels = batch['input_ids'].clone()
            labels[:, :count] = -100
            assert (labels != -100).any()
            batch['labels'] = labels
        else:
            batch = prefix_batch
        assert 'pixel_values_videos' in batch and 'pixel_values' not in batch and 'image_grid_thw' not in batch
        base = self.model.get_base_model().model
        if hasattr(base, 'rope_deltas'):
            base.rope_deltas = None
        return batch.to('cuda')
