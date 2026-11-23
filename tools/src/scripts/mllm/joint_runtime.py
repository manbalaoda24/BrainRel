"""Shared frozen-V2 restoration and native video batch construction."""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import VideoMetadata
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

from common import ROOT, PROJECT, evaluator, save
from eval_mllm import prompt_for
from video_train import video_frames
from joint_train import REGEX


class V2VideoRuntime:
    def __init__(self, out, checkpoint_path=None, native_model=None):
        self.out = out
        self.metric = evaluator()
        from run import atlas
        from nodes import NumericEvidenceFilter, measurement_nodes
        self.atlas = atlas
        self.measurement_nodes = measurement_nodes
        self.filter_class = NumericEvidenceFilter
        self.source = checkpoint_path if checkpoint_path is not None else ROOT / 'numeric_evidence_filter_full_v1/best.pt'
        assert native_model in [None, 'oneemo', 'qwen25vl3b']
        path = (ROOT / 'open_emotion_baselines_v1/assets/qwen25vl3b'
                if native_model == 'qwen25vl3b' else Path(os.environ.get('FACED_ONEEMO_MODEL', str(PROJECT / 'oneemo/weights'))))
        checkpoint = None if native_model else torch.load(self.source, map_location='cpu', weights_only=False)
        self.checkpoint_epoch = checkpoint['epoch'] if checkpoint is not None else None
        if native_model:
            self.source = path / 'config.json'
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, device_map={'': 'cuda'},
            attn_implementation='sdpa', local_files_only=True)
        self.model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
            target_modules=(['q_proj', 'k_proj', 'v_proj', 'o_proj']
                            if native_model == 'qwen25vl3b' else REGEX), bias='none', task_type='CAUSAL_LM'))
        if checkpoint is not None:
            result = set_peft_model_state_dict(self.model, checkpoint['lora'])
            assert not result.unexpected_keys
        parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, value in (checkpoint['merger'] if checkpoint is not None else {}).items():
                assert name in parameters
                parameters[name].copy_(value.to(parameters[name].device))
        self.model.eval().requires_grad_(False)
        self.model.config.use_cache = False
        self.numeric = (NumericEvidenceFilter(self.model.get_input_embeddings().weight.shape[-1]).cuda()
                        if checkpoint is not None else torch.nn.Identity())
        if checkpoint is not None:
            self.numeric.load_state_dict(checkpoint['numeric_filter'], strict=True)
        self.numeric.eval().requires_grad_(False)
        self.context = {}
        if checkpoint is not None:
            self.model.get_input_embeddings().register_forward_hook(self.inject)
        self.limits = np.load(ROOT / 'traceable_visual_alignment_v1/measurements.npz')['limits']
        self.weights = np.load(ROOT / 'official200_classification_v1/topology_geometry.npz')['weights'].astype('float32')
        self.logs = np.load(ROOT / 'official200_classification_v1/band_logpower.npy', mmap_mode='r')
        self.frames = {}
        self.images = {}
        self.nodes = {}

    def inject(self, module, arguments, output):
        positions = self.context.get('positions')
        if positions is None or output.shape[1] <= int(positions[-1]):
            return output
        with torch.no_grad():
            delta, _, _ = self.numeric(self.context['nodes'])
        reference = output[:, positions]
        residual = self.filter_class.bounded_residual(delta[:, -len(positions):], reference)
        updated = output.clone()
        updated[:, positions] = reference + residual.to(output.dtype)
        return updated

    def batch(self, record, joint=False, answer=False):
        index = record['index']
        if index not in self.images:
            values = np.array(self.logs[index])
            self.images[index] = [self.atlas(values[part].mean(0), self.weights, self.limits).resize(
                (640, 192), Image.Resampling.NEAREST) for part in [slice(0, 5), slice(5, 10)]]
            self.nodes[index] = self.measurement_nodes(values, self.weights, self.limits)
        content = [{'type': 'image'}, {'type': 'image'}]
        options = {'images': self.images[index],
                   'images_kwargs': {'min_pixels': 320 * 96, 'max_pixels': 832 * 224}}
        prompt = prompt_for(record)
        if joint:
            video_id = record['video_id']
            if video_id not in self.frames:
                frames, metadata = video_frames(ROOT / f'stimulus_videos_v1/videos/{video_id}.mp4')
                self.frames[video_id] = frames, metadata
                save(self.out / f'video_{video_id}_sampling.json', metadata)
            frames, metadata = self.frames[video_id]
            content.append({'type': 'video'})
            options.update(videos=[frames], videos_kwargs={'do_sample_frames': False},
                video_metadata=[VideoMetadata(total_num_frames=8, fps=8 / metadata['duration_seconds'],
                    duration=metadata['duration_seconds'], frames_indices=list(range(8)))])
            prompt += ('The video depicts the stimulus, not the participant. Use it as fallible context '
                       'without equating the depicted emotion with this participant\'s experience. ')
        content.append({'type': 'text', 'text': prompt})
        messages = [{'role': 'user', 'content': content}]
        prefix = self.processor.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        prefix_batch = self.processor(text=[prefix], return_tensors='pt', **options)
        count = prefix_batch['input_ids'].shape[1]
        if answer:
            full = self.processor.apply_chat_template(messages + [
                {'role': 'assistant', 'content': json.dumps(record['target'])}], tokenize=False,
                add_generation_prompt=False, enable_thinking=False)
            batch = self.processor(text=[full], return_tensors='pt', **options)
            assert torch.equal(batch['input_ids'][:, :count], prefix_batch['input_ids'])
            labels = batch['input_ids'].clone()
            labels[:, :count] = -100
            assert (labels != -100).any()
            batch['labels'] = labels
        else:
            batch = prefix_batch
        positions = torch.arange(count - 32, count, device='cuda')
        for name in ['image_token_id', 'video_token_id']:
            token = getattr(self.model.config, name, None)
            assert token is None or not (batch['input_ids'][:, count - 32:count] == token).any()
        self.context.update(positions=positions,
            nodes=torch.tensor(self.nodes[index], device='cuda').unsqueeze(0))
        base = self.model.get_base_model().model
        if hasattr(base, 'rope_deltas'):
            base.rope_deltas = None
        return batch.to('cuda')
