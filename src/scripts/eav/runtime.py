"""EAV interface for frozen BrainRel transfer and optional LoRA adaptation."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / 'scripts/mllm'), str(PROJECT / 'scripts/faced')]
from nodes import NumericEvidenceFilter, measurement_nodes
from experts import HypergraphRelationExperts
from run import atlas

PROMPT = ('Predict the participant emotion category for this conversational interaction. '
          'Return only JSON {"emotions":["one category"]}. '
          'Choose exactly one of neutral, sadness, anger, happiness, calmness. '
          'Any video shows this participant, not an external stimulus movie. '
          'The two fixed-scale five-band EEG maps cover seconds 5-10 and 10-15 of the same interaction. '
          'Red means higher displayed power and blue lower. Upper/lower image halves are not anatomical lobes. '
          'Do not infer a unique physiological cause from these maps.')


def frames(path):
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count / fps < 15:
            raise ValueError(f'Video shorter than aligned window: {path}')
        selected = np.rint(np.linspace(5 * fps, 15 * fps - 1, 8)).astype(int)
        images = []
        actual = []
        for number in selected:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(number))
            valid, image = capture.read()
            if not valid:
                raise ValueError('Failed synchronized frame decoding; no silent fallback')
            actual.append(int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            ratio = min(1., 448 / max(height, width))
            images.append(cv2.resize(image, (max(1, round(width * ratio)), max(1, round(height * ratio)))))
        return np.stack(images), {'source_fps': fps, 'source_count': count,
            'requested_indices': selected.tolist(), 'decoded_indices': actual, 'window_seconds': [5, 15]}
    finally:
        capture.release()


class EAVRuntime:
    def __init__(self, cache, model_path, language_checkpoint, graph_checkpoint):
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from joint_train import REGEX
        self.cache = Path(cache)
        self.protocol = json.loads((self.cache / 'prepared.json').read_text())
        self.logs = np.load(self.cache / 'band_logpower.npy', mmap_mode='r')
        self.weights = np.load(self.cache / 'topology_geometry.npz')['weights']
        self.limits = np.load(self.cache / 'measurements.npz')['limits']
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        base = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16,
            device_map={'': 'cuda'}, attn_implementation='sdpa', local_files_only=True)
        self.model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
            target_modules=REGEX, bias='none', task_type='CAUSAL_LM'))
        language = torch.load(language_checkpoint, map_location='cpu', weights_only=False)
        restored = set_peft_model_state_dict(self.model, language['lora'])
        if restored.unexpected_keys:
            raise ValueError(f'Unexpected LoRA keys: {restored.unexpected_keys}')
        parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, value in language['merger'].items():
                parameters[name].copy_(value.to(parameters[name].device))
        hidden = self.model.get_input_embeddings().weight.shape[-1]
        self.numeric = HypergraphRelationExperts(NumericEvidenceFilter(hidden), hidden).cuda()
        graph = torch.load(graph_checkpoint, map_location='cpu', weights_only=False)
        self.numeric.load_state_dict(graph['hypergraph'], strict=True)
        self.model.eval().requires_grad_(False)
        self.numeric.eval().requires_grad_(False)
        self.context = {}
        self.numeric_training = False
        self.model.get_input_embeddings().register_forward_hook(self.inject)
        self.language_source = language

    def inject(self, module, arguments, output):
        positions = self.context.get('positions')
        if positions is None or output.shape[1] <= int(positions[-1]):
            return output
        with torch.set_grad_enabled(self.numeric_training and torch.is_grad_enabled()):
            delta = self.numeric(self.context['nodes'])[0]
        reference = output[:, positions]
        residual = self.numeric.bounded_residual(delta, reference)
        updated = output.clone()
        updated[:, positions] = reference + residual.to(output.dtype)
        return updated

    def batch(self, record, arm='joint', answer=False, prompt=None):
        from transformers.video_utils import VideoMetadata
        if record['split'] not in ['train', 'val']:
            raise ValueError('Test access is disabled')
        values = np.array(self.logs[record['index']])
        self.frame_metadata = None
        content, options = [], {}
        if arm in ['eeg', 'joint']:
            content.extend([{'type': 'image'}, {'type': 'image'}])
            options.update(images=[atlas(values[part].mean(0), self.weights, self.limits).resize((640, 192), Image.Resampling.NEAREST)
                                   for part in [slice(0, 5), slice(5, 10)]],
                           images_kwargs={'min_pixels': 320 * 96, 'max_pixels': 832 * 224})
        if arm in ['video', 'joint']:
            sampled, metadata = frames(record['video_path'])
            content.append({'type': 'video'})
            options.update(videos=[sampled], videos_kwargs={'do_sample_frames': False},
                video_metadata=[VideoMetadata(total_num_frames=8, fps=.8, duration=10., frames_indices=list(range(8)))])
            self.frame_metadata = metadata
        content.append({'type': 'text', 'text': PROMPT if prompt is None else prompt})
        messages = [{'role': 'user', 'content': content}]
        prefix = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        batch = self.processor(text=[prefix], return_tensors='pt', **options)
        length = batch['input_ids'].shape[1]
        if answer:
            full = self.processor.apply_chat_template(messages + [{'role': 'assistant', 'content': json.dumps(record['target'])}],
                tokenize=False, add_generation_prompt=False, enable_thinking=False)
            full_batch = self.processor(text=[full], return_tensors='pt', **options)
            if not torch.equal(full_batch['input_ids'][:, :length], batch['input_ids']):
                raise ValueError('Answer template does not preserve prefix')
            batch = full_batch
            batch['labels'] = batch['input_ids'].clone()
            batch['labels'][:, :length] = -100
        self.context.clear()
        if arm != 'video':
            positions = torch.arange(length - 32, length, device='cuda')
            for name in ['image_token_id', 'video_token_id']:
                token = getattr(self.model.config, name, None)
                if token is not None and (batch['input_ids'][:, length - 32:length] == token).any():
                    raise ValueError('Residual overlaps visual placeholders')
            self.context.update(positions=positions, nodes=torch.tensor(measurement_nodes(values, self.weights, self.limits), device='cuda')[None])
        base = self.model.get_base_model().model if hasattr(self.model, 'get_base_model') else self.model.model
        if hasattr(base, 'rope_deltas'):
            base.rope_deltas = None
        return batch.to('cuda')
