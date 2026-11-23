"""Native VLM video plus deterministic EEG text, with no atlas or residual hook."""
import json

import numpy as np
import torch
from transformers.video_utils import VideoMetadata

from common import ROOT, save
from joint_runtime import V2VideoRuntime
from video_train import video_frames


def measurement_prompt(values):
    measurements = [values[part].mean(axis=(0, 2)).round(4).tolist()
                    for part in [slice(0, 5), slice(5, 10)]]
    return ('EEG log-band-power channel means '
            '(delta, theta, alpha, beta, gamma) for successive five-second windows: '
            + json.dumps(measurements) + '. These measurements do not uniquely establish emotions. '
            'Infer possible participant emotions, using unrestricted vocabulary. '
            'Return only JSON {"emotions":["emotion expression", ...]}.')


class MeasurementTextRuntime(V2VideoRuntime):
    def batch(self, record, joint=False, answer=False):
        assert isinstance(self.numeric, torch.nn.Identity) and not self.context
        content, options = [], {}
        prompt = measurement_prompt(np.array(self.logs[record['index']]))
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
            prompt = 'The video depicts the stimulus, not the participant. ' + prompt
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
        assert 'pixel_values' not in batch and 'image_grid_thw' not in batch
        if not joint:
            assert 'pixel_values_videos' not in batch
        base = self.model.get_base_model().model
        if hasattr(base, 'rope_deltas'):
            base.rope_deltas = None
        return batch.to('cuda')
