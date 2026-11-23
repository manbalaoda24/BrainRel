"""Eight input combinations at a frozen sparse checkpoint, with a common prompt."""
import argparse
import itertools
import json
import time

import numpy as np
from PIL import Image
import torch
from transformers.video_utils import VideoMetadata

from common import ROOT, lock_gpu, panel, prediction_row, save, sha, write_rows
from parser import parse_structured
from sparse import BudgetedRelationFilter
from train_graph import HypergraphRuntime
from video_train import video_frames


PROMPT = (
    'Predict the participant emotional states using only the supplied fallible inputs. '
    'Return only JSON {"emotions":["emotion expression", ...]}. '
    'Vocabulary and number of emotions are unrestricted; an empty list is permitted. '
    'Do not invent events, diagnoses or physiological causes. '
    'If EEG maps are supplied, the two fixed-scale five-band maps summarize seconds0-5 '
    'and5-10 of one segment; red is higher displayed power and blue lower. '
    'Upper/lower image halves are not anatomical lobes. '
    'A numerical EEG residual, when enabled, is an auxiliary representation of the same EEG. '
    'Measurements do not uniquely establish any emotion. '
    'If video is supplied, it depicts the stimulus, not the participant; use it as fallible '
    'context without equating the depicted emotion with the participant experience. '
    'Missing inputs provide no evidence and should not be imagined.')

COMBINATIONS = [''.join(map(str, values)) for values in itertools.product([0, 1], repeat=3)]
QUEUES = {0: ['111', '000', '100'], 1: ['011', '010', '001'], 2: ['110', '101']}


def batch_for(runtime, record, combination):
    atlas_on, video_on, numeric_on = map(int, combination)
    runtime.context.clear()
    content, options = [], {}
    index = record['index']
    if atlas_on or numeric_on:
        values = np.array(runtime.logs[index])
        if numeric_on and index not in runtime.nodes:
            runtime.nodes[index] = runtime.measurement_nodes(values, runtime.weights, runtime.limits)
        if atlas_on:
            if index not in runtime.images:
                runtime.images[index] = [runtime.atlas(values[part].mean(0), runtime.weights,
                    runtime.limits).resize((640, 192), Image.Resampling.NEAREST)
                    for part in [slice(0, 5), slice(5, 10)]]
            content.extend([{'type': 'image'}, {'type': 'image'}])
            options.update(images=runtime.images[index], images_kwargs={
                'min_pixels': 320 * 96, 'max_pixels': 832 * 224})
    if video_on:
        video_id = record['video_id']
        if video_id not in runtime.frames:
            frames, metadata = video_frames(ROOT / f'stimulus_videos_v1/videos/{video_id}.mp4')
            runtime.frames[video_id] = frames, metadata
            save(runtime.out / f'video_{video_id}_sampling.json', metadata)
        frames, metadata = runtime.frames[video_id]
        content.append({'type': 'video'})
        options.update(videos=[frames], videos_kwargs={'do_sample_frames': False},
            video_metadata=[VideoMetadata(total_num_frames=8, fps=8 / metadata['duration_seconds'],
                duration=metadata['duration_seconds'], frames_indices=list(range(8)))])
    content.append({'type': 'text', 'text': PROMPT})
    prefix = runtime.processor.apply_chat_template([{'role': 'user', 'content': content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    batch = runtime.processor(text=[prefix], return_tensors='pt', **options)
    length = batch['input_ids'].shape[1]
    if numeric_on:
        positions = torch.arange(length - 32, length, device='cuda')
        for name in ['image_token_id', 'video_token_id']:
            token = getattr(runtime.model.config, name, None)
            assert token is None or not (batch['input_ids'][:, length - 32:length] == token).any()
        runtime.context.update(positions=positions,
            nodes=torch.tensor(runtime.nodes[index], device='cuda').unsqueeze(0))
    assert ('positions' in runtime.context) == bool(numeric_on)
    assert len(options.get('images', [])) == 2 * atlas_on
    assert ('videos' in options) == bool(video_on)
    base = runtime.model.get_base_model().model
    if hasattr(base, 'rope_deltas'):
        base.rope_deltas = None
    return batch.to('cuda')


def main(args):
    gpu_lock = lock_gpu()
    torch.set_num_threads(2)
    torch.manual_seed(20261114)
    out = ROOT / f'modality_factorial_sparse_v1_gpu{args.queue}'
    out.mkdir(exist_ok=False)
    save(out / 'status.json', {'phase': 'loading', 'test_read': False})
    source = ROOT / 'relation_repair_sparse_v1/best.pt'
    language = ROOT / 'v2_video_joint_lora_pilot_v1/best.pt'
    runtime = HypergraphRuntime(out, checkpoint_path=language)
    module = BudgetedRelationFilter(runtime.numeric,
        runtime.model.get_input_embeddings().weight.shape[-1], sparse=True).cuda()
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    module.load_state_dict(checkpoint['hypergraph'], strict=True)
    module.epoch = int(checkpoint['epoch'])
    module.eval().requires_grad_(False)
    runtime.numeric, runtime.filter_class = module, BudgetedRelationFilter
    runtime.model.eval().requires_grad_(False)
    records = panel()['val']
    assert len(records) == 560
    protocol = {'training': False, 'test_read': False, 'checkpoint': str(source),
        'checkpoint_sha256': sha(source), 'language_checkpoint': str(language),
        'language_sha256': sha(language), 'epoch': module.epoch, 'budgets': list(module.budgets()),
        'combination_bits': ['EEG_atlases', 'stimulus_video', 'numerical_EEG_branch'],
        'queue': QUEUES[args.queue], 'all_combinations': COMBINATIONS,
        'prompt': PROMPT, 'identical_prompt_across_combinations': True,
        'samples_per_condition': 560, 'generation': {'do_sample': False, 'max_new_tokens': 320},
        'panel_sha256': sha(ROOT / 'numeric_evidence_filter_full_v1/val_open_records.jsonl'),
        'script_sha256': sha(__file__), 'reference': runtime.metric.protocol,
        'scope': 'Fixed-checkpoint input-removal sensitivity, not separately retrained modality baselines',
        'prompt_change': 'Common availability-conditional emotion-only prompt, unlike previous measurement-query prompt. All eight conditions regenerated; do not substitute older scores.',
        'dependencies': 'Atlases and numerical branch derive from the same EEG; they are views, not independent sensing modalities. Known stimulus videos are shared across held-out participants.',
        'zero_input': '000 is language prior only, not a sensory baseline or physiological intervention'}
    save(out / 'protocol.json', protocol)
    write_rows(out / 'validation_records.jsonl', records)
    results = {}
    started = time.time()
    with torch.inference_mode():
        preflight = {}
        for combination in COMBINATIONS:
            batch = batch_for(runtime, records[0], combination)
            preflight[combination] = {'input_tokens': batch['input_ids'].shape[1],
                'images_present': 'pixel_values' in batch,
                'video_present': 'pixel_values_videos' in batch,
                'numerical_injection_enabled': 'positions' in runtime.context}
            del batch
        save(out / 'preflight.json', preflight)
        for combination in QUEUES[args.queue]:
            rows = []
            with (out / f'{combination}_records.jsonl').open('x') as stream:
                for number, record in enumerate(records):
                    batch = batch_for(runtime, record, combination)
                    generated = runtime.model.generate(**batch, do_sample=False, max_new_tokens=320)
                    tail = generated[0, batch['input_ids'].shape[1]:]
                    raw = runtime.processor.tokenizer.decode(tail, skip_special_tokens=True).strip()
                    terms, valid = parse_structured(raw)
                    row = prediction_row(record, terms, valid, raw=raw,
                        budget_reached=len(tail) == 320, input_tokens=batch['input_ids'].shape[1])
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stream.flush()
                    if number % 20 == 0:
                        save(out / 'status.json', {'phase': 'evaluating', 'combination': combination,
                            'completed': number + 1, 'total': 560, 'queue': QUEUES[args.queue],
                            'seconds': time.time() - started, 'test_read': False})
            results[combination] = runtime.metric.score(rows)
            save(out / f'{combination}_metrics.json', results[combination])
            save(out / 'summary.json', results)
    save(out / 'complete.json', {'phase': 'complete', 'results': results,
        'seconds': time.time() - started, 'test_read': False})
    save(out / 'status.json', {'phase': 'complete', 'test_read': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=int, choices=[0, 1, 2], required=True)
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        folder = ROOT / f'modality_factorial_sparse_v1_gpu{arguments.queue}'
        if folder.exists():
            save(folder / 'status.json', {'phase': 'failed', 'error': str(error), 'test_read': False})
        raise
