"""Frozen unrestricted-label generation with explicit single-category evaluation limits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image


def prompt_for(arm):
    descriptions = {
        'eeg': 'The inputs are two fixed-scale EEG power atlases for seconds 5-10 and 10-15 of one interaction, with a numerical EEG residual. No face video is supplied.',
        'video': 'The input is a sampled face video of the participant during seconds 5-15 of one conversational interaction. No EEG input is supplied.',
        'joint': 'The inputs are two fixed-scale EEG power atlases for seconds 5-10 and 10-15, a numerical EEG residual, and sampled face video of the same participant during seconds 5-15.'}
    return (descriptions[arm] + ' Please recognize all possible emotional states of the participant. '
        'There is no predefined emotion vocabulary or required number of emotions. '
        'Return only JSON {"emotions":[...]} containing the emotion words you predict; use [] if insufficient evidence. '
        'Use English emotion words, not measurement properties. Do not list alternatives merely because they are possible. '
        'Do not assume any EEG frequency or image color uniquely identifies an emotion. '
        'Any video depicts this participant, not an external stimulus movie. '
        'For EEG maps if present, red means higher displayed power and blue lower; upper/lower image halves are not anatomical lobes.')


def parse_answer(raw):
    decoder = json.JSONDecoder()
    for start, character in enumerate(raw):
        if character != '{':
            continue
        try:
            value, _ = decoder.raw_decode(raw[start:])
        except ValueError:
            continue
        if not isinstance(value, dict):
            continue
        terms = value.get('emotions')
        if isinstance(terms, list) and all(isinstance(term, str) and term.strip() for term in terms):
            return list(dict.fromkeys(term.strip().lower() for term in terms)), True
    return [], False


def write_json(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main(args):
    import fcntl
    lock = (args.cache.parent / f'eav_gpu{args.gpu}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    used = int(subprocess.check_output(['nvidia-smi', f'--id={args.gpu}', '--query-gpu=memory.used',
        '--format=csv,noheader,nounits'], text=True).strip())
    if used >= 1000:
        raise RuntimeError('Requested GPU occupied; refusing to load generation model')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from runtime import EAVRuntime, atlas, frames, measurement_nodes
    from wheel import OfficialEW
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'status.json', {'phase': 'loading_model', 'gpu': args.gpu, 'test_read': False})
    torch.set_num_threads(2)
    torch.manual_seed(20261113)
    records = [json.loads(line) for line in (args.cache / 'val_records.jsonl').read_text().splitlines()]
    assert len(records) == 700 and {row['subject'] for row in records} == set(range(29, 36))
    runtime = EAVRuntime(args.cache, args.model, args.language_checkpoint, args.graph_checkpoint)
    metric = OfficialEW(args.faced_root)
    cases = set()
    groups = set()
    for row in records:
        key = row['subject'], row['label']
        if key not in groups:
            groups.add(key)
            cases.add(row['sample_id'])
    assert len(cases) == 35
    protocol = {
        'arms': args.arms, 'prompts': {arm: prompt_for(arm) for arm in args.arms},
        'script_sha256': sha(Path(__file__)),
        'language_checkpoint': str(args.language_checkpoint), 'language_sha256': sha(args.language_checkpoint),
        'graph_checkpoint': str(args.graph_checkpoint), 'graph_sha256': sha(args.graph_checkpoint),
        'validation_records_sha256': sha(args.cache / 'val_records.jsonl'),
        'samples': 700, 'training': False, 'test_read': False, 'same_weights_across_arms': True,
        'reference': 'Provided single EAV category, not independent open-vocabulary emotion gold',
        'primary_scope': 'Unrestricted generation and mapped category-reference coverage only',
        'generation': {'do_sample': False, 'max_new_tokens': 160},
        'parser': 'Decode explicit emotions JSON; preserve raw output; invalid parses count as empty predictions',
        'ew': metric.protocol, 'visual_cases': sorted(cases),
        'case_selection': 'First cached interaction for each validation subject/category before reading model outputs',
        'provenance': 'Actual frozen FACED-initialized numerical module; unsigned computational paths, not emotion contributions',
        'no_classifier_or_true_label_in_prompt': True}
    write_json(args.out / 'protocol.json', protocol)
    write_json(args.out / 'graph_schema.json', runtime.numeric.describe())
    started = time.monotonic()
    summary = {}
    for arm in args.arms:
        directory = args.out / arm
        directory.mkdir()
        rows = []
        with (directory / 'records.jsonl').open('x') as stream, torch.inference_mode():
            for number, record in enumerate(records):
                batch = runtime.batch(record, arm=arm, prompt=prompt_for(arm))
                result = runtime.model.generate(**batch, do_sample=False, max_new_tokens=160)
                tokens = result[0, batch['input_ids'].shape[1]:]
                raw = runtime.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                terms, valid = parse_answer(raw)
                row = {'index': record['index'], 'sample_id': record['sample_id'], 'subject': record['subject'],
                    'interaction_id': record['interaction_id'], 'required_concepts': [record['label']],
                    'emotion_terms': terms, 'parse_valid': valid, 'raw': raw, 'arm': arm,
                    'budget_reached': len(tokens) == 160,
                    'generated_tokens': len(tokens), 'response_characters': len(raw),
                    'frame_metadata': runtime.frame_metadata}
                if record['sample_id'] in cases:
                    case = directory / 'cases' / record['sample_id']
                    case.mkdir(parents=True)
                    if arm != 'video':
                        values = np.array(runtime.logs[record['index']])
                        np.savez_compressed(case / 'source_measurements.npz', logpower=values,
                            nodes=measurement_nodes(values, runtime.weights, runtime.limits), limits=runtime.limits)
                        for name, part in [('first', slice(0, 5)), ('last', slice(5, 10))]:
                            atlas(values[part].mean(0), runtime.weights, runtime.limits).resize(
                                (640, 192), Image.Resampling.NEAREST).save(case / f'{name}_atlas.png')
                        write_json(case / 'trace.json', runtime.numeric.relation_trace())
                    if arm != 'eeg':
                        sampled, metadata = frames(record['video_path'])
                        for frame_number, frame in enumerate(sampled):
                            Image.fromarray(frame).save(case / f'frame_{frame_number:02d}.png')
                        write_json(case / 'frame_metadata.json', metadata)
                    write_json(case / 'answer.json', row)
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                write_json(args.out / 'status.json', {'phase': 'generating', 'arm': arm,
                    'completed': number + 1, 'total': 700, 'elapsed_seconds': time.monotonic() - started, 'test_read': False})
        scores = metric.score(rows)
        classes = runtime.protocol['classes']
        baselines = {}
        for name, terms in [('report_all_five', classes), ('constant_neutral', ['neutral'])]:
            baselines[name] = metric.score([{**row, 'emotion_terms': terms, 'parse_valid': True} for row in rows])
        scores.update({'metric_scope': 'Official wheel mapping against provided single category; not independent open-vocabulary gold',
            'exact_category_coverage': sum(row['required_concepts'][0] in row['emotion_terms'] and row['parse_valid'] for row in rows) / len(rows),
            'unique_emotion_sets': len({tuple(sorted(row['emotion_terms'])) for row in rows}),
            'mean_terms': sum(len(row['emotion_terms']) for row in rows) / len(rows),
            'all_five_literal_inclusion': sum(set(classes) <= set(row['emotion_terms']) for row in rows),
            'generation_budget_reached': sum(row['budget_reached'] for row in rows), 'mechanical_baselines': baselines})
        write_json(directory / 'metrics.json', scores)
        name2reason = {row['sample_id']: row['raw'] for row in rows}
        np.savez_compressed(directory / 'name2reason.npz', name2reason=name2reason)
        items = np.empty(len(rows), dtype=object)
        items[:] = [row['emotion_terms'] for row in rows]
        np.savez_compressed(directory / 'openset.npz', filenames=np.array(list(name2reason), dtype=object),
            fileitems=items, parse_valid=np.array([row['parse_valid'] for row in rows]))
        summary[arm] = scores
    write_json(args.out / 'complete.json', {'phase': 'complete', 'metrics': summary,
        'elapsed_seconds': time.monotonic() - started, 'test_read': False})
    write_json(args.out / 'status.json', {'phase': 'complete', 'test_read': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--arms', nargs='+', choices=['eeg', 'video', 'joint'], required=True)
    parser.add_argument('--model', default='/home/ubuntu/work/eeg_project/oneemo/weights')
    parser.add_argument('--faced-root', type=Path, default=Path('/data/model/datasets/FACED_syn50614194'))
    parser.add_argument('--language-checkpoint', type=Path, default=Path('/data/model/datasets/EAV_13799131/brainrel_lora_joint_pilot_v1/last_language.pt'))
    parser.add_argument('--graph-checkpoint', type=Path, default=Path('/data/model/datasets/FACED_syn50614194/v2_hypergraph_experts_pilot_v1/best.pt'))
    arguments = parser.parse_args()
    preexisting = arguments.out.exists()
    try:
        main(arguments)
    except Exception as error:
        if arguments.out.exists() and not preexisting:
            write_json(arguments.out / 'status.json', {'phase': 'failed', 'error': str(error), 'test_read': False})
        raise
