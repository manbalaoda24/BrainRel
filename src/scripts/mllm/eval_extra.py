"""Fixed zero-shot FACED test evaluation; no training or test-based selection."""
import argparse
import ast
import json
import os
from pathlib import Path
import time
import traceback

from common import ROOT, PROJECT, evaluator, lock_gpu, prediction_row, save, sha
from interfaces import InternVL35, R1Omni
from eval_erv import load_frame_reader, parse as parse_native
from exact import metrics


def source_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    function = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def main(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), FACED_GPU_INDEX=str(args.gpu),
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    out = ROOT / 'final_test_20261120' / f'{args.model}_zeroshot_v1'
    out.mkdir(exist_ok=False)
    save(out / 'status.json', dict(phase='loading', training=False, test_read=False))
    gpu_lock = lock_gpu()
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    torch.set_num_threads(4)
    torch.manual_seed(20261109)
    metric = evaluator()
    prepared = ROOT / 'additional_baselines_v1' / args.model
    if not (prepared / 'cpu_readiness.json').exists():
        raise RuntimeError('Native CPU readiness check missing')
    runtime = InternVL35(prepared) if args.model == 'internvl35' else R1Omni(prepared)
    base = ROOT / 'final_test_20261120'
    records = [json.loads(line) for line in (base / 'test_records.jsonl').read_text().splitlines()]
    if len(records) != 644 or {row['subject'] for row in records} != set(range(100, 123)):
        raise ValueError('Unexpected fixed test panel')
    if len({row['index'] for row in records}) != 644 or not all(
            row['index'] == row['subject'] * 84 + (row['video_id'] - 1) * 3 + 1 for row in records):
        raise ValueError('Unexpected trial/segment indices')
    logs = np.load(base / 'band_logpower.npy', mmap_mode='r')
    reader = load_frame_reader()
    if args.model == 'internvl35':
        atlas = source_function(PROJECT / 'scripts/faced/run.py', 'atlas',
            dict(np=np, Image=Image, ImageDraw=ImageDraw,
                 BANDS=[(name, 0, 0) for name in ['delta', 'theta', 'alpha', 'beta', 'gamma']]))
        weights = np.load(ROOT / 'official200_classification_v1/topology_geometry.npz')['weights'].astype('float32')
        limits = np.load(ROOT / 'traceable_visual_alignment_v1/measurements.npz')['limits']
        parser = source_function(Path(__file__).with_name('eval_mllm.py'), 'parse_answer', dict(json=json))
        arms = ['joint', 'eeg', 'video']
    else:
        parser, arms = parse_native, ['joint', 'video']
    save(out / 'protocol.json', dict(model=args.model, training=False, test_read=True, n=644,
        subjects=list(range(100, 123)), arms=arms, model_assets=json.loads((prepared / 'assets.json').read_text()),
        script_sha256=sha(__file__), interface_sha256=sha(Path(__file__).with_name('interfaces.py')),
        manifest_sha256=sha(base / 'test_records.jsonl'), metric=metric.protocol,
        selection='Fixed released checkpoint; no test-based model or prompt selection',
        input_boundary='Known stimulus videos, unseen participants; middle 10-second EEG segment',
        parsing='Explicit JSON emotions for InternVL; native answer span for R1; invalid mapped to empty',
        generation=dict(do_sample=False, max_new_tokens=320 if args.model == 'internvl35' else 512),
        prompt_policy='Fixed model-native interface; not identical to earlier baseline prompts',
        exact_metrics='Existing exact-set diagnostics; both-empty Jaccard=1',
        video_only='One deterministic output per stimulus reused across participants',
        eeg_interface='Two measurement atlases' if args.model == 'internvl35' else 'Measured EEG text',
        audio_input=False))
    started, results, video_cache = time.time(), {}, {}
    for arm in arms:
        predictions, generated_video = [], {}
        with (out / f'{arm}_records.jsonl').open('x') as stream:
            for number, record in enumerate(records):
                video_id = record['video_id']
                frame_metadata, frames = None, ()
                if arm != 'eeg':
                    if video_id not in video_cache:
                        video_cache[video_id] = reader(ROOT / f'stimulus_videos_v1/videos/{video_id}.mp4')
                    frames, frame_metadata = video_cache[video_id]
                if arm == 'video' and video_id in generated_video:
                    response = generated_video[video_id]
                elif args.model == 'internvl35':
                    atlases = ()
                    if arm != 'video':
                        values = np.array(logs[record['index']])
                        if not np.isfinite(values).all():
                            raise ValueError('Nonfinite EEG measurements')
                        atlases = [atlas(values[part].mean(0), weights, limits).resize(
                            (640, 192), Image.Resampling.NEAREST) for part in [slice(0, 5), slice(5, 10)]]
                    response = runtime.generate(frames=frames, atlases=atlases)
                else:
                    measurements = None
                    if arm == 'joint':
                        values = np.array(logs[record['index']])
                        if not np.isfinite(values).all():
                            raise ValueError('Nonfinite EEG measurements')
                        measurements = [values[part].mean(axis=(0, 2)).round(4).tolist()
                                        for part in [slice(0, 5), slice(5, 10)]]
                    response = runtime.generate(frames, measurements)
                if arm == 'video':
                    generated_video[video_id] = response
                terms, valid = parser(response['raw'])
                row = prediction_row(record, terms, valid, **response, frame_metadata=frame_metadata)
                predictions.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                save(out / 'status.json', dict(phase='generating', arm=arm, completed=number + 1,
                    total=644, seconds=time.time() - started, training=False, test_read=True))
        results[arm] = dict(official_ew=metric.score(predictions), exact_set=metrics([
            (set(row['required_concepts']), set(row['emotion_terms']) if row['parse_valid'] else set())
            for row in predictions]), invalid=sum(not row['parse_valid'] for row in predictions),
            unique_sets=len({tuple(sorted(row['emotion_terms'])) for row in predictions}))
        save(out / f'{arm}_metrics.json', results[arm])
    save(out / 'complete.json', dict(results=results, n=644, seconds=time.time() - started,
        training=False, test_read=True))
    save(out / 'status.json', dict(phase='complete', training=False, test_read=True))
    gpu_lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['internvl35', 'r1omni'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception:
        destination = ROOT / 'final_test_20261120' / f'{arguments.model}_zeroshot_v1'
        if destination.exists():
            save(destination / 'failure.json', dict(error=traceback.format_exc(), training=False))
            save(destination / 'status.json', dict(phase='failed', training=False))
        raise
