"""Path-configurable inference using the unmodified archived BrainRel runtime."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(args):
    package = args.package.resolve()
    config = json.loads((package / 'configs/faced_main.json').read_text())
    data_root = args.data_root.resolve()
    source_root = args.source_root.resolve() if args.source_root else package / 'src'
    model_root = args.model.resolve()
    snapshot = json.loads((package / 'manifests/source_snapshot.json').read_text())
    for relative, expected in snapshot['files'].items():
        path = source_root / Path(relative).relative_to('src')
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Source drift: ' + relative)
    required = [config['language_checkpoint'], config['graph_checkpoint'],
                config['records'], config['band_logpower'], config['geometry'], config['display_limits']]
    for relative in required:
        if not (data_root / relative).is_file():
            raise FileNotFoundError(relative)
    locked = json.loads((package / 'manifests/training_artifacts.lock.json').read_text())['sha256']
    for relative in [config['language_checkpoint'], config['graph_checkpoint']]:
        if sha(data_root / relative) != locked[relative]:
            raise ValueError('Checkpoint drift: ' + relative)
    if not (model_root / 'config.json').is_file():
        raise FileNotFoundError('Authorized base-model config.json required')
    inputs = json.loads((package / 'manifests/inputs_and_eav.json').read_text())
    for relative, entry in inputs['faced_inputs'].items():
        if sha(data_root / relative) != entry['sha256']:
            raise ValueError('Input artifact drift: ' + relative)
    for relative, entry in inputs['base_model'].items():
        if sha(model_root / relative) != entry['sha256']:
            raise ValueError('Base model/processor drift: ' + relative)
    if args.validate_only:
        print(json.dumps(dict(validated_paths=True, checkpoint_hashes_match=True,
                             input_and_base_hashes_match=True, source_hashes_match=True, gpu_used=False,
                             note='Path validation is not a generation or full-input equivalence test')))
        return
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite inference output')
    os.environ.update(FACED_DATA_ROOT=str(data_root), FACED_PROJECT_ROOT=str(source_root),
                      FACED_ONEEMO_MODEL=str(model_root), FACED_INTEGRATED_HYPERGRAPH='1',
                      FACED_GPU_INDEX=str(args.gpu), CUDA_VISIBLE_DEVICES=str(args.gpu))
    sys.path[:0] = [str(source_root / 'scripts' / folder) for folder in [
        'mllm', 'faced', 'encoder']]
    import numpy as np
    import torch
    from common import lock_gpu, prediction_row, save
    from train_graph import HypergraphRuntime
    from experts import HypergraphRelationExperts
    from parser import parse_structured
    from exact import metrics
    from eval_video import PROMPT as VIDEO_PROMPT
    from video_train import video_frames
    from transformers.video_utils import VideoMetadata

    gpu_lock = lock_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = HypergraphRuntime(args.output, checkpoint_path=data_root / config['language_checkpoint'])
    expected_metric = json.loads((package / 'results/score_regression.json').read_text())['metric_protocol']
    if runtime.metric.protocol != expected_metric:
        raise ValueError('Metric implementation/assets drift')
    module = HypergraphRelationExperts(runtime.numeric,
        runtime.model.get_input_embeddings().weight.shape[-1]).cuda()
    state = torch.load(data_root / config['graph_checkpoint'], map_location='cpu', weights_only=False)
    module.load_state_dict(state['hypergraph'], strict=True)
    runtime.numeric = module.eval().requires_grad_(False)
    runtime.filter_class = HypergraphRelationExperts
    runtime.logs = np.load(data_root / config['band_logpower'], mmap_mode='r')
    model = runtime.model.eval().requires_grad_(False)
    model.config.use_cache = True
    records = [json.loads(line) for line in (data_root / config['records']).read_text().splitlines()]
    if len(records) != 644 or {row['subject'] for row in records} != set(range(100, 123)):
        raise ValueError('Unexpected test manifest; no automatic split fallback')
    if args.limit:
        records = records[:args.limit]
    rows, cache = [], {}
    with torch.inference_mode(), (args.output / 'predictions.jsonl').open('x') as stream:
        for record in records:
            if args.arm in ['joint', 'eeg']:
                batch = runtime.batch(record, joint=args.arm == 'joint')
            else:
                video_id = record['video_id']
                if video_id not in cache:
                    cache[video_id] = video_frames(data_root / f'stimulus_videos_v1/videos/{video_id}.mp4')
                frames, metadata = cache[video_id]
                prefix = runtime.processor.apply_chat_template([dict(role='user', content=[
                    dict(type='video'), dict(type='text', text=VIDEO_PROMPT)])],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                batch = runtime.processor(text=[prefix], videos=[frames], return_tensors='pt',
                    videos_kwargs=dict(do_sample_frames=False), video_metadata=[VideoMetadata(
                        total_num_frames=8, fps=8 / metadata['duration_seconds'],
                        duration=metadata['duration_seconds'], frames_indices=list(range(8)))]).to('cuda')
                runtime.context.clear()
                base = model.get_base_model().model
                if hasattr(base, 'rope_deltas'):
                    base.rope_deltas = None
            generated = model.generate(**batch, do_sample=False, max_new_tokens=320)
            tail = generated[0, batch['input_ids'].shape[1]:]
            raw = runtime.processor.tokenizer.decode(tail, skip_special_tokens=True).strip()
            terms, valid = parse_structured(raw)
            row = prediction_row(record, terms, valid, raw=raw, budget_reached=len(tail) == 320)
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
    pairs = [(set(row['required_concepts']), set(row['emotion_terms']) if row['parse_valid'] else set())
             for row in rows]
    save(args.output / 'metrics.json', dict(
        official_ew=runtime.metric.score(rows) if len(rows) == 644 else None,
        exact_set=metrics(pairs), partial=len(rows) != 644))
    if args.compare_archived:
        original = data_root / f'final_test_20261120/brainrel/adapted_{args.arm}_records.jsonl'
        archived = {row['index']: row for row in map(json.loads, original.read_text().splitlines())}
        failures = [row['index'] for row in rows if any(
            row[key] != archived[row['index']][key] for key in ['raw', 'emotion_terms', 'parse_valid'])]
        save(args.output / 'generation_regression.json', dict(n=len(rows), different_indices=failures,
                                                              passed=not failures))
        if failures:
            raise RuntimeError('New generation differs; archived results remain untouched')
    save(args.output / 'complete.json', dict(n=len(rows), arm=args.arm, training=False,
        full_test=len(rows) == 644, reused_source_runtime=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=['joint', 'eeg', 'video'], default='joint')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--compare-archived', action='store_true')
    arguments = parser.parse_args()
    if arguments.limit < 0 or arguments.gpu < 0:
        parser.error('limit and gpu must be nonnegative; --limit 0 means full test')
    main(arguments)
