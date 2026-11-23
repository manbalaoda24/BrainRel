"""Developer smoke followed by fixed zero-shot ERV test inference; no training."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from common import ROOT, PROJECT, lock_gpu, save


OUT = ROOT / 'erv_zeroshot_v1'
TEST = ROOT / 'final_test_20261120'
PROMPT = ('Infer possible emotions experienced by the participant, using unrestricted emotion vocabulary. '
          'The supplied video depicts the stimulus, not the participant. '
          'Output the thinking process in <think> </think> and final emotions in <answer> </answer> tags. '
          'Keep reasoning brief. In the answer, use comma-separated emotion expressions; [] is allowed. '
          'Do not invent a unique physiological cause for an emotion.')


def parse(raw):
    answers = re.findall(r'<answer>\s*(.*?)\s*</answer>', raw, flags=re.DOTALL | re.IGNORECASE)
    if answers:
        if len(answers) != 1:
            return [], False
        answer = answers[0].strip()
        if not answer:
            return [], False
        if answer.startswith(('[', '{')):
            try:
                value = json.loads(answer)
            except json.JSONDecodeError:
                return [], False
            terms = value.get('emotions') if isinstance(value, dict) else value
        else:
            if re.search(r'[<>.!?:]', answer):
                return [], False
            terms = re.split(r'\s*(?:[,;|\n]|\band\b)\s*', answer, flags=re.IGNORECASE)
        if isinstance(terms, list) and all(isinstance(term, str) and term.strip() for term in terms):
            terms = list(dict.fromkeys(term.strip().strip('\"\'').lower() for term in terms))
            return (terms, True) if all(terms) or not terms else ([], False)
        return [], False
    if re.search(r'</?answer\b', raw, flags=re.IGNORECASE):
        return [], False
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE)
    decoder = json.JSONDecoder()
    for position, character in enumerate(raw):
        if character != '{':
            continue
        try:
            value, _ = decoder.raw_decode(raw[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get('emotions'), list):
            terms = value['emotions']
            if all(isinstance(term, str) and term.strip() for term in terms):
                return list(dict.fromkeys(term.strip().lower() for term in terms)), True
    return [], False


def load_frame_reader():
    source = PROJECT / 'scripts/mllm/video_train.py'
    tree = ast.parse(source.read_text())
    function = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == 'video_frames')
    import cv2
    import numpy as np
    from common import sha
    namespace = dict(cv2=cv2, np=np, sha=sha)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['video_frames']


def main(gpu, smoke_only=False):
    from importlib.metadata import version
    if version('peft') != '0.13.2':
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                        '--cache-dir', str(OUT / 'pip_cache'), 'peft==0.13.2'], check=True, timeout=300)
        with (OUT / 'pip_freeze.txt').open('w') as stream:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    save(OUT / 'inference_status.json', dict(phase='waiting_for_existing_evaluations', training=False, test_read=False))
    deadline = time.monotonic() + 48 * 3600
    jobs = json.loads((TEST / 'analysis/manifest.json').read_text())['jobs']
    while True:
        if all((TEST / 'analysis' / job['name'] / 'complete.json').exists() for job in jobs):
            break
        failures = [path for path in (TEST / 'analysis').glob('worker_gpu*.json')
                    if json.loads(path.read_text()).get('phase') in ['failed', 'stopped_before_analysis']]
        if failures:
            raise RuntimeError('Existing evaluation worker failed; refusing automatic GPU use')
        if time.monotonic() >= deadline:
            raise TimeoutError('Existing evaluation queue did not complete within 48 hours; no GPU acquired')
        time.sleep(60)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), FACED_GPU_INDEX=str(gpu))
    gpu_lock = lock_gpu()
    import numpy as np
    import torch
    assets = json.loads((OUT / 'assets.json').read_text())
    os.environ['ERV_BERT_PATH'] = assets['bert']['path']
    sys.path.insert(0, str(OUT / 'runtime_source'))
    from humanomni import model_init, mm_infer
    from transformers import BertTokenizer
    torch.set_num_threads(4)
    torch.manual_seed(20261109)
    save(OUT / 'inference_status.json', dict(phase='loading', training=False, test_read=False))
    model, processor, tokenizer = model_init(assets['model']['path'],
        model=dict(mm_vision_tower=assets['siglip']['path'], mm_audio_tower=assets['whisper']['path'], model_size='0.5B'),
        use_flash_attn=False, device_map={'': 0})
    model.eval().requires_grad_(False)
    bert_tokenizer = BertTokenizer.from_pretrained(assets['bert']['path'], local_files_only=True)
    reader = load_frame_reader()
    frame_cache = {}

    def generate(record, arm, logs):
        video_id = record['video_id']
        if video_id not in frame_cache:
            frames, metadata = reader(ROOT / f'stimulus_videos_v1/videos/{video_id}.mp4')
            frame_cache[video_id] = (processor['video'](frames), metadata)
        tensor, metadata = frame_cache[video_id]
        prompt = PROMPT
        if arm == 'joint':
            values = np.array(logs[record['index']])
            measurements = [values[part].mean(axis=(0, 2)).round(4).tolist()
                            for part in [slice(0, 5), slice(5, 10)]]
            prompt += (' EEG log-band-power channel means for consecutive five-second windows '
                '(delta, theta, alpha, beta, gamma): ' + json.dumps(measurements)
                + '. These measurements alone do not uniquely establish emotions.')
        raw = mm_infer(tensor, prompt, model=model, tokenizer=tokenizer, audio=None,
            modal='video', question=prompt, bert_tokeni=bert_tokenizer,
            model_size='0.5B', do_sample=False, max_new_tokens=512)
        terms, valid = parse(raw)
        return dict(index=record['index'], subject=record['subject'], video_id=video_id,
            raw=raw, emotion_terms=terms, parse_valid=valid, frame_metadata=metadata)

    train_path = ROOT / 'numeric_evidence_filter_full_v1/train_open_records.jsonl'
    development = [json.loads(line) for line in train_path.read_text().splitlines()[:2]]
    if len(development) != 2 or not all(0 <= record['subject'] < 80 for record in development):
        raise ValueError('Smoke inputs must be existing training examples')
    train_logs = np.load(ROOT / 'official200_classification_v1/band_logpower.npy', mmap_mode='r')
    smoke = [dict(generate(record, arm, train_logs), arm=arm)
             for arm in ['joint', 'video'] for record in development]
    save(OUT / 'development_smoke.json', dict(records=smoke, training=False, test_read=False))
    if not all(row['parse_valid'] for row in smoke):
        raise RuntimeError('Development smoke generated no valid JSON or native answer span; test evaluation not started')
    if smoke_only:
        save(OUT / 'inference_status.json', dict(phase='smoke_passed', training=False, test_read=False))
        gpu_lock.close()
        return
    from common import evaluator
    from exact import metrics
    metric = evaluator()
    records = [json.loads(line) for line in (TEST / 'test_records.jsonl').read_text().splitlines()]
    if len(records) != 644 or {record['subject'] for record in records} != set(range(100, 123)):
        raise ValueError('Unexpected fixed test panel')
    logs = np.load(TEST / 'band_logpower.npy', mmap_mode='r')
    destination = TEST / 'erv_05b'
    if destination.exists() and any(destination.glob('*_records.jsonl')):
        raise FileExistsError('Existing test predictions must not be overwritten')
    destination.mkdir(exist_ok=True)
    save(destination / 'protocol.json', dict(source_commit=json.loads((OUT / 'prepared.json').read_text())['source_commit'],
        assets=assets, prompt=PROMPT, input='8 fixed stimulus frames; joint adds EEG measurement text',
        arms=['joint', 'video'], audio=False, adaptation='zero-shot', do_sample=False,
        max_new_tokens=512, n=644, training=False, checkpoint_selection=False,
        parser='Single closed native answer span takes priority; otherwise JSON emotions outside think spans. '
               'Native answers accept unrestricted comma/semicolon/and-separated expressions or JSON lists; '
               'no extraction from reasoning, no label remapping. Invalid outputs become empty predictions.',
        format_repair_only=True, prompt_changed=True,
        prompt_change='Use native ERV answer tags and the official default 512-token generation cap; '
                      'verified only on training examples before test access'))
    results = {}
    started = time.time()
    for arm in ['joint', 'video']:
        rows, video_outputs = [], {}
        with (destination / f'zeroshot_{arm}_records.jsonl').open('x') as stream:
            for position, record in enumerate(records):
                if arm == 'video' and record['video_id'] in video_outputs:
                    row = dict(video_outputs[record['video_id']], index=record['index'], subject=record['subject'])
                else:
                    row = generate(record, arm, logs)
                    if arm == 'video':
                        video_outputs[record['video_id']] = row
                row['required_concepts'] = record['required_concepts']
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                if position % 20 == 0 or position == len(records) - 1:
                    save(OUT / 'inference_status.json', dict(phase='generating', arm=arm, completed=position + 1,
                        total=644, seconds=time.time() - started, training=False, test_read=True))
        pairs = [(set(row['required_concepts']), set(row['emotion_terms']) if row['parse_valid'] else set())
                 for row in rows]
        results[arm] = dict(official_ew=metric.score(rows), exact_set=metrics(pairs),
                           invalid=sum(not row['parse_valid'] for row in rows))
        save(destination / f'zeroshot_{arm}_metrics.json', results[arm])
    save(destination / 'complete.json', dict(results=results, n=644, training=False, test_read=True))
    save(OUT / 'inference_status.json', dict(phase='complete', training=False, test_read=True))
    gpu_lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    try:
        main(args.gpu, args.smoke_only)
    except Exception:
        save(OUT / 'inference_failure.json', dict(error=traceback.format_exc(), training=False))
        raise
