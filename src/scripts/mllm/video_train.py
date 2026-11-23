"""Frozen visual-frame stimulus pilot; validation only, with no classifier prior."""
import json
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import VideoMetadata

from common import ROOT, OUTPUT, evaluator, lock_gpu, panel, prediction_row, save, sha
from eval_mllm import parse_answer


DESTINATION = ROOT / 'stimulus_video_qwen_pilot_v1_timed'
PROMPT = ('Infer the emotional states the participant may experience from the supplied inputs. '
          'Return only JSON {"emotions": ["emotion expression", ...]}. '
          'Use any appropriate emotion words and any number of emotions; an empty list is allowed. '
          'Video depicts the stimulus, not the participant. If present, the two five-band EEG maps '
          'summarize consecutive five-second windows; red denotes higher power and blue lower power. '
          'Do not invent events absent from the inputs or claim a unique physiological cause.')


def video_frames(path):
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if count < 8 or fps <= 0:
            raise ValueError(f'Invalid video metadata: {path}')
        indices = np.linspace(0, count - 1, 8).round().astype(int).tolist()
        frames = []
        decoded_indices = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                fallback = max(0, index - max(1, round(fps * .25)))
                capture.set(cv2.CAP_PROP_POS_FRAMES, fallback)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f'Frame {index} and bounded fallback unavailable: {path}')
            decoded_indices.append(max(0, round(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            scale = min(448 / width, 448 / height, 1.)
            frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
            frames.append(frame)
        return np.stack(frames), {'frame_indices': indices, 'source_fps': fps,
                                 'decoded_frame_indices': decoded_indices,
                                 'decoder_timestamps_seconds': [index / fps for index in decoded_indices],
                                 'fallback_rule': 'On failed seek/read, retry once 0.25 seconds earlier; otherwise fail',
                                 'source_frame_count': count, 'duration_seconds': count / fps,
                                 'timestamps_seconds': [index / fps for index in indices],
                                 'video_sha256': sha(path)}
    finally:
        capture.release()


def main():
    gpu_lock = lock_gpu()
    DESTINATION.mkdir(exist_ok=True)
    if (DESTINATION / 'complete.json').exists():
        raise RuntimeError('Pilot already complete; refusing to overwrite')
    started = time.time()
    save(DESTINATION / 'status.json', {'phase': 'loading', 'test_read': False})
    torch.set_num_threads(4)
    records = [record for record in panel()['val'] if record['subject'] in [80, 81]]
    assert len(records) == 56
    metric = evaluator()
    from run import atlas
    model_path = OUTPUT / 'assets/qwen25vl3b'
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={'': 'cuda'},
        attn_implementation='sdpa', local_files_only=True).eval().requires_grad_(False)
    protocol = {'prompt': PROMPT, 'training': False, 'lora': False, 'test_read': False,
                'subjects': [80, 81], 'trials_per_arm': 56,
                'arms': ['eeg', 'video', 'joint'], 'do_sample': False, 'max_new_tokens': 160,
                'video_input': 'Native video placeholder with eight uniformly sampled RGB frames; no audio',
                'video_cache': '28 unique clips reused across the two subjects; not individual decoding',
                'timing': 'Whole-clip frames versus middle ten-second EEG; not temporally synchronized',
                'video_id_alignment': 'Numbered filenames assumed to match official video index; semantic alignment unverified',
                'comparison': 'New simplified shared prompt; not an isolated comparison to previous prompt results',
                'visual_budget': 'Maps 640x192; frames bounded to448px longest side; joint uses more visual tokens',
                'reference': 'Eight self-report concepts threshold3.5; not independent open annotation',
                'metric': metric.protocol, 'source_sha256': sha(__file__),
                'model_revision': '66285546d2b821cf421d4f5eb2576359d3770cd3',
                'panel_indices': [record['index'] for record in records]}
    save(DESTINATION / 'protocol.json', protocol)
    limits = np.load(ROOT / 'traceable_visual_alignment_v1/measurements.npz')['limits']
    weights = np.load(ROOT / 'official200_classification_v1/topology_geometry.npz')['weights'].astype('float32')
    logs = np.load(ROOT / 'official200_classification_v1/band_logpower.npy', mmap_mode='r')
    cache = {}
    frame_cache = {}
    scores = {}
    for arm in ['eeg', 'video', 'joint']:
        record_path = DESTINATION / f'{arm}_records.jsonl'
        rows = [json.loads(line) for line in record_path.read_text().splitlines()] if record_path.exists() else []
        assert [row['index'] for row in rows] == [row['index'] for row in records[:len(rows)]]
        if arm == 'video':
            cache.update({row['video_id']: {key: row[key] for key in [
                'raw', 'emotion_terms', 'parse_valid', 'generated_tokens', 'budget_reached',
                'input_tokens', 'second_per_grid_ts', 'image_grid', 'video_grid']} for row in rows})
        with record_path.open('a') as stream:
            for number, record in enumerate(records):
                if number < len(rows):
                    continue
                video_id = record['video_id']
                cached = arm == 'video' and video_id in cache
                if cached:
                    result = dict(cache[video_id])
                else:
                    content, kwargs = [], {}
                    if arm in ['eeg', 'joint']:
                        values = np.array(logs[record['index']])
                        images = [atlas(values[part].mean(0), weights, limits).resize(
                            (640, 192), Image.Resampling.NEAREST) for part in [slice(0, 5), slice(5, 10)]]
                        content.extend([{'type': 'image'}, {'type': 'image'}])
                        kwargs['images'] = images
                    if arm in ['video', 'joint']:
                        if video_id not in frame_cache:
                            frames, metadata = video_frames(ROOT / f'stimulus_videos_v1/videos/{video_id}.mp4')
                            frame_cache[video_id] = (frames, metadata)
                            save(DESTINATION / f'video_{video_id}_sampling.json', metadata)
                        content.append({'type': 'video'})
                        frames, metadata = frame_cache[video_id]
                        kwargs['videos'] = [frames]
                        kwargs['videos_kwargs'] = {'do_sample_frames': False}
                        kwargs['video_metadata'] = [VideoMetadata(
                            total_num_frames=8, fps=8 / metadata['duration_seconds'],
                            duration=metadata['duration_seconds'], frames_indices=list(range(8)))]
                    content.append({'type': 'text', 'text': PROMPT})
                    text = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                                         tokenize=False, add_generation_prompt=True)
                    batch = processor(text=[text], return_tensors='pt', **kwargs).to('cuda')
                    with torch.inference_mode():
                        generated = model.generate(**batch, do_sample=False, max_new_tokens=160)
                    tail = generated[0, batch['input_ids'].shape[1]:]
                    raw = processor.decode(tail, skip_special_tokens=True)
                    terms, valid = parse_answer(raw)
                    result = {'raw': raw, 'emotion_terms': terms, 'parse_valid': valid,
                              'generated_tokens': len(tail), 'budget_reached': len(tail) == 160,
                              'input_tokens': batch['input_ids'].shape[1],
                              'second_per_grid_ts': (batch['second_per_grid_ts'].tolist()
                                                     if 'second_per_grid_ts' in batch else None),
                              'image_grid': batch['image_grid_thw'].tolist() if 'image_grid_thw' in batch else [],
                              'video_grid': batch['video_grid_thw'].tolist() if 'video_grid_thw' in batch else []}
                    if arm == 'video':
                        cache[video_id] = result
                row = prediction_row(record, result['emotion_terms'], result['parse_valid'],
                                     **{key: value for key, value in result.items()
                                        if key not in ['emotion_terms', 'parse_valid']}, reused_video=cached)
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                save(DESTINATION / 'status.json', {'phase': 'generating', 'arm': arm, 'completed': number + 1,
                                                 'total': 56, 'seconds': time.time() - started, 'test_read': False})
        scores[arm] = metric.score(rows)
        save(DESTINATION / f'{arm}_metrics.json', scores[arm])
    save(DESTINATION / 'complete.json', {'scores': scores, 'seconds': time.time() - started, 'test_read': False})
    save(DESTINATION / 'status.json', {'phase': 'complete', 'test_read': False})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if DESTINATION.exists() and not (DESTINATION / 'complete.json').exists():
            save(DESTINATION / 'status.json', {'phase': 'failed', 'error': repr(error), 'test_read': False})
        traceback.print_exc()
        raise
