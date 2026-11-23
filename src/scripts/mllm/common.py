"""Shared development panel and pinned wheel scoring for model baselines."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(os.environ.get('FACED_DATA_ROOT', '/data/model/datasets/FACED_syn50614194'))
PROJECT = Path(os.environ.get('FACED_PROJECT_ROOT', '/home/ubuntu/work/eeg_project'))
OUTPUT = ROOT / 'open_emotion_baselines_v1'
EMOTIONS = ['anger', 'disgust', 'fear', 'sadness', 'amusement', 'inspiration', 'joy', 'tenderness']
SEED = 20261109


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def panel():
    result = {}
    source = ROOT / 'numeric_evidence_filter_full_v1'
    for split, count in [('train', 2240), ('val', 560)]:
        rows = [json.loads(line) for line in (source / f'{split}_open_records.jsonl').read_text().splitlines()]
        assert len(rows) == count and len({row['index'] for row in rows}) == count
        for row in rows:
            assert row['task'] == 'open' and row['index'] % 3 == 1
            assert row['index'] == row['subject'] * 84 + (row['video_id'] - 1) * 3 + 1
            assert (0 <= row['subject'] < 80) if split == 'train' else (80 <= row['subject'] < 100)
            assert set(row['required_concepts']) <= set(EMOTIONS)
        result[split] = rows
    assert not {row['subject'] for row in result['train']} & {row['subject'] for row in result['val']}
    return result


def evaluator():
    sys.path.insert(0, str(PROJECT / 'scripts/faced'))
    from wheel import OfficialEW
    return OfficialEW(ROOT)


def prediction_row(record, terms, valid=True, **extra):
    return {'index': record['index'], 'subject': record['subject'], 'video_id': record['video_id'],
            'required_concepts': record['required_concepts'], 'emotion_terms': terms,
            'parse_valid': valid, **extra}


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))


def lock_gpu():
    import fcntl
    import subprocess
    gpu = int(os.environ.get('FACED_GPU_INDEX', '2'))
    if gpu < 0:
        raise ValueError('GPU index must be nonnegative')
    handle = (ROOT / f'open_baselines_gpu{gpu}.lock').open('w')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    used = int(subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.used',
                                       '--format=csv,noheader,nounits'], text=True).strip())
    if used >= 1000:
        raise RuntimeError(f'GPU{gpu} is occupied ({used} MiB); refusing to start')
    return handle
