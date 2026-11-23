"""Archive-native EAV EEG/video adapter. Reads training and validation contents only."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np

CLASSES = ['neutral', 'sadness', 'anger', 'happiness', 'calmness']
VIDEO = re.compile(r'^EAV/subject(\d+)/Video/(\d+)_Trial_(\d+)_(Listening|Speaking)_(Neutral|Sadness|Anger|Happiness|Calmness)\.mp4$', re.I)
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def label_matrix(value):
    if value.shape == (10, 200):
        value = value.T
    if value.shape != (200, 10) or not np.isin(value, [0, 1]).all() or not (value.sum(1) == 1).all():
        raise ValueError('Invalid 200-trial one-hot labels')
    return value.argmax(1)


def eeg_matrix(value):
    if value.shape != (10000, 30, 200) or not np.isfinite(value).all():
        raise ValueError(f'Unexpected actual EEG layout: {value.shape}')
    return value


def split_for(subject):
    if not 1 <= subject <= 42:
        raise ValueError('Subject outside 1..42')
    return 'train' if subject <= 28 else ('val' if subject <= 35 else 'test')


def video_ledger(names):
    ledger = {}
    for name in names:
        match = VIDEO.fullmatch(name)
        if not match:
            continue
        subject, interaction, local_trial = map(int, match.group(1, 2, 3))
        key = subject, interaction
        if key in ledger or not 1 <= interaction <= 200:
            raise ValueError(f'Duplicate or invalid global interaction: {key}')
        ledger[key] = dict(subject=subject, interaction_id=interaction, local_trial_id=local_trial,
                           task=match[4].lower(), label=match[5].lower(), archive_video=name)
    if set(ledger) != {(subject, interaction) for subject in range(1, 43) for interaction in range(1, 201)}:
        raise ValueError('Archive must provide exactly 42 x 200 uniquely aligned videos')
    return ledger


def geometry():
    definition = json.loads(Path(__file__).with_name('geometry.json').read_text())
    positions = np.array(definition['positions_mm'], dtype='float64')
    xy = positions[:, :2] / np.linalg.norm(positions[:, :2], axis=1).max()
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 16), np.linspace(1, -1, 16)), -1).reshape(-1, 2)
    mask = np.linalg.norm(grid, axis=1) <= 1
    distances = ((grid[:, None] - xy[None]) ** 2).sum(-1)
    neighbors = np.argsort(distances, axis=1)[:, :4]
    weights = np.zeros((256, 30))
    for number in range(256):
        inverse = 1 / np.maximum(distances[number, neighbors[number]], 1e-8)
        weights[number, neighbors[number]] = inverse / inverse.sum()
    weights[~mask] = 0
    return weights.astype('float32'), definition


def bandpower(value):
    from scipy.signal import resample_poly
    cropped = value[2500:7500].T
    sampled = resample_poly(cropped, 2, 5, axis=-1)
    assert sampled.shape == (30, 2000)
    windows = sampled.reshape(30, 10, 200)
    windows -= windows.mean(-1, keepdims=True)
    taper = np.hanning(200)
    frequency = np.fft.rfftfreq(200, 1 / 200)
    spectrum = abs(np.fft.rfft(windows * taper, axis=-1)) ** 2 / (200 * (taper ** 2).sum())
    spectrum[..., 1:-1] *= 2
    power = np.stack([spectrum[..., (frequency >= low) & (frequency < high)].sum(-1)
                      for low, high in BANDS], -1)
    result = np.log10(power + 1e-6).transpose(1, 2, 0).astype('float32')
    assert result.shape == (10, 5, 30) and np.isfinite(result).all()
    return result


def prepare(arguments):
    from scipy.io import loadmat
    out = arguments.output
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'status.json', {'phase': 'indexing', 'test_contents_read': False})
    with zipfile.ZipFile(arguments.archive) as archive:
        ledger = video_ledger(archive.namelist())
        save(out / 'inventory.json', {'videos': len(ledger), 'archive': str(arguments.archive),
            'subject_split': {split: [subject for subject in range(1, 43) if split_for(subject) == split]
                              for split in ['train', 'val', 'test']}})
        subjects = [1, 29] if arguments.smoke else list(range(1, 36))
        weights, definition = geometry()
        np.savez(out / 'topology_geometry.npz', weights=weights)
        save(out / 'geometry_provenance.json', definition)
        rows, values, checks = [], [], []
        for subject in subjects:
            prefix = f'EAV/subject{subject}/EEG/'
            label_name = next(name for name in archive.namelist() if name.startswith(prefix) and name.endswith('_eeg_label.mat'))
            eeg_name = next(name for name in archive.namelist() if name.startswith(prefix) and name.endswith('_eeg.mat'))
            labels = label_matrix(loadmat(io.BytesIO(archive.read(label_name)))['label'])
            data = loadmat(io.BytesIO(archive.read(eeg_name)))
            signal = eeg_matrix(data['seg1'] if 'seg1' in data else data['seg'])
            for interaction in range(1, 201):
                record = dict(ledger[subject, interaction])
                expected = 2 * CLASSES.index(record['label']) + int(record['task'] == 'speaking')
                if int(labels[interaction - 1]) != expected:
                    raise ValueError(f'EEG/video label mismatch subject={subject}, interaction={interaction}')
                if record['task'] != arguments.task and arguments.task != 'both':
                    continue
                record.update(index=len(rows), sample_id=f'EAV_S{subject:02d}_I{interaction:03d}',
                    split=split_for(subject), video_role='participant_face', window_seconds=[5, 15],
                    label_id=CLASSES.index(record['label']), required_concepts=[record['label']],
                    target={'emotions': [record['label']]}, reference_type='provided_single_category_not_independent_open_gold')
                member = PurePosixPath(record['archive_video'])
                if member.is_absolute() or '..' in member.parts:
                    raise ValueError('Unsafe archive path')
                destination = out / 'videos' / f'{record["sample_id"]}.mp4'
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes(archive.read(record['archive_video']))
                record['video_path'] = str(destination.resolve())
                values.append(bandpower(signal[:, :, interaction - 1]))
                rows.append(record)
            checks.append({'subject': subject, 'eeg_shape': list(signal.shape), 'dtype': str(signal.dtype),
                           'first_trial_rms_native': float(np.sqrt(np.mean(signal[:, :, 0] ** 2))),
                           'all_200_labels_match_video': True})
            del signal, data
            save(out / 'status.json', {'phase': 'preparing', 'last_subject': subject, 'samples': len(rows), 'test_contents_read': False})
    values = np.stack(values)
    np.save(out / 'band_logpower.npy', values)
    training = np.array([row['split'] == 'train' for row in rows])
    limits = np.quantile(values[training], [.01, .99], axis=(0, 1, 3)).astype('float32')
    if not (limits[1] > limits[0]).all():
        raise ValueError('Degenerate training display limits')
    np.savez(out / 'measurements.npz', limits=limits)
    for split in ['train', 'val']:
        (out / f'{split}_records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows if row['split'] == split))
    save(out / 'sample_checks.json', checks)
    save(out / 'prepared.json', {'phase': 'complete', 'smoke': arguments.smoke, 'task': arguments.task,
        'counts': {split: sum(row['split'] == split for row in rows) for split in ['train', 'val']},
        'classes': CLASSES, 'test_contents_read': False, 'fs_original': 500, 'fs_target': 200,
        'window_seconds': [5, 15], 'input_axis': 'time,channel,interaction', 'label_axis': 'class10,interaction',
        'units': 'Native MAT units, no assumed uV conversion; fixed display quantiles fit training only',
        'formula': 'crop [2500:7500]; polyphase 2/5; 10 one-second windows; per-window mean subtraction; Hann one-sided PSD; log10(sum band power + 1e-6)',
        'additional_filtering': None, 'standardization': None, 'bands_hz': BANDS,
        'display': '1st/99th training quantiles, clip to [0,1] only for rendered maps/nodes',
        'channel_order': definition['channel_order_source'], 'video_alignment': 'global interaction prefix, not repeating local Trial ID',
        'split': 'Custom fixed subject-independent 28/7/7, not claimed as official EAV protocol',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    save(out / 'status.json', {'phase': 'complete', 'test_contents_read': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=Path('/data/model/datasets/EAV_13799131/EAV.zip'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=['speaking', 'listening', 'both'], default='speaking')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    try:
        prepare(args)
    except Exception as error:
        if args.output.exists():
            save(args.output / 'status.json', {'phase': 'failed', 'error': str(error)})
        raise
