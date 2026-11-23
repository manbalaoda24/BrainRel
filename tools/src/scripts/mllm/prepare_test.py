"""Prepare held-out subjects with unchanged official preprocessing and display scales."""
import ast
import io
import json
import pickle
import zipfile

import numpy as np
from scipy import signal
from scipy.io import loadmat

from common import ROOT, EMOTIONS, save, sha, write_rows
from description_targets import build_target
from weak_labels import ITEMS
from run import BANDS


OUT = ROOT / 'final_test_20261120'


def main():
    OUT.mkdir(exist_ok=False)
    save(OUT / 'status.json', dict(phase='preparing', training=False, test_read=True))
    source = ROOT / 'cbramod_official_probe_v1/upstream/preprocessing/preprocessing_faced.py'
    integrity = json.loads((ROOT / 'cbramod_official_probe_v1/source_integrity.json').read_text())
    assert sha(source) == integrity['upstream_preprocessing_sha256']
    tree = ast.parse(source.read_text())

    def assignment(name):
        return next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))

    resample = compile(ast.Module(body=[assignment('eeg'), assignment('eeg_')], type_ignores=[]), str(source), 'exec')
    sample = compile(ast.Module(body=[assignment('sample')], type_ignores=[]), str(source), 'exec')
    loader = ROOT / 'cbramod_official_probe_v1/upstream/datasets/faced_dataset.py'
    assert sha(loader) == integrity['loader_sha256']
    assert '/ 100' in loader.read_text() or '/100' in loader.read_text()
    weights = np.load(ROOT / 'official200_classification_v1/topology_geometry.npz')['weights'].astype('float32')
    limits = np.load(ROOT / 'traceable_visual_alignment_v1/measurements.npz')['limits']
    old = np.load(ROOT / 'official200_classification_v1/band_logpower.npy', mmap_mode='r')
    logs = np.lib.format.open_memmap(OUT / 'band_logpower.npy', mode='w+', dtype='float32', shape=(10332, 10, 5, 30))
    logs[:8400] = old
    window = np.hanning(200)
    frequencies = np.fft.rfftfreq(200, 1 / 200)
    rows, access = [], []
    with zipfile.ZipFile(ROOT / 'files/Processed_data.zip') as archive, zipfile.ZipFile(ROOT / 'files/Data.zip') as reports:
        for subject in [80] + list(range(100, 123)):
            member = next(name for name in archive.namelist() if name.endswith(f'/sub{subject:03d}.pkl'))
            payload = archive.read(member)
            array = pickle.loads(payload)
            assert array.shape == (28, 32, 7500) and np.isfinite(array).all()
            namespace = dict(np=np, signal=signal, array=array)
            exec(resample, namespace)
            eeg = namespace['eeg_']
            assert eeg.shape == (28, 32, 30, 200)
            samples = []
            for trial in eeg:
                namespace.update(samples=trial, j=1)
                exec(sample, namespace)
                samples.append((namespace['sample'] / 100).astype('float32'))
            values = np.stack(samples)
            assert values.shape == (28, 32, 10, 200)
            uv = values[:, :30].astype('float64') * 100
            uv -= uv.mean(-1, keepdims=True)
            psd = np.abs(np.fft.rfft(uv * window, axis=-1)) ** 2 / (200 * np.sum(window ** 2))
            psd[..., 1:-1] *= 2
            power = np.stack([psd[..., (frequencies >= low) & (frequencies < high)].sum(-1)
                              for _, low, high in BANDS], -1)
            band_logs = np.log10(power + 1e-6).transpose(0, 2, 3, 1).astype('float32')
            if subject == 80:
                known_indices = [subject * 84 + trial * 3 + 1 for trial in range(28)]
                assert np.array_equal(band_logs, old[known_indices]), 'Existing validation preprocessing replay differs'
                save(OUT / 'preprocessing_replay.json', dict(subject=80, n=28, bitwise_equal=True))
                continue
            report_member = f'Data/sub{subject:03d}/After_remarks.mat'
            report_payload = reports.read(report_member)
            records = loadmat(io.BytesIO(report_payload), simplify_cells=True)['After_remark']
            assert len(records) == 28 and sorted(int(row['vid']) for row in records) == list(range(1, 29))
            by_video = {int(row['vid']): row for row in records}
            for video in range(1, 29):
                index = subject * 84 + (video - 1) * 3 + 1
                logs[index] = band_logs[video - 1]
                scores = np.asarray(by_video[video]['score'], dtype='float64')
                assert scores.shape == (12,) and np.isfinite(scores).all() and ((scores >= 0) & (scores <= 7)).all()
                concepts = [name for name in EMOTIONS if scores[ITEMS.index(name)] >= 3.5]
                row = dict(task='open', subject=subject, video_id=video, index=index, split='test',
                           required_concepts=concepts, target=dict(emotions=concepts))
                row['description_task'] = build_target(row, band_logs[video - 1], weights, limits,
                                                       [band[0] for band in BANDS])
                rows.append(row)
            import hashlib
            access.append(dict(subject=subject, eeg_member=member, eeg_sha256=hashlib.sha256(payload).hexdigest(),
                               self_report_member=report_member, self_report_sha256=hashlib.sha256(report_payload).hexdigest()))
            save(OUT / 'status.json', dict(phase='preparing', subjects_done=subject - 99, total=23, test_read=True))
    logs.flush()
    assert len(rows) == len({row['index'] for row in rows}) == 644
    write_rows(OUT / 'test_records.jsonl', rows)
    save(OUT / 'protocol.json', dict(training=False, test_read=True, subjects=list(range(100, 123)),
        n=644, segment=1, threshold=3.5, member_access=access, source_integrity=integrity,
        input_formula='Official resample and sample AST; loader /100 then float32; restore uV and unchanged Hann periodogram/log-power',
        display_limits_sha256=sha(ROOT / 'traceable_visual_alignment_v1/measurements.npz'),
        geometry_sha256=sha(ROOT / 'official200_classification_v1/topology_geometry.npz'),
        fitting='None: existing training-only display limits and geometry retained',
        cache='Middle segment only for test; other test segment positions are unused zeros',
        reference='Retrospective eight-emotion self-report >=3.5, not independent unrestricted open annotations',
        selection='Existing main-table checkpoints and prompts fixed before final test; no test selection'))
    save(OUT / 'status.json', dict(phase='prepared', n=644, test_read=True, training=False))


if __name__ == '__main__':
    main()
