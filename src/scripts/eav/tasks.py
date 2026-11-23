"""Training-pool attribute supervision from the exact fixed-scale atlas panels."""
import numpy as np


def attribute_records(rows, cache):
    logs = np.load(cache / 'band_logpower.npy', mmap_mode='r')
    weights = np.load(cache / 'topology_geometry.npz')['weights']
    limits = np.load(cache / 'measurements.npz')['limits']
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    result = {'train': [], 'val': []}
    for row in rows:
        values = np.asarray(logs[row['index']])
        panels = np.stack([(values[part].mean(0) @ weights.T).reshape(5, 16, 16)
                           for part in [slice(0, 5), slice(5, 10)]])
        displayed = np.clip((panels - limits[0, :, None, None]) /
                            (limits[1] - limits[0])[:, None, None], 0, 1)
        displayed = np.rint(displayed * 255) / 255
        band = row['index'] % 5
        if (row['index'] // 5) % 2 == 0:
            part = (row['index'] // 10) % 2
            gap = float(displayed[part, band, :8].mean() - displayed[part, band, 8:].mean())
            allowed = ['UPPER_HIGHER', 'LOWER_HIGHER', 'SIMILAR']
            question = f'In the {"first" if part == 0 else "second"} atlas, compare upper and lower image-half mean {bands[band]} displayed power.'
        else:
            gap = float(displayed[1, band].mean() - displayed[0, band].mean())
            allowed = ['INCREASE', 'DECREASE', 'SIMILAR']
            question = f'Compare whole-panel mean {bands[band]} displayed power in the second atlas against the first atlas.'
        answer = 'SIMILAR' if abs(gap) <= .03 else allowed[0 if gap > 0 else 1]
        prompt = ('Two fixed-scale EEG atlases show seconds 5-10 then 10-15. '
                  'Red means higher displayed power; upper/lower are image regions, not anatomical lobes. '
                  + question + ' Treat absolute differences at most 0.03 on the normalized 0-1 scale as SIMILAR. '
                  + 'Return only JSON {"answer":"one label"}. Allowed labels: ' + ', '.join(allowed) + '.')
        enriched = {**row, 'target': {'answer': answer}, 'task_prompt': prompt,
                    'allowed_answers': allowed, 'measurement_gap': gap}
        result['train' if row['subject'] <= 24 else 'val'].append(enriched)
    assert len(result['train']) == 2400 and len(result['val']) == 400
    return result
