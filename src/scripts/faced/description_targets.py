"""Deterministic descriptions from displayed measurements and rating anchors."""
import hashlib

import numpy as np


def build_target(record, values, weights, limits, bands):
    assert values.shape == (10, 5, 30)
    key = f"{record['index']}:description-v1"
    band = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 5
    panels = []
    for part in [slice(0, 5), slice(5, 10)]:
        measured = (values[part].mean(0) @ weights.T).reshape(5, 16, 16)
        panels.append(np.clip((measured - limits[0, :, None, None]) /
                              (limits[1] - limits[0])[:, None, None], 0, 1))
    first, second = panels
    spatial_gap = float(first[band, :8].mean() - first[band, 8:].mean())
    temporal_gap = float(second[band].mean() - first[band].mean())
    spatial = 'SIMILAR' if abs(spatial_gap) <= .03 else ('UPPER_HIGHER' if spatial_gap > 0 else 'LOWER_HIGHER')
    temporal = 'SIMILAR' if abs(temporal_gap) <= .05 else ('INCREASE' if temporal_gap > 0 else 'DECREASE')
    spatial_text = {'SIMILAR': 'similar displayed means in its upper and lower image halves',
                    'UPPER_HIGHER': 'a higher displayed mean in its upper image half',
                    'LOWER_HIGHER': 'a higher displayed mean in its lower image half'}[spatial]
    temporal_text = {'SIMILAR': 'remains similar', 'INCREASE': 'increases', 'DECREASE': 'decreases'}[temporal]
    observation = (f'The {bands[band]} panel has {spatial_text} in seconds 0-5. '
                   f'Its full-map displayed mean {temporal_text} from seconds 0-5 to 5-10.')
    terms = record['target']['emotions']
    description = ('The predicted emotional states are ' + ', '.join(terms) + '.') if terms else (
        'No emotional state is selected; this does not establish neutrality.')
    return {'band': band, 'spatial': spatial, 'temporal': temporal,
            'spatial_gap': spatial_gap, 'temporal_gap': temporal_gap,
            'observation': observation, 'emotion_description': description,
            'provenance': 'Observation: fixed-scale displayed measurement rules; emotion prose: unchanged rating-derived terms. No emotion-causal claim.'}
