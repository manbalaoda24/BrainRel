"""Read-only pinned AffectGPT wheel evaluator for explicit emotion lists."""
import glob
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd

from score import COMMIT, functions


class OfficialEW:
    def __init__(self, root):
        root = Path(root)
        upstream = root / 'ovmer_faced_hit_v1/upstream'
        ledger = json.loads((root / 'ovmer_faced_hit_v1/protocol.json').read_text())['asset_hashes']
        for name, expected in ledger.items():
            path = upstream.parent / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        namespace = {'np': np, 'pd': pd, 'os': os, 're': re, 'glob': glob,
                     'config': SimpleNamespace(EMOTION_WHEEL_ROOT=str(upstream / 'AffectGPT/emotion_wheel'))}
        functions(upstream / 'OV-MER/toolkit/utils/functions.py', namespace, {'string_to_list'})
        functions(upstream / 'OV-MER/toolkit/utils/read_files.py', namespace, {'func_read_key_from_csv'})
        functions(upstream / 'AffectGPT/my_affectgpt/evaluation/wheel.py', namespace)
        self.namespace = namespace
        self.format_mapping = namespace['read_format2raws']()
        self.raw_mapping = namespace['read_candidate_synonym_merge']()
        self.protocol = {'commit': COMMIT, 'mapping': 'unmodified official',
                         'asset_hashes': ledger,
                         'empty_policy': 'Original code omits empty mapped truth per wheel; report denominators',
                         'aggregation': 'Sample-average P/R, harmonic F1 per wheel, mean across five level1 wheels',
                         'known_mapping_anomalies': 'Retained for official compatibility, including previously audited fear-form conflation'}

    def score(self, rows):
        targets = {str(row['index']): row['required_concepts'] for row in rows}
        predictions = {str(row['index']): row['emotion_terms'] if row['parse_valid'] else [] for row in rows}
        assert len(targets) == len(rows) and rows
        wheels = []
        for number in range(1, 6):
            metric = f'case3_wheel{number}_level1'
            cluster = self.namespace['func_get_wheel_cluster'](f'wheel{number}', 'level1')
            precision, recall = self.namespace['calculate_openset_overlap_rate'](
                name2gt=targets, name2pred=predictions, metric=metric,
                format_mapping=self.format_mapping, raw_mapping=self.raw_mapping, inter_print=False)
            eligible = sum(bool(self.namespace['func_map_label_to_synonym'](
                terms, self.format_mapping, self.raw_mapping, cluster, metric)) for terms in targets.values())
            assert np.isfinite(precision) and np.isfinite(recall) and eligible
            wheels.append({'wheel': number, 'precision': float(precision), 'recall': float(recall),
                           'f1': float(2 * precision * recall / (precision + recall)) if precision + recall else 0.,
                           'eligible': eligible, 'omitted': len(rows) - eligible})
        return {'f1': float(np.mean([wheel['f1'] for wheel in wheels])),
                'precision': float(np.mean([wheel['precision'] for wheel in wheels])),
                'recall': float(np.mean([wheel['recall'] for wheel in wheels])),
                'n': len(rows), 'empty_truth': sum(not terms for terms in targets.values()),
                'empty_predictions': sum(not terms for terms in predictions.values()),
                'invalid': sum(not row['parse_valid'] for row in rows), 'wheels': wheels}
