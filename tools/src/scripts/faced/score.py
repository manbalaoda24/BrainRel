"""Pinned AffectGPT HIT on existing FACED validation responses; CPU only."""
import ast
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace
import urllib.request

import numpy as np
import pandas as pd


ROOT = Path('/data/model/datasets/FACED_syn50614194')
OUT = ROOT / 'ovmer_faced_hit_v1'
COMMIT = '95b9257758df29fa6481c138252cdd317cce1735'
BASE = f'https://raw.githubusercontent.com/zeroQiaoba/AffectGPT/{COMMIT}/'
NAMES = ['anger', 'disgust', 'fear', 'sadness', 'neutral', 'amusement', 'inspiration', 'joy', 'tenderness']


def save(name, value):
    temporary = (OUT / name).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(OUT / name)


def fetch(relative):
    destination = OUT / 'upstream' / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    for attempt in range(4):
        try:
            payload = urllib.request.urlopen(BASE + relative, timeout=90).read()
            destination.write_bytes(payload)
            return destination
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2)


def functions(path, namespace, allowed=None):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and (allowed is None or node.name in allowed)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def main():
    if OUT.exists():
        if not (OUT / 'status.json').is_file() or json.loads((OUT / 'status.json').read_text()).get('phase') != 'failed' or (OUT / 'complete.json').exists():
            raise RuntimeError('Refusing to overwrite active or completed run')
        (OUT / 'status.json').rename(OUT / f'failure_{time.time_ns()}.json')
    OUT.mkdir(exist_ok=True)
    save('status.json', {'phase': 'fetching_pinned_metric_assets'})
    assets = [fetch('AffectGPT/emotion_wheel/' + name) for name in ['format.csv', 'synonym.xlsx', 'wheel1.xlsx', 'wheel2.xlsx', 'wheel3.xlsx', 'wheel4.xlsx', 'wheel5.xlsx']]
    wheel_path = fetch('AffectGPT/my_affectgpt/evaluation/wheel.py')
    utility_path = fetch('OV-MER/toolkit/utils/functions.py')
    read_path = fetch('OV-MER/toolkit/utils/read_files.py')
    namespace = {'np': np, 'pd': pd, 'os': os, 're': re, 'glob': glob, 'config': SimpleNamespace(EMOTION_WHEEL_ROOT=str(assets[0].parent))}
    functions(utility_path, namespace, {'string_to_list'})
    functions(read_path, namespace, {'func_read_key_from_csv'})
    functions(wheel_path, namespace)
    format_mapping = namespace['read_format2raws']()
    raw_mapping = namespace['read_candidate_synonym_merge']()
    original_path = ROOT / 'ovmer_generation_pilot_v1/records.jsonl'
    extraction_path = ROOT / 'ovmer_qwen25_extraction_v1/explicit_prediction.jsonl'
    validation_path = ROOT / 'readout_recipe_2x2_v1/full_borrowed/best_validation_predictions.npz'
    original = [json.loads(line) for line in original_path.read_text().splitlines()]
    extracted = [json.loads(line) for line in extraction_path.read_text().splitlines()]
    validation = np.load(validation_path)
    assert all(80 <= int(subject) < 100 for subject in validation['subject'])
    truth = {int(index): NAMES[int(label)] for index, label in zip(validation['index'], validation['label'])}
    assert len(truth) == len(validation['index'])
    assert len(original) == len(extracted) == 108
    for source, result in zip(original, extracted):
        assert (source['name'], source['condition'], source['response']) == (result['name'], result['condition'], result['response'])
        assert result['parse_valid'] and source['index'] in truth
    protocol = {'metric_commit': COMMIT, 'metric_execution': 'Unmodified function definitions compiled from pinned official wheel.py; dependency function definitions only, to avoid unused model imports', 'test_read': False, 'training': False, 'unit': '36 selected validation source questions per condition; may repeat segment/trial; not full validation or independent trial benchmark', 'ground_truth': 'Stored validation dataset label, NOT classifier prediction; original class-index order from classifier implementation', 'class_index_order': NAMES, 'input_hashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [original_path, extraction_path, validation_path]}, 'asset_hashes': {str(path.relative_to(OUT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in assets + [wheel_path, utility_path, read_path]}, 'scope': 'Single-label support diagnostic; outside-nine predictions are not independently verified', 'empty_handling': 'All extraction outputs valid here; official unmappable ground-truth filtering retained and explicitly counted'}
    save('protocol.json', protocol)
    output = {}
    for condition in sorted({row['condition'] for row in original}):
        subset = [(source, result) for source, result in zip(original, extracted) if source['condition'] == condition]
        variants = {'generated': {source['name']: result['labels'] for source, result in subset}, 'classifier_label_only': {source['name']: [source['classifier_prior_supplied']] if source['classifier_prior_supplied'] else [] for source, _ in subset}, 'all_nine_reference': {source['name']: NAMES for source, _ in subset}}
        targets = {source['name']: truth[source['index']] for source, _ in subset}
        results = {}
        for variant, predictions in variants.items():
            if variant == 'classifier_label_only' and condition == 'evidence_only':
                continue
            per_wheel = []
            for wheel in range(1, 6):
                metric = f'case3_wheel{wheel}_level1'
                hit, mscore = namespace['calculate_openset_onehot_hitrate'](name2gt=targets, name2pred=predictions, metric=metric, format_mapping=format_mapping, raw_mapping=raw_mapping, inter_print=False)
                mapping = namespace['func_get_wheel_cluster'](f'wheel{wheel}', 'level1')
                mapped = lambda labels: set(namespace['func_map_label_to_synonym']([label.lower().strip() for label in labels], format_mapping, raw_mapping, mapping, metric))
                counts, hits, per_class = {}, [], {}
                unmapped, total_terms = 0, 0
                for name, target in targets.items():
                    target_set, prediction_set = mapped([target]), mapped(predictions[name])
                    for term in predictions[name]:
                        total_terms += 1
                        unmapped += not bool(mapped([term]))
                    counts.setdefault(target, {'total': 0, 'eligible': 0})['total'] += 1
                    if not target_set:
                        continue
                    counts[target]['eligible'] += 1
                    matched = int(bool(target_set & prediction_set))
                    hits.append(matched)
                    per_class.setdefault(target, []).append(matched)
                assert hits and abs(float(np.mean(hits)) - hit) < 1e-10
                recalls = {label: float(np.mean(values)) for label, values in per_class.items()}
                per_wheel.append({'wheel': wheel, 'hit': float(hit), 'mscore': float(mscore), 'eligible': len(hits), 'class_counts': counts, 'per_class_hit': recalls, 'macro_hit_present_classes': float(np.mean(list(recalls.values()))), 'unmapped_prediction_terms': unmapped, 'prediction_terms': total_terms})
            results[variant] = {'official_five_wheel_mean_hit': float(np.mean([row['hit'] for row in per_wheel])), 'official_five_wheel_mean_mscore': float(np.mean([row['mscore'] for row in per_wheel])), 'mean_macro_hit_present_classes': float(np.mean([row['macro_hit_present_classes'] for row in per_wheel])), 'mean_raw_set_size': float(np.mean([len(set(labels)) for labels in predictions.values()])), 'empty_sets': sum(not labels for labels in predictions.values()), 'wheels': per_wheel}
        output[condition] = {'n': len(subset), 'unique_segments': len({source['index'] for source, _ in subset}), 'ground_truth_counts': {label: sum(target == label for target in targets.values()) for label in NAMES}, 'variants': results}
    save('complete.json', output)
    save('status.json', {'phase': 'complete'})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        if OUT.exists():
            save('status.json', {'phase': 'failed', 'traceback': traceback.format_exc()})
        raise
