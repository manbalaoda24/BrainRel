"""Read-only exact-set diagnostics on archived full560 generation records."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/faced_traceable_visual_v1'
OUT = REPORT / 'alternate_metrics_v1'
LABELS = ['anger', 'disgust', 'fear', 'sadness', 'amusement', 'inspiration', 'joy', 'tenderness']


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.


def metrics(pairs):
    counts = {'tp': 0, 'fp': 0, 'fn': 0}
    jaccard, sample_f1, exact = [], [], []
    per_class = {label: {'tp': 0, 'fp': 0, 'fn': 0} for label in LABELS}
    for truth, prediction in pairs:
        true_positive, false_positive, false_negative = len(truth & prediction), len(prediction - truth), len(truth - prediction)
        for name, value in zip(['tp', 'fp', 'fn'], [true_positive, false_positive, false_negative]):
            counts[name] += value
        jaccard.append(len(truth & prediction) / len(truth | prediction) if truth | prediction else 1.)
        sample_f1.append(2 * true_positive / (len(truth) + len(prediction)) if truth or prediction else 1.)
        exact.append(truth == prediction)
        for label, counter in per_class.items():
            counter['tp'] += label in truth and label in prediction
            counter['fp'] += label not in truth and label in prediction
            counter['fn'] += label in truth and label not in prediction
    for counter in per_class.values():
        counter['f1'] = ratio(2 * counter['tp'], 2 * counter['tp'] + counter['fp'] + counter['fn'])
        counter['recall'] = ratio(counter['tp'], counter['tp'] + counter['fn'])
    return {'n': len(pairs), 'micro_precision': ratio(counts['tp'], counts['tp'] + counts['fp']),
            'micro_recall': ratio(counts['tp'], counts['tp'] + counts['fn']),
            'micro_f1': ratio(2 * counts['tp'], 2 * counts['tp'] + counts['fp'] + counts['fn']),
            'macro_f1_eight': sum(row['f1'] for row in per_class.values()) / 8,
            'sample_f1': sum(sample_f1) / len(pairs), 'jaccard': sum(jaccard) / len(pairs),
            'exact_set_match': sum(exact) / len(pairs), 'counts': counts, 'per_class': per_class}


def main():
    assert metrics([(set(), set())])['jaccard'] == 1.
    checked = metrics([({'joy'}, {'joy', 'anger'}), ({'fear'}, set())])
    assert checked['micro_f1'] == .5 and checked['jaccard'] == .25
    sources = {}
    for arm in ['eeg', 'joint']:
        sources[f'BrainRel_{arm}'] = REPORT / f'hypergraph_integrated_pilot/selected_full560_{arm}_records.jsonl'
        for model in ['oneemo', 'qwen25vl3b']:
            sources[f'{model}_{arm}'] = OUT / f'inputs/{model}/{arm}.jsonl'
    output, common_truth = {}, None
    for name, path in sources.items():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 560 and len({row['index'] for row in rows}) == 560
        truth = {row['index']: set(row['required_concepts']) for row in rows}
        assert all(value <= set(LABELS) for value in truth.values())
        if common_truth is None:
            common_truth = truth
        assert truth == common_truth, 'Mismatched evaluation items or reference labels'
        pairs = [(truth[row['index']], {term.strip().lower() for term in row['emotion_terms']}
                  if row['parse_valid'] else set()) for row in rows]
        output[name] = {'all560': metrics(pairs), 'nonempty_reference': metrics([pair for pair in pairs if pair[0]]),
                        'empty_reference_count': sum(not pair[0] for pair in pairs),
                        'empty_predictions': sum(not pair[1] for pair in pairs),
                        'mean_prediction_size': sum(len(pair[1]) for pair in pairs) / len(pairs),
                        'out_of_vocabulary_predictions': sorted(set().union(*(pair[1] for pair in pairs)) - set(LABELS)),
                        'invalid': sum(not row['parse_valid'] for row in rows),
                        'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    pairs = [(truth, set(LABELS)) for truth in common_truth.values()]
    output['mechanical_all_eight'] = {'all560': metrics(pairs),
                                     'nonempty_reference': metrics([pair for pair in pairs if pair[0]])}
    result = {'protocol': {'reference': 'Original self-report anchored eight-concept weak sets; not independent open annotations',
               'normalization': 'Lowercase and trim only, exact lexical set matching; no synonyms or emotion wheels',
               'empty_policy': 'All560 includes empty references; both-empty sample F1/Jaccard=1, exact equality counts correct',
               'micro': 'Global set TP/FP/FN; empty references can add false positives',
               'macro': 'Unweighted mean of F1 over eight reference concepts; OOVs counted in micro FP, listed separately',
               'nonempty_subset': '405 trials with nonempty raw reference, not official wheel-specific eligible subsets',
               'model_selection': 'No new model selection, no retraining and no test access'}, 'results': output}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'metrics.json').write_text(json.dumps(result, indent=2))
    lines = ['# Alternate metrics on the same full560 validation predictions', '',
             'Exact lexical matching, no wheel mapping. All560 includes 155 empty-reference trials. '
             'Both-empty sample F1/Jaccard is one. These are eight-concept diagnostics, not open-vocabulary semantic equivalence metrics.', '',
             '| Model | Micro-P | Micro-R | Micro-F1 | Macro-F1 (8) | Jaccard | Exact set |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, row in output.items():
        value = row['all560']
        columns = [value[key] * 100 for key in ['micro_precision', 'micro_recall', 'micro_f1', 'macro_f1_eight', 'jaccard', 'exact_set_match']]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{value:.2f}' for value in columns) + ' |')
    lines += ['', 'All predictions/references were checked to use identical indices and reference sets. '
              'Candidate initialization/training histories are unchanged; previous unequal total pretraining budgets still apply. '
              'See metrics.json for the 405-nonempty subset, per-class counts, source hashes and invalid/OOV outputs.']
    lines += ['', 'Important lexical bias: BrainRel outputs only the eight canonical reference terms, whereas '
              'baselines also output synonyms and related terms. Exact matching penalizes happy versus joy and sad versus sadness. '
              'In particular Qwen EEG-only emits happy/sad and therefore scores zero lexically; this does not establish zero semantic emotion ability. '
              'These metrics diagnose reference-set coverage, not independent unrestricted semantic quality. '
              'Joint BrainRel exceeds all-eight reporting on strict micro-F1, macro-F1 and Jaccard; EEG-only does not exceed it on micro/macro-F1.']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
