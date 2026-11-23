"""CPU-only six-metric summaries from selected archived EAV predictions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def normalized(terms):
    return {term.strip().lower() for term in terms}


def exact_metrics(rows, vocabulary):
    counts = Counter()
    classes = {word: Counter() for word in vocabulary}
    jaccard = []
    for row in rows:
        truth = normalized(row['required_concepts'])
        prediction = normalized(row['emotion_terms']) if row['parse_valid'] else set()
        counts.update(tp=len(truth & prediction), fp=len(prediction - truth), fn=len(truth - prediction))
        jaccard.append(len(truth & prediction) / len(truth | prediction) if truth | prediction else 1.)
        for word, values in classes.items():
            values.update(tp=int(word in truth & prediction), fp=int(word in prediction - truth),
                          fn=int(word in truth - prediction))
    def f1(values):
        denominator = 2 * values['tp'] + values['fp'] + values['fn']
        return 2 * values['tp'] / denominator if denominator else 0.
    return dict(micro_f1=f1(counts), macro_f1=sum(f1(value) for value in classes.values()) / len(classes),
                jaccard=sum(jaccard) / len(rows), counts=dict(counts),
                per_term_f1={word: f1(value) for word, value in classes.items()})


def main(args):
    assert exact_metrics([dict(required_concepts=['joy'], emotion_terms=['joy', 'fear'], parse_valid=True)],
                         ['joy', 'fear'])['micro_f1'] == 2 / 3
    names = [
        'weak_baseline_oneemo_eeg_visual_v2', 'weak_baseline_oneemo_face_video_v2',
        'weak_baseline_qwen25vl_joint_visual_v2', 'open_weak_student_expand1120_joint_v2',
        'open_weak_asr_student_joint_v3', 'open_weak_asr_student_video_v3',
        *[f'open_curriculum_S{stage}_v4' for stage in range(1, 5)],
        'open_full_eeg_visual_v5', 'open_full_face_video_v5',
        *[f'open_full_joint_S{stage}_v5' for stage in range(2, 5)]]
    groups = defaultdict(list)
    pending = []
    for name in names:
        directory = args.root / name
        complete = directory / 'complete.json'
        if not complete.exists():
            pending.append(name)
            continue
        summary = json.loads(complete.read_text())
        epoch = summary['best_epoch']
        source = directory / f'epoch_{epoch}/records.jsonl'
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        assert rows and len({row['sample_id'] for row in rows}) == len(rows)
        assert all(25 <= row['subject'] <= 28 for row in rows)
        panel = sorted((row['sample_id'], sorted(normalized(row['required_concepts']))) for row in rows)
        digest = hashlib.sha256(json.dumps(panel).encode()).hexdigest()
        groups[digest].append(dict(name=name, epoch=epoch, rows=rows,
            source=str(source), prediction_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            wheel=summary['validation']))
    output = dict(test_read=False, new_generation=False, pending=pending, panels=[],
        exact_policy='Lowercase+trim literal sets; no synonyms/wheel mapping; invalid prediction becomes empty; empty-empty Jaccard=1; zero-division F1=0',
        macro_policy='Shared frozen union of reference and prediction vocabulary across all completed listed arms in each panel; save vocabulary for reproducibility. Not FACED eight-class macro-F1.')
    lines = ['# EAV six metrics on archived selected predictions', '',
             'All values are percentages. Separate panels have different IDs or targets and must not be pooled.',
             'Exact-set metrics do not equate synonyms. Macro-F1 uses the same recorded reference/prediction union vocabulary for every arm within a panel.', '']
    for digest, models in sorted(groups.items()):
        vocabulary = sorted({word for model in models for row in model['rows']
                             for word in normalized(row['required_concepts']) |
                             (normalized(row['emotion_terms']) if row['parse_valid'] else set())})
        group = dict(panel_sha256=digest, n=len(models[0]['rows']), vocabulary=vocabulary, models=[])
        lines += [f'## Panel {digest[:10]}: n={group["n"]}, vocabulary={len(vocabulary)}', '',
                  '| Model | Epoch | EW-P | EW-R | EW-F1 | Micro-F1 | Macro-F1 | Jaccard |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for model in models:
            exact = exact_metrics(model['rows'], vocabulary)
            result = {key: model[key] for key in ['name', 'epoch', 'source', 'prediction_sha256']}
            result.update(ew_p=model['wheel']['precision'], ew_r=model['wheel']['recall'],
                          ew_f1=model['wheel']['f1'], **exact)
            group['models'].append(result)
            values = [result[key] * 100 for key in ['ew_p', 'ew_r', 'ew_f1', 'micro_f1', 'macro_f1', 'jaccard']]
            lines.append(f'| {model["name"]} | {model["epoch"]} | ' + ' | '.join(f'{value:.2f}' for value in values) + ' |')
        output['panels'].append(group)
        lines.append('')
    args.out.mkdir(exist_ok=False)
    (args.out / 'metrics.json').write_text(json.dumps(output, indent=2))
    report = '\n'.join(lines) + '\nPending: ' + ', '.join(pending) + '\n'
    (args.out / 'RESULTS.md').write_text(report)
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
