"""Format-only audit of existing generations; no inference or semantic rewriting."""
import collections
import json

from common import ROOT, evaluator, prediction_row, save, sha, write_rows


def parse_structured(raw):
    text = raw.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines[-1].strip() == '```':
            text = '\n'.join(lines[1:-1]).strip()
    decoder = json.JSONDecoder()
    value = None
    for offset, character in enumerate(text):
        if character not in '[{':
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
            break
        except json.JSONDecodeError:
            continue
    if value is None:
        return [], False
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict) and 'emotions' in value[0]:
        value = value[0]
    if isinstance(value, dict):
        value = value.get('emotions')
    if not isinstance(value, list):
        return [], False
    if all(isinstance(item, str) for item in value):
        terms = value
    elif all(isinstance(item, dict) and set(item) == {'emotion'}
             and isinstance(item['emotion'], str) for item in value):
        terms = [item['emotion'] for item in value]
    else:
        return [], False
    return list(dict.fromkeys(term.strip().lower() for term in terms if term.strip())), True


def main():
    root = ROOT / 'stimulus_video_qwen_pilot_v1_timed'
    assert (root / 'complete.json').exists()
    metric = evaluator()
    summary = {'test_read': False, 'training': False,
               'audit': 'Post-hoc format compatibility only; no new generation, inferred emotions or synonym changes',
               'accepted_formats': ['emotions object (optionally in a singleton list)', 'string list',
                                    'list of single-key emotion objects'],
               'source_sha256': sha(__file__), 'arms': {}}
    for arm in ['eeg', 'video', 'joint']:
        original = [json.loads(line) for line in (root / f'{arm}_records.jsonl').read_text().splitlines()]
        rows = []
        for row in original:
            terms, valid = parse_structured(row['raw'])
            if row['parse_valid']:
                assert valid and terms == row['emotion_terms'], (arm, row['index'], row['raw'], terms, row['emotion_terms'])
            rows.append(prediction_row(row, terms, valid, raw=row['raw']))
        counts = collections.Counter(tuple(sorted(row['emotion_terms'])) for row in rows)
        summary['arms'][arm] = {'official_ew': metric.score(rows),
                                'strict_ew': json.loads((root / f'{arm}_metrics.json').read_text()),
                                'unique_sets': len(counts),
                                'most_common_sets': [{'terms': list(terms), 'count': count}
                                                     for terms, count in counts.most_common(5)],
                                'format_recovered': sum(not old['parse_valid'] and new['parse_valid']
                                                        for old, new in zip(original, rows))}
        write_rows(root / f'{arm}_format_audited.jsonl', rows)
    save(root / 'format_audit.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
