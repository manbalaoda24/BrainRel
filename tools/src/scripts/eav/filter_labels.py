import argparse
from collections import Counter
import json
from pathlib import Path

from build_labels import validate_decision
from generate import write_json, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args()
    diagnostic = json.loads(args.diagnostics.read_text())
    assert diagnostic['correct'] >= 9 and diagnostic['false_accepts'] == 0 and diagnostic['review_schema_failures'] == 0
    assert json.loads((args.source / 'status.json').read_text())['phase'] == 'complete'
    rows = [json.loads(path.read_text()) for path in sorted(args.source.glob('EAV_*.json'))]
    assert rows
    manifest = {row['sample_id']: row for row in json.loads((args.source / 'manifest.json').read_text())}
    assert set(manifest) == {row['sample_id'] for row in rows}
    assert all(row['split'] == ('val' if args.reference_only else 'train') for row in manifest.values())
    if args.reference_only:
        assert all(29 <= row['subject'] <= 35 for row in manifest.values())
    candidates, unknown, review_failures = [], [], []
    final_terms = {}
    retained_occurrences = 0
    for row in rows:
        sources = {source['id']: source['text'] for source in row['sources']}
        support = []
        for review in row['review_raw']:
            raw = review.get('format_repair_raw') or review['raw']
            decision, valid, keep = validate_decision(raw, review['candidate'], review['candidate_id'])
            candidate = review['candidate']
            assert candidate['source_id'] in sources and candidate['quote'] in sources[candidate['source_id']]
            if not valid:
                review_failures.append(dict(sample_id=row['sample_id'], candidate=candidate, decision=decision))
            elif keep and not set(review.get('evidence_flags', [])).intersection(
                    {'suspected_asr_requires_audio_review', 'visual_quote_lacks_specific_affect_cue'}):
                support.append(dict(**candidate, decision=decision))
        terms = list(dict.fromkeys(candidate['term'] for candidate in support))
        final_terms[row['sample_id']] = terms
        retained_occurrences += len(support)
        if terms:
            candidates.append(dict(sample_id=row['sample_id'], sources=row['sources'], support=support,
                target=dict(emotions=terms, emotion_description='Predicted emotional states: '+', '.join(terms)+'.'),
                supervision='Automatically reviewed expressed-affect weak labels; not verified inner experience or independent gold'))
        else:
            unknown.append(dict(sample_id=row['sample_id'], disposition='unknown_or_no_supported_affect',
                use_as_empty_target=False, sources=row['sources']))
    required = {'EAV_S08_I184': 'excitement', 'EAV_S07_I126': 'fear', 'EAV_S09_I104': 'happiness', 'EAV_S09_I114': 'longing'}
    regression_ids = set(required) | {'EAV_S08_I164', 'EAV_S04_I146', 'EAV_S01_I138'}
    checks = {}
    if regression_ids.issubset(final_terms):
        regression_forms = {'happiness': {'happiness', 'happy'}}
        checks = {sample: bool(set(final_terms[sample]) & regression_forms.get(term, {term}))
                  for sample, term in required.items()}
        checks.update(environment_excluded='serene' not in final_terms['EAV_S08_I164'],
                      other_person_excluded='embarrassment' not in final_terms['EAV_S04_I146'],
                      garbled_longing_excluded='longing' not in final_terms['EAV_S01_I138'])
    assert all(checks.values()) and not review_failures
    args.out.mkdir(exist_ok=False)
    for name, records in [('trial_reference.jsonl' if args.reference_only else 'trial_training_candidates.jsonl', candidates), ('unknown_quarantine.jsonl', unknown)]:
        (args.out / name).write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in records))
    write_json(args.out / 'audit.json', dict(samples=len(rows), nonempty_trials=len(candidates),
        unknown_trials=len(unknown), retained_candidate_occurrences=retained_occurrences,
        emotion_counts=dict(Counter(term for terms in final_terms.values() for term in terms)),
        review_schema_failures=len(review_failures), developer_checks=checks,
        source=str(args.source), source_complete_sha256=sha(args.source / 'complete.json'),
        finalizer_sha256=sha(Path(__file__)), validator_sha256=sha(Path(__file__).with_name('build_labels.py')),
        disposition='Passes engineering/developer gates for a small weak-supervision trial only; not independent annotation validation',
        reference_only=args.reference_only,
        limitations=['Same teacher proposes and reviews labels', 'Developer cases informed prompt changes',
                    'Incomplete candidate recall is unmeasured', 'Audio identity/synchrony unverified',
                    'Descriptions summarize terms, not rich explanatory narratives',
                    'Weak labels must not serve as independent test gold'], test_read=False))
    print((args.out / 'audit.json').read_text())


if __name__ == '__main__':
    main()
