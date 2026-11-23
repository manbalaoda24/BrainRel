"""Derive display tables from archived full-precision metrics; never select models."""
import argparse
import json
from pathlib import Path


def main(args):
    package = args.package.resolve()
    metadata = json.loads((package / 'results/archived_metadata.json').read_text())

    def score(path):
        value = metadata[path]['value']
        return value.get('official_ew', value)

    rows = []
    for name, label in [('direct', 'Direct joint training'), ('mixed', 'Mixed-task training'),
                        ('staged', 'Staged training')]:
        prefix = f'curriculum_ablation_fresh_v1/{name}'
        selected = metadata[prefix + '/selection_lock.json']['value']['selected_update']
        rows.append(dict(name=label, selected_update=selected,
            joint=score(prefix + '/selected_test644_joint_metrics.json')['f1'],
            eeg=score(prefix + '/selected_test644_eeg_metrics.json')['f1']))
    rows.append(dict(name='BrainRel: full curriculum (reference)', selected_update=None,
        joint=score('final_test_20261120/brainrel/adapted_joint_metrics.json')['f1'],
        eeg=score('final_test_20261120/brainrel/adapted_eeg_metrics.json')['f1']))
    lines = [r'\begin{table}[!t]', r'\centering\footnotesize',
        r'\caption{Training-Strategy Comparison on the FACED Test Set (\%)}',
        r'\label{tab:training_strategy}', r'\begin{tabular}{lrr}', r'\toprule',
        r'Training Strategy & Joint EW-F1 & EEG-only EW-F1 \\', r'\midrule']
    for number, row in enumerate(rows):
        if number == 3:
            lines.append(r'\midrule')
        label = r'BrainRel: full curriculum$^{\dagger}$' if number == 3 else row['name']
        lines.append(f"{label} & {row['joint'] * 100:.2f} & {row['eeg'] * 100:.2f} " + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\par\smallskip\scriptsize\raggedright',
        'All results use 644 test trials and development-selected checkpoints. '
        'The first three runs were each allocated 150 updates and 1,200 presentations. '
        'Mixed and staged runs share the complete task-batch multiset and planned per-module budgets, '
        'but their selected update-100 prefixes have different per-module update counts. '
        'The staged checkpoint precedes final relation refinement. '
        r'$^{\dagger}$The historical full curriculum has different initialization history and cumulative budget; '
        'it is a performance reference, not a budget-matched ablation.', r'\end{table}', ''])
    content = '\n'.join(lines)
    if args.check:
        expected = package / 'results/training_strategy.tex'
        if not expected.is_file() or expected.read_text() != content:
            raise SystemExit('Derived table drift detected')
        print('Derived table matches locked full-precision sources')
    else:
        output = package / 'results/training_strategy.tex'
        if output.exists():
            raise FileExistsError('Refusing to overwrite a derived table; use --check')
        output.write_text(content)
        print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--check', action='store_true')
    main(parser.parse_args())
