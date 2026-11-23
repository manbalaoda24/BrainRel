"""Frozen-checkpoint component interventions, not retrained ablations."""
import argparse
import json
import time

import torch

from common import ROOT, lock_gpu, panel, prediction_row, save, sha, write_rows
from parser import parse_structured
from train_graph import HypergraphRuntime
from sparse import BudgetedRelationFilter


CONDITIONS = ('without_spatial', 'without_temporal', 'without_cross_band',
              'uniform_router', 'uniform_members', 'without_all_relations')


def install(module, condition):
    handles = []
    kinds = {'without_spatial': [0], 'without_temporal': [1],
             'without_cross_band': [2], 'without_all_relations': [0, 1, 2]}
    if condition in kinds:
        for kind in kinds[condition]:
            handles.append(module.experts[kind].register_forward_hook(
                lambda layer, inputs, output: torch.zeros_like(output)))
    elif condition == 'uniform_router':
        handles.append(module.router.register_forward_hook(
            lambda layer, inputs, output: torch.zeros_like(output)))
    elif condition == 'uniform_members':
        for scorer in module.member_scores:
            handles.append(scorer.register_forward_hook(
                lambda layer, inputs, output: torch.zeros_like(output)))
    else:
        raise ValueError(condition)
    return handles


def main(args):
    gpu_lock = lock_gpu()
    torch.set_num_threads(2)
    torch.manual_seed(20261114)
    source_dir = ROOT / f'relation_repair_{args.variant}_v1'
    out = ROOT / f'relation_components_{args.variant}_v1'
    out.mkdir(exist_ok=False)
    source = source_dir / 'best.pt'
    language = ROOT / 'v2_video_joint_lora_pilot_v1/best.pt'
    records = panel()['val']
    assert len(records) == 560
    save(out / 'status.json', {'phase': 'loading', 'test_read': False})
    runtime = HypergraphRuntime(out, checkpoint_path=language)
    module = BudgetedRelationFilter(runtime.numeric,
        runtime.model.get_input_embeddings().weight.shape[-1],
        sparse=args.variant == 'sparse').cuda()
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    module.load_state_dict(checkpoint['hypergraph'], strict=True)
    module.epoch = int(checkpoint['epoch'])
    module.eval().requires_grad_(False)
    runtime.numeric = module
    runtime.filter_class = BudgetedRelationFilter
    runtime.model.eval().requires_grad_(False)
    protocol = {'variant': args.variant, 'checkpoint': str(source), 'sha256': sha(source),
        'language_checkpoint': str(language), 'language_sha256': sha(language),
        'epoch': module.epoch, 'budgets': list(module.budgets()), 'conditions': CONDITIONS,
        'validation_trials': 560, 'arms': ['eeg', 'joint'], 'test_read': False,
        'training_updates': 0, 'interpretation': 'Inference interventions on a fixed checkpoint; not retrained architecture ablations',
        'expert_removal': 'Zero entire transformed expert output including bias; leave other router weights unchanged, without renormalizing',
        'uniform_router': 'Set router logits to zero before softmax, giving each expert weight 1/3',
        'uniform_members': 'Set member logits to zero inside existing incidence masks. Resulting edge summaries, gate scores and sparse selections may change; not a fixed-selection intervention',
        'without_all_relations': 'Keep all encoded-node bypass, slot readout, projection and bounded injection; zero only expert messages',
        'baseline': 'Reuse selected_metrics.json and selected_auxiliary_off_metrics.json from the same source directory',
        'generation': {'do_sample': False, 'max_new_tokens': 320}, 'script_sha256': sha(__file__)}
    save(out / 'protocol.json', protocol)
    write_rows(out / 'validation_records.jsonl', records)
    results = {'full': json.loads((source_dir / 'selected_metrics.json').read_text()),
               'auxiliary_off': json.loads((source_dir / 'selected_auxiliary_off_metrics.json').read_text())}
    started = time.time()
    with torch.inference_mode():
        probe = torch.randn(1, 100, 8, device='cuda')
        reference = module(probe)[0].clone()
        assert torch.equal(reference, module(probe)[0])
        checks = {}
        for condition in CONDITIONS:
            handles = install(module, condition)
            try:
                actual = module(probe)[0]
                assert torch.isfinite(actual).all()
                checks[condition] = float((actual - reference).abs().max())
                assert checks[condition] > 0
                if condition == 'uniform_router':
                    assert torch.allclose(module.trace['node_router'],
                                          torch.full_like(module.trace['node_router'], 1 / 3))
            finally:
                for handle in handles:
                    handle.remove()
            assert torch.equal(reference, module(probe)[0])
        save(out / 'preflight.json', {'max_abs_changes': checks,
            'baseline_restored_after_every_condition': True, 'budgets': list(module.budgets())})
        for condition in CONDITIONS:
            handles = install(module, condition)
            results[condition] = {}
            try:
                for arm in ['eeg', 'joint']:
                    rows = []
                    for number, record in enumerate(records):
                        batch = runtime.batch(record, joint=arm == 'joint')
                        generated = runtime.model.generate(**batch, do_sample=False, max_new_tokens=320)
                        tail = generated[0, batch['input_ids'].shape[1]:]
                        raw = runtime.processor.tokenizer.decode(tail, skip_special_tokens=True).strip()
                        terms, valid = parse_structured(raw)
                        rows.append(prediction_row(record, terms, valid, raw=raw,
                                                   budget_reached=len(tail) == 320))
                        if number % 20 == 0:
                            save(out / 'status.json', {'phase': 'evaluating', 'condition': condition,
                                'arm': arm, 'completed': number + 1, 'total': 560,
                                'seconds': time.time() - started, 'test_read': False})
                    write_rows(out / f'{condition}_{arm}_records.jsonl', rows)
                    results[condition][arm] = runtime.metric.score(rows)
                    save(out / f'{condition}_metrics.json', results[condition])
            finally:
                for handle in handles:
                    handle.remove()
            save(out / 'summary.json', results)
    save(out / 'complete.json', {'phase': 'complete', 'results': results,
        'seconds': time.time() - started, 'test_read': False, 'training_updates': 0})
    save(out / 'status.json', {'phase': 'complete', 'test_read': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['dense', 'sparse'], required=True)
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        folder = ROOT / f'relation_components_{arguments.variant}_v1'
        if folder.exists():
            save(folder / 'status.json', {'phase': 'failed', 'error': str(error), 'test_read': False})
        raise
