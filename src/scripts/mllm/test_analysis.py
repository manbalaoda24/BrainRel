"""Predeclared final-test replay of existing FACED analysis configurations."""
import argparse
from contextlib import nullcontext
import itertools
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from peft import set_peft_model_state_dict

from common import ROOT, lock_gpu, prediction_row, save, sha, write_rows
from parser import parse_structured
from eval_modalities import batch_for, PROMPT
from eval_relations import install, CONDITIONS as COMPONENTS
from eval_rms import CapController, CONDITIONS as CAPS
from eval_masks import MaskedRelations, mask_sets
from exact import metrics
from train_graph import HypergraphRuntime
from sparse import BudgetedRelationFilter


BASE = ROOT / 'final_test_20261120'
OUT = BASE / 'analysis'
LANGUAGE = ROOT / 'v2_video_joint_lora_pilot_v1/best.pt'
SPARSE = 'relation_repair_sparse_v1'


def make_manifest():
    OUT.mkdir(exist_ok=False)
    jobs = []

    def add(name, kind, conditions, checkpoint=SPARSE, native=False):
        source = ROOT / checkpoint / 'best.pt'
        assert source.is_file(), source
        jobs.append(dict(name=name, kind=kind, conditions=conditions, checkpoint=str(source),
                         checkpoint_sha256=sha(source), native=native))

    add('modalities', 'modality', [{'name': combination, 'combination': combination}
        for combination in map(''.join, itertools.product('01', repeat=3))])
    add('masking', 'mask', [dict(name=name, combination='111')
        for name in ['full', 'mask_selected'] + [f'random_{seed}' for seed in range(5)]])
    add('rms', 'cap', [dict(name=name, ratio=ratio, combination='111') for name, ratio in CAPS.items()])
    add('budgets', 'budget', [dict(name=name, budget=budget, combination='111') for name, budget in
        {'half': [3, 5, 1], 'original': [5, 9, 2], 'double': [10, 18, 4], 'dense': [50, 100, 20]}.items()])
    for kind, expert in enumerate(['spatial', 'temporal', 'cross_band']):
        add('gain_' + expert, 'gain', [dict(name=f'gain_{gain:g}', kind=kind, gain=gain, combination='111')
            for gain in [0., .5, 1., 1.5, 2.]])
    for variant in ['dense', 'sparse']:
        add('components_' + variant, 'component', [dict(name=f'{condition}_{arm}',
            intervention=condition, combination=combination)
            for condition in ['full', 'auxiliary_off'] + list(COMPONENTS)
            for arm, combination in [('eeg', '101'), ('joint', '111')]], 'relation_repair_' + variant + '_v1')
    for checkpoint in ['relation_seed_20261112_v2', 'relation_seed_20261113_v2', 'relation_seed_20261114_v2',
                       'relation_data_quarter_v2', 'relation_data_half_v2']:
        add(checkpoint, 'readout', [dict(name=arm, combination=combination)
            for arm, combination in [('eeg', '101'), ('joint', '111')]], checkpoint)
    for model in ['brainrel', 'oneemo']:
        checkpoint = 'v2_hypergraph_experts_pilot_v1' if model == 'brainrel' else 'native_mllm_adaptation_v1/oneemo'
        add('paired_swaps_' + model, 'swap', [dict(name=arm, combination='111' if model == 'brainrel' else '110')
            for arm in ['original', 'swap_video', 'swap_eeg']], checkpoint, native=model == 'oneemo')
    result = dict(jobs=jobs, n_per_condition=644, conditions=sum(len(job['conditions']) for job in jobs),
        training=False, test_read=True, language=str(LANGUAGE), language_sha256=sha(LANGUAGE),
        source_sha256=sha(__file__), prompt=PROMPT, selection='None; report all previously configured conditions',
        checkpoint_boundary='Main table and sparse analysis use different pre-existing checkpoints and prompts; preserve separate controls',
        seed_boundary='Only existing branch-adaptation seeds, not independent full-curriculum seeds',
        data_boundary='Existing 25/50/100-percent runs have unequal updates; no retraining or test-dependent selection',
        swaps='Full644; next test participant cyclically, same stimulus; or next video cyclically, fixed EEG',
        traces='Masking/full saves original routing, pooled contents, messages and responses for temporal/qualitative analyses')
    save(OUT / 'manifest.json', result)
    print(json.dumps(dict(jobs=len(jobs), conditions=result['conditions'], inputs=result['conditions'] * 644)))


def run_job(job):
    gpu_lock = lock_gpu()
    torch.set_num_threads(2)
    torch.manual_seed(20261114)
    out = OUT / job['name']
    out.mkdir(exist_ok=False)
    save(out / 'status.json', dict(phase='loading', test_read=True, training=False))
    source = Path(job['checkpoint'])
    assert sha(source) == job['checkpoint_sha256']
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert sha(LANGUAGE) == manifest['language_sha256']
    records = [json.loads(line) for line in (BASE / 'test_records.jsonl').read_text().splitlines()]
    assert len(records) == 644 and {row['subject'] for row in records} == set(range(100, 123))
    if job['native']:
        from joint_runtime import V2VideoRuntime
        runtime = V2VideoRuntime(out, native_model='oneemo')
        state = torch.load(source, map_location='cpu', weights_only=False)
        loaded = set_peft_model_state_dict(runtime.model, state['lora'])
        assert not loaded.unexpected_keys
        module = None
    else:
        runtime = HypergraphRuntime(out, checkpoint_path=LANGUAGE)
        if job['kind'] == 'swap':
            from experts import HypergraphRelationExperts
            module_class = HypergraphRelationExperts
            options = {}
        else:
            module_class = MaskedRelations if job['kind'] == 'mask' else BudgetedRelationFilter
            options = dict(sparse='dense' not in job['checkpoint'])
        module = module_class(runtime.numeric, runtime.model.get_input_embeddings().weight.shape[-1], **options).cuda()
        state = torch.load(source, map_location='cpu', weights_only=False)
        module.load_state_dict(state['hypergraph'], strict=True)
        module.epoch = int(state['epoch'])
        runtime.numeric, runtime.filter_class = module.eval().requires_grad_(False), module_class
    runtime.logs = np.load(BASE / 'band_logpower.npy', mmap_mode='r')
    runtime.model.eval().requires_grad_(False)
    runtime.model.config.use_cache = True
    actual_prompt = ('Original runtime.batch measurement-query prompt' if job['kind'] in ['component', 'readout'] else PROMPT)
    save(out / 'protocol.json', dict(**job, training=False, test_read=True, n=644, prompt=actual_prompt,
        metric=runtime.metric.protocol, manifest_sha256=sha(OUT / 'manifest.json'),
        panel_sha256=sha(BASE / 'test_records.jsonl'), generation=dict(do_sample=False, max_new_tokens=320)))
    lookup = {(row['subject'], row['video_id']): row for row in records}
    traces, contents = [], []
    started, results = time.time(), {}
    controller = CapController() if job['kind'] == 'cap' else None
    if controller:
        runtime.filter_class = controller
    with torch.inference_mode():
        for condition in job['conditions']:
            name = condition['name']
            handles = []
            if job['kind'] == 'component' and condition['intervention'] not in ['full', 'auxiliary_off']:
                handles = install(module, condition['intervention'])
            if job['kind'] == 'gain':
                gain = condition['gain']
                handles = [module.experts[condition['kind']].register_forward_hook(
                    lambda layer, inputs, output, gain=gain: output * gain)]
            if job['kind'] == 'budget':
                module.budgets = lambda budget=tuple(condition['budget']): budget
            if controller:
                controller.ratio = condition['ratio']
            rows = []
            try:
                with (out / f'{name}_records.jsonl').open('x') as stream:
                    for number, record in enumerate(records):
                        supplied = dict(record)
                        extra = {}
                        if job['kind'] == 'swap' and name == 'swap_video':
                            supplied['video_id'] = record['video_id'] % 28 + 1
                        if job['kind'] == 'swap' and name == 'swap_eeg':
                            donor = lookup[100 + (record['subject'] - 100 + 1) % 23, record['video_id']]
                            supplied['index'] = donor['index']
                            extra['donor_index'] = donor['index']
                        if job['kind'] == 'mask':
                            module.drop_ids = []
                            batch_for(runtime, supplied, condition['combination'])
                            module(runtime.context['nodes'])
                            choices, trace = mask_sets(module, record['index'])
                            if name == 'full':
                                traces.append(dict(index=record['index'], subject=record['subject'],
                                    video_id=record['video_id'], required_concepts=record['required_concepts'],
                                    trace=trace, masks=choices, gates=module.trace['gates'][0].cpu().tolist(),
                                    node_router=module.trace['node_router'][0].cpu().tolist()))
                                contents.append({key: value[0].cpu().numpy() for key, value in module.content_trace.items()})
                            module.drop_ids = choices[name]
                            extra['mask_ids'] = choices[name]
                        if job['kind'] in ['component', 'readout']:
                            batch = runtime.batch(supplied, joint=condition['combination'] == '111')
                            if condition.get('intervention') == 'auxiliary_off':
                                runtime.context.clear()
                        else:
                            batch = batch_for(runtime, supplied, condition['combination'])
                        assert 'labels' not in batch
                        generated = runtime.model.generate(**batch, do_sample=False, max_new_tokens=320)
                        tail = generated[0, batch['input_ids'].shape[1]:]
                        raw = runtime.processor.tokenizer.decode(tail, skip_special_tokens=True).strip()
                        terms, valid = parse_structured(raw)
                        if controller:
                            extra.update(controller.diagnostics)
                        row = prediction_row(record, terms, valid, raw=raw, budget_reached=len(tail) == 320, **extra)
                        rows.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                        stream.flush()
                        save(out / 'status.json', dict(phase='generating', condition=name, completed=number + 1,
                            total=644, conditions_complete=len(results), conditions_total=len(job['conditions']),
                            seconds=time.time() - started, training=False, test_read=True))
                pairs = [(set(row['required_concepts']), set(row['emotion_terms']) if row['parse_valid'] else set()) for row in rows]
                results[name] = dict(official_ew=runtime.metric.score(rows), exact_set=metrics(pairs),
                                    invalid=sum(not row['parse_valid'] for row in rows))
                save(out / f'{name}_metrics.json', results[name])
                if job['kind'] == 'mask' and name == 'full':
                    write_rows(out / 'baseline_traces.jsonl', traces)
                    np.savez_compressed(out / 'baseline_content.npz', indices=[row['index'] for row in records],
                        **{key: np.stack([entry[key] for entry in contents]) for key in contents[0]})
                    save(out / 'edge_schema.json', module.describe())
            finally:
                for handle in handles:
                    handle.remove()
    save(out / 'complete.json', dict(results=results, seconds=time.time() - started, n=644, test_read=True, training=False))
    save(out / 'status.json', dict(phase='complete', test_read=True, training=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--job')
    args = parser.parse_args()
    if args.prepare:
        make_manifest()
    else:
        jobs = json.loads((OUT / 'manifest.json').read_text())['jobs']
        job = next(entry for entry in jobs if entry['name'] == args.job)
        try:
            run_job(job)
        except Exception:
            save(OUT / job['name'] / 'failure.json', dict(error=traceback.format_exc(), training=False, test_read=True))
            raise
