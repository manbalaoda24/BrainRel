"""Full-training-set open-format adaptation with single-category EAV targets."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from generate import parse_answer, prompt_for, sha, write_json


def configure_trainable(runtime, stage):
    runtime.numeric.requires_grad_(False)
    for name, parameter in runtime.model.named_parameters():
        parameter.requires_grad_(('lora_' in name and stage not in ['S2', 'S4']) or
                                 (stage == 'S0' and name in runtime.language_source['merger']))
    if stage == 'S2':
        runtime.numeric.requires_grad_(True)
        runtime.numeric.member_scores.requires_grad_(False)
        runtime.numeric.edge_gates.requires_grad_(False)
    elif stage == 'S4':
        for module in [runtime.numeric.experts, runtime.numeric.router,
                       runtime.numeric.member_scores, runtime.numeric.edge_gates]:
            module.requires_grad_(True)
    runtime.numeric_training = stage in ['S2', 'S4']
    return [parameter for module in [runtime.model, runtime.numeric]
            for parameter in module.parameters() if parameter.requires_grad]


def evaluate(runtime, records, metric, directory, status, epoch):
    import torch
    directory.mkdir()
    rows = []
    with torch.inference_mode(), (directory / 'records.jsonl').open('x') as stream:
        for number, record in enumerate(records):
            batch = runtime.batch(record, arm=runtime.open_arm, prompt=record.get('task_prompt', prompt_for(runtime.open_arm)))
            generated = runtime.model.generate(**batch, do_sample=False, max_new_tokens=160)
            tail = generated[0, batch['input_ids'].shape[1]:]
            raw = runtime.processor.tokenizer.decode(tail, skip_special_tokens=True).strip()
            if runtime.curriculum_stage == 'S0':
                try:
                    parsed = json.loads(raw)
                    valid = isinstance(parsed, dict) and parsed.get('answer') in record['allowed_answers']
                    terms = [parsed['answer']] if valid else []
                except (ValueError, TypeError):
                    terms, valid = [], False
            else:
                terms, valid = parse_answer(raw)
            row = dict(index=record['index'], sample_id=record['sample_id'], subject=record['subject'],
                interaction_id=record['interaction_id'], required_concepts=record['target'].get('emotions', [record['target'].get('answer')]),
                emotion_terms=terms, parse_valid=valid, raw=raw, budget_reached=len(tail) == 160)
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            write_json(status, dict(phase='validating', epoch=epoch, completed=number + 1,
                total=len(records), test_read=False))
    if runtime.curriculum_stage == 'S0':
        accuracy = float(np.mean([row['parse_valid'] and row['emotion_terms'] == row['required_concepts'] for row in rows]))
        scores = dict(selection_score=accuracy, exact_accuracy=accuracy,
                      valid=sum(row['parse_valid'] for row in rows), total=len(rows), test_read=False)
        write_json(directory / 'metrics.json', scores)
        return scores
    scores = metric.score(rows)
    scores['selection_score'] = scores['f1']
    scores.update(unique_emotion_sets=len({tuple(sorted(row['emotion_terms'])) for row in rows}),
        mean_terms=float(np.mean([len(row['emotion_terms']) for row in rows])),
        literal_reference_recall=float(np.mean([len(set(row['required_concepts']) & set(row['emotion_terms'])) / max(1, len(row['required_concepts'])) for row in rows])),
        budget_reached=sum(row['budget_reached'] for row in rows))
    scores['mechanical_all_five'] = metric.score([
        {**row, 'emotion_terms': runtime.protocol['classes'], 'parse_valid': True} for row in rows])
    write_json(directory / 'metrics.json', scores)
    return scores


def main(args):
    import fcntl
    lock = (args.cache.parent / f'eav_gpu{args.gpu}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    used = int(subprocess.check_output(['nvidia-smi', f'--id={args.gpu}',
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if used >= 1000:
        raise RuntimeError('GPU occupied; refusing to train')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from peft import get_peft_model_state_dict
    from runtime import EAVRuntime
    if args.native_visual:
        from visual_runtime import VisualBaselineRuntime
        EAVRuntime = VisualBaselineRuntime
    if args.native_visual or args.disable_numeric:
        original_prompt_for = globals()['prompt_for']
        globals()['prompt_for'] = lambda arm: original_prompt_for(arm).replace(
            ', with a numerical EEG residual', '').replace(', a numerical EEG residual,', ',')
    from wheel import OfficialEW
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = args.out / 'status.json'
    write_json(status, dict(phase='loading_model', test_read=False))
    records = {split: [json.loads(line) for line in (args.cache / f'{split}_records.jsonl').read_text().splitlines()]
        for split in ['train', 'val']}
    assert len(records['train']) == 2800 and len(records['val']) == 700
    assert {row['subject'] for row in records['train']} == set(range(1, 29))
    assert {row['subject'] for row in records['val']} == set(range(29, 36))
    for split, rows in records.items():
        assert len({row['sample_id'] for row in rows}) == len(rows)
        assert all(row['split'] == split and row['target'] == {'emotions': [row['label']]} for row in rows)
    if args.curriculum_stage == 'S0':
        from tasks import attribute_records
        records = attribute_records(records['train'], args.cache)
    elif args.weak_release:
        weak = [json.loads(line) for line in args.weak_release.read_text().splitlines()]
        lookup = {row['sample_id']: row for row in records['train']}
        assert len(weak) == len({row['sample_id'] for row in weak})
        assert all(row['sample_id'] in lookup and row['target']['emotions'] for row in weak)
        matched = [{**lookup[row['sample_id']], 'target': {'emotions': row['target']['emotions']}}
                   for row in weak]
        records = {'train': [row for row in matched if row['subject'] <= 24],
                   'val': [row for row in matched if row['subject'] >= 25]}
        if args.weak_dev_release:
            fixed = {row['sample_id']: row for row in
                     [json.loads(line) for line in args.weak_dev_release.read_text().splitlines()]
                     if lookup[row['sample_id']]['subject'] >= 25}
            available = {row['sample_id']: row for row in records['val']}
            assert fixed and set(fixed) <= set(available)
            assert all(fixed[name]['target']['emotions'] == available[name]['target']['emotions'] for name in fixed)
            records['val'] = [available[name] for name in sorted(fixed)]
        assert records['train'] and records['val']
        assert not {row['subject'] for row in records['train']} & {row['subject'] for row in records['val']}
        for split, rows in records.items():
            (args.out / f'{split}_weak_records.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in rows))
    runtime = EAVRuntime(args.cache, args.model, args.language_checkpoint, args.graph_checkpoint)
    if args.disable_numeric and not args.native_visual:
        original_batch = runtime.batch
        def visual_batch(*values, **options):
            batch = original_batch(*values, **options)
            runtime.context.clear()
            return batch
        runtime.batch = visual_batch
    runtime.open_arm = args.arm
    runtime.curriculum_stage = args.curriculum_stage
    parameters = configure_trainable(runtime, args.curriculum_stage)
    assert parameters
    if args.curriculum_stage:
        for split, rows in records.items():
            (args.out / f'{split}_curriculum_records.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in rows))
    steps_per_epoch = args.updates_per_epoch or math.ceil(len(records['train']) / args.accumulate)
    train_probe = []
    if args.train_probe_ids:
        probe_ids = json.loads(args.train_probe_ids.read_text())
        train_lookup = {row['sample_id']: row for row in records['train']}
        assert len(probe_ids) == len(set(probe_ids)) and set(probe_ids) <= set(train_lookup)
        train_probe = [train_lookup[name] for name in probe_ids]
    total_steps = steps_per_epoch * args.epochs
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    protocol.update(prompt=prompt_for(args.arm), training_count=len(records['train']), validation_count=len(records['val']),
        target='Existing single category as emotions singleton; no invented open labels or descriptions',
        language_sha256=sha(args.language_checkpoint), graph_sha256=sha(args.graph_checkpoint),
        train_records_sha256=sha(args.cache / 'train_records.jsonl'),
        val_records_sha256=sha(args.cache / 'val_records.jsonl'), script_sha256=sha(Path(__file__)),
        total_updates=total_steps, selection='Highest validation category-reference EW-F1; ties keep earlier epoch; epoch0 eligible',
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
        frozen='Base language, native vision and merger, numerical module',
        loss='Assistant answer and end suffix CE; prompt masked; no classifier prior or true label in prompt',
        schedule='5% linear warmup then cosine decay to zero', test_read=False,
        limitation='Not FACED eight-dimensional self-report supervision or independent open-vocabulary gold',
        model_mode='Eval mode retained for deterministic frozen backbone; LoRA dropout is zero')
    protocol.update(numerical_injection=args.arm != 'video' and not (args.native_visual or args.disable_numeric),
        inherited_language_checkpoint_loaded=not args.native_visual,
        initialization=('Native checkpoint with fresh Q/V LoRA' if args.native_visual
            else 'Inherited task-adapted language checkpoint'),
        comparison='Matched samples, epochs, LR, seed and update count; architectures and prior adaptation may differ')
    if args.weak_release:
        protocol.update(target='Filtered teacher emotion sets only; no description supervision',
            weak_release_sha256=sha(args.weak_release),
            selection='Highest weak-development EW-F1; epoch0 eligible; ties keep earlier epoch',
            limitation='Training-subject internal development only: subjects1-24 fit,25-28 development. Teacher-derived labels are not independent gold. Official validation/test not evaluated.',
            sources_in_prompt=False)
        if args.updates_per_epoch:
            protocol['sampling'] = 'Seeded shuffled cycles over the fit pool, truncated at the fixed update budget; no target rewriting'
        if args.train_probe_ids:
            protocol.update(train_probe_ids_sha256=sha(args.train_probe_ids),
                train_probe_count=len(train_probe), train_probe_selection_used=False)
        if args.weak_dev_release:
            protocol.update(weak_dev_release_sha256=sha(args.weak_dev_release),
                development_panel='Unchanged original ASR-only v3 development IDs and targets; extra development-pool weak labels excluded')
    if args.curriculum_stage:
        protocol.update(curriculum_stage=args.curriculum_stage,
            frozen='All parameters except the explicit trainable-name manifest',
            selection='S0 exact attribute accuracy; S1-S4 weak-development official EW-F1; epoch0 eligible',
            trainable_names=[name for prefix, module in [('language', runtime.model), ('numeric', runtime.numeric)]
                             for name, parameter in module.named_parameters() if parameter.requires_grad],
            initialization='Sequential adaptation from preceding selected checkpoint; not an equal-budget ablation')
        if args.curriculum_stage == 'S0':
            protocol.update(target='Deterministic displayed-map upper/lower or first/last comparison; threshold 0.03',
                sources_in_prompt=False, limitation='Training-pool subjects1-24 fit,25-28 development; no emotion target; no official validation or test')
    write_json(args.out / 'protocol.json', protocol)
    metric = OfficialEW(args.faced_root)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0., eps=1e-6)
    warmup = max(1, round(total_steps * .05))
    history, updates = [], []

    def save(path, epoch, scores):
        state = dict(runtime.language_source)
        state['lora'] = {name: value.detach().cpu() for name, value in get_peft_model_state_dict(runtime.model).items()}
        named_parameters = dict(runtime.model.named_parameters())
        state['merger'] = {name: named_parameters[name].detach().cpu() for name in runtime.language_source['merger']}
        state['eav_open_protocol'] = protocol
        state['eav_open_epoch'] = epoch
        state['eav_open_validation'] = scores
        temporary = path.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(path)
        graph_path = path.with_name(path.name.replace('language', 'graph'))
        temporary_graph = graph_path.with_suffix('.tmp')
        torch.save(dict(hypergraph={name: value.detach().cpu() for name, value in runtime.numeric.state_dict().items()},
                        protocol=protocol, epoch=epoch), temporary_graph)
        temporary_graph.replace(graph_path)

    scores = evaluate(runtime, records['val'], metric, args.out / 'epoch_0', status, 0)
    best_epoch, best_score = 0, scores['selection_score']
    save(args.out / 'best_language.pt', 0, scores)
    history.append(dict(epoch=0, validation=scores))
    if train_probe:
        history[-1]['train_probe'] = evaluate(runtime, train_probe, metric,
            args.out / 'train_probe_0', status, 0)
    write_json(args.out / 'history.json', history)
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(records['train']))
        if args.updates_per_epoch:
            required = steps_per_epoch * args.accumulate
            cycles = [order]
            while sum(len(cycle) for cycle in cycles) < required:
                cycles.append(rng.permutation(len(records['train'])))
            order = np.concatenate(cycles)[:required]
        epoch_losses = []
        for offset in range(0, len(order), args.accumulate):
            selected = order[offset:offset + args.accumulate]
            step = len(updates) + 1
            factor = step / warmup if step <= warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
            for group in optimizer.param_groups:
                group['lr'] = args.lr * factor
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in selected:
                row = records['train'][int(index)]
                batch = runtime.batch(row, arm=args.arm, answer=True, prompt=row.get('task_prompt', prompt_for(args.arm)))
                supervised = batch['labels'][batch['labels'] != -100]
                if not supervised.numel():
                    raise ValueError('Empty assistant supervision')
                if step == 1 and not losses:
                    write_json(args.out / 'supervision_audit.json', dict(
                        supervised_text=runtime.processor.tokenizer.decode(supervised),
                        target=row['target'], prompt=prompt_for(args.arm)))
                loss = runtime.model(**batch, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite CE')
                (loss / len(selected)).backward()
                losses.append(float(loss.detach()))
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(gradient) or gradient <= 0:
                raise ValueError('Invalid LoRA gradient')
            if step == 1:
                assert not any(parameter.grad is not None for parameter in runtime.model.parameters() if not parameter.requires_grad)
                assert not any(parameter.grad is not None for parameter in runtime.numeric.parameters() if not parameter.requires_grad)
                numeric_gradient = sum(float(parameter.grad.float().square().sum()) for parameter in runtime.numeric.parameters() if parameter.grad is not None) ** .5
                if runtime.numeric_training and numeric_gradient <= 0:
                    raise ValueError('Numerical training has no gradient')
                write_json(args.out / 'gpu_preflight.json', dict(phase='passed', gradient_norm=float(gradient),
                    frozen_gradients=False, numeric_gradient_norm=numeric_gradient, test_read=False))
            optimizer.step()
            epoch_losses.extend(losses)
            updates.append(dict(step=step, epoch=epoch, loss=float(np.mean(losses)),
                gradient_norm=float(gradient), lr=args.lr * factor))
            write_json(args.out / 'updates.json', updates)
            write_json(status, dict(phase='training', epoch=epoch, step=step, total_steps=total_steps,
                loss=updates[-1]['loss'], elapsed_seconds=time.monotonic() - started, test_read=False))
        optimizer.zero_grad(set_to_none=True)
        scores = evaluate(runtime, records['val'], metric, args.out / f'epoch_{epoch}', status, epoch)
        history.append(dict(epoch=epoch, train_ce=float(np.mean(epoch_losses)), validation=scores))
        if train_probe:
            history[-1]['train_probe'] = evaluate(runtime, train_probe, metric,
                args.out / f'train_probe_{epoch}', status, epoch)
        write_json(args.out / 'history.json', history)
        if scores['selection_score'] > best_score:
            best_epoch, best_score = epoch, scores['selection_score']
            save(args.out / 'best_language.pt', epoch, scores)
        save(args.out / 'last_language.pt', epoch, scores)
    write_json(args.out / 'complete.json', dict(phase='complete', best_epoch=best_epoch,
        validation=history[best_epoch]['validation'], total_updates=len(updates),
        elapsed_seconds=time.monotonic() - started, test_read=False))
    write_json(status, dict(phase='complete', best_epoch=best_epoch, test_read=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--arm', choices=['eeg', 'joint', 'video'], required=True)
    parser.add_argument('--native-visual', action='store_true')
    parser.add_argument('--disable-numeric', action='store_true')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--accumulate', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--weak-release', type=Path)
    parser.add_argument('--weak-dev-release', type=Path)
    parser.add_argument('--updates-per-epoch', type=int)
    parser.add_argument('--train-probe-ids', type=Path)
    parser.add_argument('--curriculum-stage', choices=['S0', 'S1', 'S2', 'S3', 'S4'])
    parser.add_argument('--model', default='/home/ubuntu/work/eeg_project/oneemo/weights')
    parser.add_argument('--faced-root', type=Path, default=Path('/data/model/datasets/FACED_syn50614194'))
    parser.add_argument('--language-checkpoint', type=Path, default=Path('/data/model/datasets/EAV_13799131/brainrel_lora_joint_pilot_v1/last_language.pt'))
    parser.add_argument('--graph-checkpoint', type=Path, default=Path('/data/model/datasets/FACED_syn50614194/v2_hypergraph_experts_pilot_v1/best.pt'))
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.accumulate < 1 or arguments.lr <= 0 or (arguments.updates_per_epoch is not None and arguments.updates_per_epoch < 1):
        parser.error('Invalid training budget')
    preexisting = arguments.out.exists()
    try:
        main(arguments)
    except Exception as error:
        if arguments.out.exists() and not preexisting:
            write_json(arguments.out / 'status.json', dict(phase='failed', error=str(error), test_read=False))
        raise
