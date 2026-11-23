"""Fresh-start direct/mixed/staged BrainRel curriculum comparison."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

from common import ROOT, PROJECT, panel, prediction_row, save, sha, write_rows


SUITE = ROOT / 'curriculum_ablation_fresh_v1'
SEED = 20261123
TASKS = ['reading', 'emotion', 'numeric', 'joint', 'relation']
ACTIVE = dict(reading=['lora', 'merger'], emotion=['lora'], numeric=['core', 'relation'],
              joint=['lora'], relation=['relation'], direct=['lora', 'merger', 'core', 'relation'])
RATES = dict(lora=1e-5, merger=1e-5, core=1e-3, relation=1e-4)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def displayed(values, weights, limits):
    import numpy as np
    maps = (values @ weights.T).reshape(10, 5, 16, 16)
    return np.clip((maps - limits[0][None, :, None, None]) /
                   (limits[1] - limits[0])[None, :, None, None], 0, 1)


def reading_records(rows, logs, weights, limits, groups, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    candidates = {(task, band): [] for task in ['space', 'time'] for band in range(5)}
    for row in rows:
        maps = displayed(np.array(logs[row['index']]), weights, limits)
        for window in range(10):
            for band in range(5):
                current = maps[window, band]
                gap = float(current[:8].mean() - current[8:].mean())
                if abs(gap) >= .06:
                    candidates['space', band].append((row['index'], window, gap))
                if window:
                    gap = float(current.mean() - maps[window - 1, band].mean())
                    if abs(gap) >= .08:
                        candidates['time', band].append((row['index'], window, gap))
    result = []
    for (task, band), pool in candidates.items():
        assert len(pool) >= groups, (task, band, len(pool), groups)
        for selected in rng.choice(len(pool), groups, replace=False):
            index, window, gap = pool[int(selected)]
            if task == 'space':
                options = [('identity', 'UPPER_HIGHER' if gap > 0 else 'LOWER_HIGHER'),
                           ('flip', 'LOWER_HIGHER' if gap > 0 else 'UPPER_HIGHER'), ('symmetric', 'SIMILAR')]
            else:
                options = [('identity', 'INCREASE' if gap > 0 else 'DECREASE'),
                           ('swap', 'DECREASE' if gap > 0 else 'INCREASE'), ('same', 'SIMILAR')]
            for operation, target in options:
                result.append(dict(index=index, subject=index // 84, task=task, band=band,
                    window=window, operation=operation, target=target))
    return result


def prepare():
    import numpy as np
    SUITE.mkdir(exist_ok=False)
    datasets = panel()
    train = read(ROOT / 'v2_video_joint_lora_pilot_v1/train_records.jsonl')
    dev = [row for row in datasets['val'] if row['subject'] in [80, 81]]
    assert len(train) == 240 and len(dev) == 56
    assert {row['subject'] for row in train} <= set(range(80))
    assert all(row in datasets['train'] for row in train)
    logs = np.load(ROOT / 'official200_classification_v1/band_logpower.npy', mmap_mode='r')
    weights = np.load(ROOT / 'official200_classification_v1/topology_geometry.npz')['weights'].astype('float32')
    limits = np.load(ROOT / 'traceable_visual_alignment_v1/measurements.npz')['limits']
    reading = reading_records(train, logs, weights, limits, 8, SEED)
    reading_dev = reading_records(dev, logs, weights, limits, 2, SEED + 1)
    assert len(reading) == 240 and len(reading_dev) == 60
    assert {row['subject'] for row in reading} <= set(range(80))
    for name, rows in [('train', train), ('development', dev), ('reading_train', reading), ('reading_development', reading_dev)]:
        write_rows(SUITE / f'{name}.jsonl', rows)
    rng = np.random.default_rng(SEED)
    staged = []
    for task in TASKS:
        order = rng.permutation(240).tolist()
        for offset in range(0, 240, 8):
            staged.append(dict(task=task, rows=order[offset:offset + 8]))
    mixed = [staged[int(index)] for index in rng.permutation(150)]
    direct = []
    for repeat in range(5):
        order = rng.permutation(240).tolist()
        direct.extend(dict(task='direct', rows=order[offset:offset + 8]) for offset in range(0, 240, 8))
    assert sorted(json.dumps(row, sort_keys=True) for row in mixed) == sorted(json.dumps(row, sort_keys=True) for row in staged)
    for arm, schedule in [('direct', direct), ('mixed', mixed), ('staged', staged)]:
        assert len(schedule) == 150 and all(len(step['rows']) == 8 for step in schedule)
        write_rows(SUITE / f'{arm}_schedule.jsonl', schedule)
    save(SUITE / 'prepared.json', dict(seed=SEED, train_trials=240, development_trials=56,
        reading_train=240, reading_development=60, updates=150, presentations=1200,
        initialization='Released OneEmo + fresh rank16 language LoRA + fresh integrated hypergraph; native merger unadapted',
        task_counts_mixed_staged=dict(Counter(step['task'] for step in staged)), active=ACTIVE, rates=RATES,
        mixed_staged_same_batches=True, reading_source='Same240training trials only; transformed display comparisons, not emotion or physiology labels',
        reading_labels='Display mean upper-minus-lower threshold0.03; query-minus-reference threshold0.05; regions are image halves',
        selection='Joint development56 EW-F1 at updates0/50/100/150; earliest ties, fixed before evaluation',
        comparison='Update/presentation-matched, NOT GPU-hour/FLOP matched. B/C also match per-group update counts and learning-rate exposure. Direct updates all groups150times.',
        module_updates_direct={key: 150 for key in RATES}, module_updates_mixed_staged=dict(lora=90, merger=30, core=30, relation=60),
        limitation='Compact reconstructed curriculum, not exact replay or full-budget ablation of historical main checkpoint; follow-up on previously inspected test split',
        script_sha256=sha(__file__), test_read=False))


def run(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), FACED_GPU_INDEX=str(args.gpu),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from common import lock_gpu
    gpu_lock = lock_gpu()
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from joint_runtime import V2VideoRuntime
    from nodes import NumericEvidenceFilter
    from experts import HypergraphRelationExperts
    from parser import parse_structured
    from exact import metrics
    out = SUITE / args.arm
    out.mkdir(exist_ok=False)
    save(out / 'status.json', dict(phase='loading', test_read=False))
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    runtime = V2VideoRuntime(out, native_model='oneemo')
    model = runtime.model
    base = NumericEvidenceFilter(model.get_input_embeddings().weight.shape[-1])
    runtime.numeric = HypergraphRelationExperts(base, model.get_input_embeddings().weight.shape[-1]).cuda()
    runtime.filter_class = HypergraphRelationExperts
    del base

    def inject(module, arguments, output):
        positions = runtime.context.get('positions')
        if positions is None or output.shape[1] <= int(positions[-1]):
            return output
        delta, _, _ = runtime.numeric(runtime.context['nodes'])
        reference = output[:, positions]
        residual = runtime.filter_class.bounded_residual(delta[:, -len(positions):], reference)
        updated = output.clone()
        updated[:, positions] = reference + residual.to(output.dtype)
        return updated

    hook = model.get_input_embeddings().register_forward_hook(inject)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    groups = {key: [] for key in RATES}
    names = {key: [] for key in RATES}
    for name, parameter in model.named_parameters():
        group = 'lora' if 'lora_' in name else 'merger' if '.visual.merger.' in name else None
        if group is not None:
            groups[group].append(parameter)
            names[group].append(name)
    for name, parameter in runtime.numeric.named_parameters():
        group = 'relation' if name.startswith(('experts.', 'router.', 'member_scores.', 'edge_gates.')) else 'core'
        groups[group].append(parameter)
        names[group].append('numeric.' + name)
    assert all(groups.values())
    digest = hashlib.sha256()
    for group in groups:
        for name, parameter in zip(names[group], groups[group]):
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    signature = digest.hexdigest()
    model.eval()
    runtime.numeric.eval()
    train = read(SUITE / 'train.jsonl')
    dev = read(SUITE / 'development.jsonl')
    reading = read(SUITE / 'reading_train.jsonl')
    reading_dev = read(SUITE / 'reading_development.jsonl')
    schedule = read(SUITE / f'{args.arm}_schedule.jsonl')
    protocol = json.loads((SUITE / 'prepared.json').read_text())
    counts = {key: sum(key in ACTIVE[step['task']] for step in schedule) for key in groups}
    protocol.update(arm=args.arm, initialization_tensor_sha256=signature,
        schedule_sha256=sha(SUITE / f'{args.arm}_schedule.jsonl'), module_updates=counts,
        trainable_parameter_counts={key: sum(parameter.numel() for parameter in values) for key, values in groups.items()},
        model_config_sha256=sha(runtime.source), lr_policy='Per-group warmup/cosine keyed to that group update count, no optimizer resets; mixed/staged identical LR sequences',
        fresh_initialization=True, inherited_checkpoint=None)
    save(out / 'protocol.json', protocol)
    torch.save(dict(lora=get_peft_model_state_dict(model), numeric=runtime.numeric.state_dict(),
        merger={name: parameter.detach().cpu() for name, parameter in model.named_parameters() if '.visual.merger.' in name}), out / 'initial.pt')
    for other in ['direct', 'mixed', 'staged']:
        path = SUITE / other / 'protocol.json'
        if path.exists():
            assert json.loads(path.read_text())['initialization_tensor_sha256'] == signature, 'Initialization mismatch'
    all_parameters = [parameter for values in groups.values() for parameter in values]
    optimizer = torch.optim.AdamW([dict(params=groups[key], lr=RATES[key], name=key) for key in groups],
        weight_decay=.01, eps=1e-6)
    group_updates = {key: 0 for key in groups}
    picture_cache = {}

    def reading_batch(record, answer):
        runtime.context.clear()
        key = (record['index'], record['window'], record['band'], record['operation'], record['task'])
        if key not in picture_cache:
            maps = displayed(np.array(runtime.logs[record['index']]), runtime.weights, runtime.limits)
            window, band, operation = record['window'], record['band'], record['operation']
            if record['task'] == 'space':
                images = [maps[window].copy()]
                if operation == 'flip':
                    images[0][band] = np.flip(images[0][band], 0)
                if operation == 'symmetric':
                    images[0][band] = (images[0][band] + np.flip(images[0][band], 0)) / 2
                gap = float(images[0][band, :8].mean() - images[0][band, 8:].mean())
                target = 'SIMILAR' if abs(gap) <= .03 else 'UPPER_HIGHER' if gap > 0 else 'LOWER_HIGHER'
            else:
                windows = [window - 1, window] if operation == 'identity' else [window, window - 1] if operation == 'swap' else [window - 1, window - 1]
                images = [maps[index] for index in windows]
                gap = float(images[1][band].mean() - images[0][band].mean())
                target = 'SIMILAR' if abs(gap) <= .05 else 'INCREASE' if gap > 0 else 'DECREASE'
            assert target == record['target']
            rendered = []
            for values in images:
                rgb = np.stack([values, 1 - np.abs(2 * values - 1), 1 - values], -1)
                image = Image.new('RGB', (320, 96), 'white')
                draw = ImageDraw.Draw(image)
                for number, band_name in enumerate(['delta', 'theta', 'alpha', 'beta', 'gamma']):
                    tile = Image.fromarray(np.uint8(np.rint(rgb[number] * 255))).resize((64, 64), Image.Resampling.NEAREST)
                    image.paste(tile, (number * 64, 16))
                    draw.text((number * 64 + 2, 1), band_name, fill='black')
                rendered.append(image.resize((640, 192), Image.Resampling.NEAREST))
            picture_cache[key] = rendered
        images = picture_cache[key]
        band = ['delta', 'theta', 'alpha', 'beta', 'gamma'][record['band']]
        legend = 'Five fixed-scale panels ordered delta, theta, alpha, beta, gamma. Red is high, blue low, green intermediate. '
        question = (f'Compare mean displayed power in upper versus lower image halves of the {band} panel. '
            'Reply SIMILAR if normalized difference magnitude is at most0.03; otherwise UPPER_HIGHER or LOWER_HIGHER. '
            'These are image halves, not anatomical regions.' if record['task'] == 'space' else
            f'First image is reference, second is query. Compare full-map mean displayed power in the {band} panel. '
            'Reply SIMILAR if query-minus-reference is within0.05; otherwise INCREASE or DECREASE.')
        messages = [dict(role='user', content=[dict(type='image') for image in images] +
            [dict(type='text', text=legend + question + ' Output exactly one label.')])]
        prefix = runtime.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        options = dict(images=images, images_kwargs={'min_pixels': 320 * 96, 'max_pixels': 832 * 224})
        prefix_batch = runtime.processor(text=[prefix], return_tensors='pt', **options)
        length = prefix_batch['input_ids'].shape[1]
        if answer:
            full = runtime.processor.apply_chat_template(messages + [dict(role='assistant', content=record['target'])],
                tokenize=False, add_generation_prompt=False, enable_thinking=False)
            batch = runtime.processor(text=[full], return_tensors='pt', **options)
            assert torch.equal(batch['input_ids'][:, :length], prefix_batch['input_ids'])
            labels = batch['input_ids'].clone()
            labels[:, :length] = -100
            assert (labels != -100).any()
            batch['labels'] = labels
        else:
            batch = prefix_batch
        base_model = model.get_base_model().model
        if hasattr(base_model, 'rope_deltas'):
            base_model.rope_deltas = None
        return batch.to('cuda')

    def set_active(task):
        model.requires_grad_(False)
        runtime.numeric.requires_grad_(False)
        for group in ACTIVE[task]:
            for parameter in groups[group]:
                parameter.requires_grad_(True)
        model.eval()
        runtime.numeric.eval()

    def checkpoint(filename, update):
        temporary = out / f'{filename}.tmp'
        torch.save(dict(lora=get_peft_model_state_dict(model), numeric=runtime.numeric.state_dict(),
            merger={name: parameter.detach().cpu() for name, parameter in model.named_parameters() if '.visual.merger.' in name},
            update=update, initialization_tensor_sha256=signature), temporary)
        temporary.replace(out / f'{filename}.pt')

    def evaluate(records, tag, update, arm='joint', test_read=False):
        model.eval()
        runtime.numeric.eval()
        rows = []
        with torch.no_grad(), (out / f'{tag}_{arm}_records.jsonl').open('x') as stream:
            for number, record in enumerate(records):
                batch = runtime.batch(record, joint=arm == 'joint')
                generated = model.generate(**batch, do_sample=False, max_new_tokens=320)
                tail = generated[0, batch['input_ids'].shape[1]:]
                raw = runtime.processor.tokenizer.decode(tail, skip_special_tokens=True).strip()
                terms, valid = parse_structured(raw)
                row = prediction_row(record, terms, valid, raw=raw, budget_reached=len(tail) == 320)
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                save(out / 'status.json', dict(phase='evaluating', panel=tag, arm=arm,
                    completed=number + 1, total=len(records), update=update, test_read=test_read))
        counts_sets = Counter(tuple(sorted(set(row['emotion_terms']))) for row in rows)
        result = dict(official_ew=runtime.metric.score(rows), exact_set=metrics([
            (set(row['required_concepts']), set(row['emotion_terms']) if row['parse_valid'] else set()) for row in rows]),
            unique_sets=len(counts_sets), most_common_sets=counts_sets.most_common(5), n=len(rows))
        save(out / f'{tag}_{arm}_metrics.json', result)
        return result['official_ew']['f1']

    save(out / 'status.json', dict(phase='preflight', test_read=False))
    set_active('direct')
    batch = runtime.batch(train[0], joint=True, answer=True)
    value = model(**batch, use_cache=False).loss
    assert torch.isfinite(value)
    value.backward()
    norm = torch.nn.utils.clip_grad_norm_(all_parameters, 1., error_if_nonfinite=True)
    assert float(norm) > 0
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    save(out / 'gradient_check.json', dict(loss=float(value.detach()), gradient_norm=float(norm),
        frozen_gradients=False, test_read=False))
    optimizer.zero_grad(set_to_none=True)
    set_active('reading')
    batch = reading_batch(reading[0], True)
    value = model(**batch, use_cache=False).loss
    assert torch.isfinite(value)
    value.backward()
    norm = torch.nn.utils.clip_grad_norm_([parameter for parameter in all_parameters if parameter.requires_grad], 1., error_if_nonfinite=True)
    assert float(norm) > 0
    save(out / 'reading_check.json', dict(loss=float(value.detach()), gradient_norm=float(norm),
        target=reading[0]['target'], source_subject=reading[0]['subject'], test_read=False))
    optimizer.zero_grad(set_to_none=True)
    if args.smoke:
        runtime.context.clear()
        with torch.inference_mode():
            batch = reading_batch(reading_dev[0], False)
            generated = model.generate(**batch, do_sample=False, max_new_tokens=16)
            raw = runtime.processor.tokenizer.decode(generated[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)
        save(out / 'complete.json', dict(phase='smoke_complete', raw=raw, test_read=False))
        return
    best, selected = evaluate(dev, 'development0', 0), 0
    checkpoint('best', 0)
    history = []
    training_seconds = 0.
    for update, step in enumerate(schedule, 1):
        torch.cuda.synchronize()
        step_started = time.time()
        task = step['task']
        set_active(task)
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            key = group['name']
            if key in ACTIVE[task]:
                group_updates[key] += 1
                count = group_updates[key]
                warmup = max(1, math.ceil(.05 * counts[key]))
                factor = count / warmup if count <= warmup else .5 * (1 + math.cos(math.pi * (count - warmup) / (counts[key] - warmup)))
                group['lr'] = RATES[key] * factor
        losses = []
        token_count = 0
        for index in step['rows']:
            record = reading[index] if task == 'reading' else train[index]
            batch = reading_batch(record, True) if task == 'reading' else runtime.batch(record,
                joint=task in ['direct', 'joint', 'relation'], answer=True)
            token_count += int(batch['attention_mask'].sum())
            value = model(**batch, use_cache=False).loss
            assert torch.isfinite(value), (task, update)
            losses.append(float(value.detach()))
            (value / 8).backward()
        active_parameters = [parameter for parameter in all_parameters if parameter.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(active_parameters, 1., error_if_nonfinite=True)
        assert all(parameter.grad is None for parameter in all_parameters if not parameter.requires_grad)
        optimizer.step()
        torch.cuda.synchronize()
        training_seconds += time.time() - step_started
        row = dict(update=update, task=task, mean_ce=float(np.mean(losses)), gradient_norm=float(norm),
            input_tokens=token_count, group_updates=dict(group_updates), train_seconds=training_seconds)
        history.append(row)
        save(out / 'history.json', history)
        save(out / 'status.json', dict(phase='training', total_updates=150, test_read=False, **row))
        if update in [50, 100, 150]:
            score = evaluate(dev, f'development{update}', update)
            save(out / f'development{update}_selection.json', dict(ew_f1=score, update=update, test_read=False))
            if score > best:
                best, selected = score, update
                checkpoint('best', update)
            checkpoint('last', update)
    assert group_updates == counts
    saved = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    assert saved['initialization_tensor_sha256'] == signature
    loaded = set_peft_model_state_dict(model, saved['lora'])
    assert not loaded.unexpected_keys
    runtime.numeric.load_state_dict(saved['numeric'], strict=True)
    parameters_by_name = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in saved['merger'].items():
            parameters_by_name[name].copy_(value.to(parameters_by_name[name].device))
    save(out / 'selection_lock.json', dict(selected_update=selected, development_f1=best,
        checkpoint_sha256=sha(out / 'best.pt'), test_read=False))
    model.requires_grad_(False)
    runtime.numeric.requires_grad_(False)
    evaluate(panel()['val'], 'selected_full560', selected)
    reading_outputs = []
    with torch.inference_mode():
        for record in reading_dev:
            batch = reading_batch(record, False)
            generated = model.generate(**batch, do_sample=False, max_new_tokens=16)
            raw = runtime.processor.tokenizer.decode(generated[0, batch['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            reading_outputs.append(dict(**record, raw=raw, correct=raw.strip(' .\n').upper() == record['target']))
    write_rows(out / 'selected_reading60_records.jsonl', reading_outputs)
    reading_scores = {}
    for task in ['space', 'time']:
        recalls = [np.mean([row['correct'] for row in reading_outputs if row['task'] == task and row['target'] == label])
                   for label in sorted({row['target'] for row in reading_outputs if row['task'] == task})]
        reading_scores[task] = float(np.mean(recalls))
    save(out / 'selected_reading60_metrics.json', dict(bacc=reading_scores, selection_use=False, test_read=False))
    final = read(ROOT / 'final_test_20261120/test_records.jsonl')
    assert len(final) == 644 and {row['subject'] for row in final} == set(range(100, 123))
    runtime.logs = np.load(ROOT / 'final_test_20261120/band_logpower.npy', mmap_mode='r')
    runtime.images.clear()
    runtime.nodes.clear()
    for arm in ['joint', 'eeg']:
        evaluate(final, 'selected_test644', selected, arm=arm, test_read=True)
    save(out / 'complete.json', dict(selected_update=selected, development_f1=best,
        seconds=time.time() - started, train_seconds=history[-1]['train_seconds'],
        training_input_tokens=sum(row['input_tokens'] for row in history), module_updates=group_updates,
        initial_tensor_sha256=signature, test_read=True))
    save(out / 'status.json', dict(phase='complete', test_read=True))
    hook.remove()
    gpu_lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--arm', choices=['direct', 'mixed', 'staged'])
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    try:
        if args.prepare_only:
            prepare()
        else:
            assert args.arm is not None and args.gpu is not None
            if args.smoke:
                SUITE = SUITE / 'smoke'
                SUITE.mkdir(exist_ok=False)
                source = SUITE.parent
                for name in ['train.jsonl', 'development.jsonl', 'reading_train.jsonl', 'reading_development.jsonl', 'prepared.json', f'{args.arm}_schedule.jsonl']:
                    (SUITE / name).symlink_to(source / name)
            run(args)
    except Exception:
        destination = SUITE / args.arm if args.arm else SUITE
        if destination.exists():
            save(destination / 'failure.json', dict(error=traceback.format_exc()))
            save(destination / 'status.json', dict(phase='failed'))
        raise
