"""Input-matched native-MLLM LoRA baselines, without numeric evidence injection."""
import argparse
import json
import math
import time
import traceback

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from common import ROOT, lock_gpu, panel, prediction_row, save, sha, write_rows
from parser import parse_structured
from joint_runtime import V2VideoRuntime


def main(name, out, measurement_text=False, video_only=False):
    gpu_lock = lock_gpu()
    out.mkdir(exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(20261109)
    rng = np.random.default_rng(20261109)
    datasets = panel()
    train = []
    for subject in range(80):
        pool = [row for row in datasets['train'] if row['subject'] == subject]
        train.extend(pool[int(index)] for index in rng.choice(len(pool), 3, replace=False))
    pilot = ROOT / 'v2_video_joint_lora_pilot_v1'
    original_train = [json.loads(line) for line in (pilot / 'train_records.jsonl').read_text().splitlines()]
    assert train == original_train and len(train) == 240
    val = [row for row in datasets['val'] if row['subject'] in [80, 81]]
    assert len(val) == 56
    write_rows(out / 'train_records.jsonl', train)
    write_rows(out / 'selection_records.jsonl', val)
    save(out / 'status.json', {'phase': 'loading_model', 'test_read': False})
    if video_only:
        from video_runtime import VideoOnlyRuntime
        runtime = VideoOnlyRuntime(out, native_model=name)
    elif measurement_text:
        from text_runtime import MeasurementTextRuntime
        runtime = MeasurementTextRuntime(out, native_model=name)
    else:
        runtime = V2VideoRuntime(out, native_model=name)
    model = runtime.model
    for parameter_name, parameter in model.named_parameters():
        parameter.requires_grad_('lora_' in parameter_name)
        assert not parameter.requires_grad or 'visual' not in parameter_name
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert parameters and isinstance(runtime.numeric, torch.nn.Identity)
    protocol = {'model': name, 'initialization': 'Native released weights plus fresh zero-output LoRA; no V2 weights',
        'native_model_config': str(runtime.source), 'config_sha256': sha(runtime.source),
        'source_sha256': sha(__file__), 'runtime_sha256': sha(__import__('v2_video_runtime').__file__),
        'train_panel_sha256': sha(out / 'train_records.jsonl'), 'training_trials': 240,
        'selection_trials': 56, 'final_validation_trials': 560, 'test_read': False,
        'epochs': 2, 'updates': 120, 'effective_batch': 8, 'microbatch': 1,
        'mix': 'Each update four EEG-map-only and four EEG-map+video inputs; unchanged pilot order and seed',
        'seed': 20261109, 'lora_rank': 16, 'lora_alpha': 32, 'lora_dropout': 0.,
        'trainable_parameters': sum(parameter.numel() for parameter in parameters),
        'lora_targets': 'Language q/k/v/o;OneEmo also existing linear-attention in_proj_qkv/out_proj targets',
        'optimizer': 'AdamW LR1e-5,weight_decay0.01,6warmup updates then cosine to0,clip1',
        'loss': 'Answer-only CE on the exact pilot weak emotion targets; prefix fully masked',
        'numeric_expert_injection': False, 'base_and_visual_frozen': True,
        'input': 'Same two EEG maps,eight full-clip RGB frames,prompts and pixel budgets as adapted method;no audio',
        'selection': 'Joint56EW-F1 then EEG-only56EW-F1;initial released checkpoint eligible',
        'final': 'Fixed selected checkpoint;report complete560,prior56 andremaining504;no reselection',
        'comparison_boundary': 'Matches joint-adaptation stage,not total training history or parameter count of our method',
        'reference': 'Eight threshold3.5 self-report concepts;known stimuli across disjoint subjects,not independent OV truth',
        'parsing': 'Same format-compatible parser;invalid counted separately,strict score does not measure all raw-text content',
        'metric': runtime.metric.protocol}
    if name == 'qwen25vl3b':
        protocol['model_revision'] = '66285546d2b821cf421d4f5eb2576359d3770cd3'
    if measurement_text:
        protocol.update(input='Eight native video frames plus deterministic EEG measurement text; no EEG images, audio, or residual injection',
            mix='Each update four EEG-text-only and four EEG-text+video presentations',
            measurement='Source log-power [10,5,30]; mean over windows0:5 or5:10 and all channels; round4; two five-band vectors; no self-report or stimulus label in prompt',
            runtime_sha256=sha(__import__('measurement_text_runtime').__file__),
            final='Lock best checkpoint on56validation, report full560validation then644test once; no test selection',
            development_boundary='Follow-up protocol after earlier test results were observed; not a new untouched test or confirmatory preregistered comparison')
    if video_only:
        protocol.update(input='Eight stimulus frames only; no EEG, classifier, participant identifiers or ratings in prompt',
            mix='Video-only throughout. Each trial presented twice per nominal epoch:480 presentations/epoch,960 total; matches mixed baseline120updates rather than its per-modality exposure.',
            selection='Video-only development56 EW-F1; epoch0 eligible; earliest ties',
            prompt='Unchanged existing video-only evaluation prompt; used for training and all evaluations',
            final='Lock best checkpoint on56validation before full560validation and644test; no test reselection',
            runtime_sha256=sha(__import__('video_only_runtime').__file__),
            comparison_boundary='Matches native joint baseline final-stage samples/updates, not BrainRel cumulative curriculum; video exposure is higher than mixed adaptation',
            development_boundary='Follow-up after prior test inspection, not untouched confirmatory evaluation')
    save(out / 'protocol.json', protocol)

    def evaluate(epoch, records, prefix, test_read=False):
        scores = {}
        with torch.no_grad():
            for arm in (['video'] if video_only else ['eeg', 'joint']):
                rows = []
                with (out / f'{prefix}_{arm}.jsonl').open('x') as stream:
                    for number, record in enumerate(records):
                        batch = runtime.batch(record, joint=arm == 'joint')
                        generated = model.generate(**batch, do_sample=False, max_new_tokens=320)
                        tail = generated[0, batch['input_ids'].shape[1]:]
                        raw = runtime.processor.tokenizer.decode(tail, skip_special_tokens=True).strip()
                        terms, valid = parse_structured(raw)
                        row = prediction_row(record, terms, valid, raw=raw,
                            budget_reached=len(tail) == 320, input_tokens=batch['input_ids'].shape[1])
                        rows.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                        stream.flush()
                        save(out / 'status.json', {'phase': 'evaluating', 'panel': prefix, 'epoch': epoch,
                            'arm': arm, 'completed': number + 1, 'total': len(records), 'test_read': test_read})
                scores[arm] = runtime.metric.score(rows)
                from exact import metrics
                from collections import Counter
                counts = Counter(tuple(sorted(set(row['emotion_terms']))) for row in rows)
                scores[arm]['exact_set'] = metrics([(set(row['required_concepts']),
                    set(row['emotion_terms']) if row['parse_valid'] else set()) for row in rows])
                scores[arm]['unique_sets'] = len(counts)
                scores[arm]['most_common_sets'] = counts.most_common(5)
                if len(records) == 560:
                    scores[arm]['prior56'] = runtime.metric.score([row for row in rows if row['subject'] in [80, 81]])
                    scores[arm]['remaining504'] = runtime.metric.score([row for row in rows if row['subject'] not in [80, 81]])
        save(out / f'{prefix}_metrics.json', scores)
        return scores

    def checkpoint(filename, epoch):
        temporary = out / f'{filename}.tmp'
        torch.save({'lora': get_peft_model_state_dict(model), 'epoch': epoch, 'protocol': protocol}, temporary)
        temporary.replace(out / f'{filename}.pt')

    if video_only:
        batch = runtime.batch(train[0], answer=True)
        model.eval()
        loss = model(**batch, use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        assert float(gradient) > 0
        assert all(parameter.grad is None for parameter_name, parameter in model.named_parameters()
                   if 'lora_' not in parameter_name)
        model.zero_grad(set_to_none=True)
        save(out / 'gradient_check.json', dict(loss=float(loss.detach()), gradient_norm=float(gradient),
            frozen_gradients=False, test_read=False))
    initial = evaluate(0, val, 'epoch0')
    best_key = (initial['video']['f1'],) if video_only else (initial['joint']['f1'], initial['eeg']['f1'])
    best_epoch = 0
    checkpoint('best', 0)
    history = [{'epoch': 0, 'updates': 0, 'scores': initial}]
    save(out / 'history.json', history)
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    update = 0
    training_started = time.time()
    for epoch in [1, 2]:
        order = rng.permutation(len(train)).tolist()
        losses = {'video': []} if video_only else {'eeg': [], 'joint': []}
        for offset in range(0, len(order), 4):
            factor = (update + 1) / 6 if update < 6 else .5 * (1 + math.cos(math.pi * (update - 6) / 113))
            optimizer.param_groups[0]['lr'] = 1e-5 * factor
            presentations = [(train[index], joint) for index in order[offset:offset + 4]
                             for joint in ([True, True] if video_only else [False, True])]
            rng.shuffle(presentations)
            for record, joint in presentations:
                batch = runtime.batch(record, joint=joint, answer=True)
                loss = model(**batch, use_cache=False).loss
                assert torch.isfinite(loss), 'Non-finite training CE'
                losses['video' if video_only else 'joint' if joint else 'eeg'].append(float(loss.detach()))
                (loss / 8).backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            if update == 0:
                assert float(gradient) > 0
                assert all(parameter.grad is None for parameter_name, parameter in model.named_parameters()
                           if 'lora_' not in parameter_name)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            save(out / 'status.json', {'phase': 'training', 'epoch': epoch, 'update': update, 'total_updates': 120,
                'gradient_norm': float(gradient), 'learning_rate': optimizer.param_groups[0]['lr'],
                'train_ce': {key: float(np.mean(value)) for key, value in losses.items()},
                'training_seconds': time.time() - training_started, 'test_read': False})
            if update % 10 == 0:
                checkpoint('last', epoch)
                print('UPDATE', name, update, flush=True)
        scores = evaluate(epoch, val, f'epoch{epoch}')
        history.append({'epoch': epoch, 'updates': update, 'scores': scores,
                        'train_ce': {key: float(np.mean(value)) for key, value in losses.items()}})
        save(out / 'history.json', history)
        key = (scores['video']['f1'],) if video_only else (scores['joint']['f1'], scores['eeg']['f1'])
        if key > best_key:
            best_key, best_epoch = key, epoch
            checkpoint('best', epoch)
        checkpoint('last', epoch)
    assert update == 120
    del optimizer
    state = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    result = set_peft_model_state_dict(model, state['lora'])
    assert not result.unexpected_keys and state['epoch'] == best_epoch
    model.requires_grad_(False)
    final = evaluate(best_epoch, datasets['val'], 'selected_full560')
    test = None
    if measurement_text or video_only:
        save(out / 'selection_lock.json', dict(epoch=best_epoch, checkpoint_sha256=sha(out / 'best.pt'),
            selection='56validation only; no subsequent reselection', test_read=False))
        test_records = [json.loads(line) for line in
                        (ROOT / 'final_test_20261120/test_records.jsonl').read_text().splitlines()]
        assert len(test_records) == 644 and {row['subject'] for row in test_records} == set(range(100, 123))
        assert all(row['index'] % 3 == 1 for row in test_records)
        if not video_only:
            runtime.logs = np.load(ROOT / 'final_test_20261120/band_logpower.npy', mmap_mode='r')
            assert np.isfinite(runtime.logs[[row['index'] for row in test_records]]).all()
        test = evaluate(best_epoch, test_records, 'selected_test644', test_read=True)
    save(out / 'complete.json', {'phase': 'complete', 'model': name, 'selected_epoch': best_epoch,
        'updates': update, 'full560': final, 'checkpoint_sha256': sha(out / 'best.pt'),
        'test644': test, 'seconds': time.time() - started, 'test_read': measurement_text or video_only})
    save(out / 'status.json', {'phase': 'complete', 'test_read': measurement_text or video_only})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['qwen25vl3b', 'oneemo'])
    parser.add_argument('--measurement-text', action='store_true')
    parser.add_argument('--video-only', action='store_true')
    args = parser.parse_args()
    assert not (args.measurement_text and args.video_only)
    destination = ROOT / ('native_video_only_adaptation_v1' if args.video_only else
        'native_text_mllm_adaptation_v1' if args.measurement_text else 'native_mllm_adaptation_v1') / args.model
    try:
        main(args.model, destination, measurement_text=args.measurement_text, video_only=args.video_only)
    except Exception as error:
        if destination.exists() and not (destination / 'complete.json').exists():
            save(destination / 'status.json', {'phase': 'failed', 'error': repr(error), 'test_read': False})
        traceback.print_exc()
        raise
