"""FACED joint LoRA baseline with audited restoration of delayed tower weights."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from common import ROOT, PROJECT, evaluator, lock_gpu, panel, prediction_row, save, sha
from interfaces import R1Omni, PROMPT
from eval_erv import load_frame_reader, parse


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), FACED_GPU_INDEX=str(args.gpu),
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    out = ROOT / ('native_video_only_adaptation_v1/r1omni' if args.video_only else
        'additional_text_baselines_lora_v1/r1omni') / ('smoke' if args.smoke else 'adaptation')
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'status.json', dict(phase='loading', test_read=False))
    gpu_lock = lock_gpu()
    import numpy as np
    import torch
    from safetensors import safe_open
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    torch.set_num_threads(4)
    torch.manual_seed(20261122)
    rng = np.random.default_rng(20261122)
    prepared = ROOT / 'additional_baselines_v1/r1omni'
    runtime = R1Omni(prepared)
    base = runtime.model
    restored = {}
    with safe_open(str(prepared / 'runtime_model/model.safetensors'), framework='pt') as source:
        for name in ['vision_tower', 'audio_tower']:
            prefix = f'model.{name}.'
            tower = getattr(base.model, name)
            weights = {key[len(prefix):]: source.get_tensor(key) for key in source.keys() if key.startswith(prefix)}
            assert weights, name
            before = tower.state_dict()
            differences = sum(not torch.equal(before[key].cpu(), value.to(before[key].dtype))
                              for key, value in weights.items())
            tower.load_state_dict(weights, strict=True)
            after = tower.state_dict()
            assert all(torch.equal(after[key].cpu(), value.to(after[key].dtype)) for key, value in weights.items())
            restored[name] = dict(keys=len(weights), differing_before= differences, exact_after=True)
    save(out / 'loader_audit.json', dict(restored=restored, original_weights_modified=False,
        old_zero_shot_comparability='Prior loader delayed tower construction and ignored nested tower keys; do not treat as identical initialization.',
        checkpoint_sha256=sha(prepared / 'runtime_model/model.safetensors')))
    base.requires_grad_(False)
    targets = [name for name, module in base.named_modules() if name.startswith('model.layers.')
               and name.endswith(('.q_proj', '.k_proj', '.v_proj', '.o_proj'))
               and isinstance(module, torch.nn.Linear)]
    assert len(targets) == 96
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
        target_modules=targets, bias='none', task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert parameters and all('lora_' in name for name, parameter in model.named_parameters() if parameter.requires_grad)
    for parameter in parameters:
        parameter.data = parameter.data.float()
    runtime.model = model
    from humanomni.constants import DEFAULT_VIDEO_TOKEN
    from humanomni.mm_utils import tokenizer_multimodal_token
    tokenizer = runtime.tokenizer
    train = read(ROOT / 'v2_video_joint_lora_pilot_v1/train_records.jsonl')
    datasets = panel()
    dev = [row for row in datasets['val'] if row['subject'] in [80, 81]]
    assert len(train) == 240 and len(dev) == 56
    assert {row['subject'] for row in train} <= set(range(80))
    logs = None if args.video_only else np.load(ROOT / 'official200_classification_v1/band_logpower.npy', mmap_mode='r')
    reader, frame_cache, tensor_cache = load_frame_reader(), {}, {}
    metric = evaluator()
    protocol = dict(seed=20261122, train_trials=240, development_trials=56, epochs=2,
        updates=60, accumulation=8, learning_rate=1e-5, warmup_updates=6, scheduler='cosine',
        optimizer='AdamW', optimizer_epsilon=1e-6, weight_decay=.01, rank=16, alpha=32,
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
        targets=targets, frozen='Vision/audio towers, mergers, BERT and all non-LoRA parameters',
        input='8 stimulus frames + two five-second channel-mean log-band-power vectors; no EEG atlas or audio',
        supervision='Answer-only emotion set from existing training weak labels; no invented reasoning targets',
        selection='Joint development56 EW-F1; epoch0 eligible; earliest ties; lock before final test',
        video_only='Same selected joint adapter with EEG text removed, not separately adapted',
        full_dataset_adaptation=False, script_sha256=sha(__file__), test_read=False,
        loader_audit='loader_audit.json', metric=metric.protocol)
    if args.video_only:
        protocol.update(input='Eight stimulus video frames only; no EEG, audio, classifier or participant metadata',
            updates=120, presentations_per_epoch=480, repeat_per_trial_per_epoch=2,
            selection='Video-only development56 EW-F1; epoch0 eligible; earliest ties; lock before test',
            video_only='Dedicated video-only training and selection, not joint adapter with EEG removed',
            comparison_boundary='Same120updates as dedicated video baselines; more updates than previous R1 joint60',
            development_boundary='Follow-up after prior test inspection, not untouched confirmatory evaluation')
    save(out / 'protocol.json', protocol)

    def inputs(record, arm='joint'):
        key = record['video_id']
        if key not in frame_cache:
            frame_cache[key] = reader(ROOT / f'stimulus_videos_v1/videos/{key}.mp4')[0]
            tensor_cache[key] = runtime.processor['video'](frame_cache[key]).half().cuda()
        question = ('Infer possible emotions experienced by the participant using unrestricted vocabulary. '
            'The stimulus video is fallible context, not a recording of the participant. '
            'No EEG, audio or classifier prediction is supplied. Do not invent absent evidence. '
            if args.video_only else PROMPT)
        if arm == 'joint' and not args.video_only:
            values = np.array(logs[record['index']])
            assert np.isfinite(values).all()
            measurements = [values[part].mean(axis=(0, 2)).round(4).tolist()
                            for part in [slice(0, 5), slice(5, 10)]]
            question += ' EEG log-band-power channel means for successive five-second windows, ordered delta/theta/alpha/beta/gamma: ' + json.dumps(measurements)
        question += ' Return final emotions in <answer> </answer> tags, comma-separated; [] is allowed.'
        messages = [dict(role='user', content=DEFAULT_VIDEO_TOKEN + '\n' + question)]
        prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if model.config.mm_use_x_start_end:
            prefix = prefix.replace(DEFAULT_VIDEO_TOKEN, '<vi_start>' + DEFAULT_VIDEO_TOKEN + '<vi_end>')
        prefix_ids = tokenizer_multimodal_token(prefix, tokenizer, DEFAULT_VIDEO_TOKEN,
                                               return_tensors='pt').unsqueeze(0).long().cuda()
        bert = runtime.bert([question], return_tensors='pt', padding=True, truncation=True,
                            add_special_tokens=True).to('cuda')
        return prefix_ids, [(tensor_cache[key], 'video')], bert, question

    def loss(record):
        prefix, images, bert, _ = inputs(record)
        terms = record['target']['emotions']
        answer = '<answer>' + (', '.join(terms) if terms else '[]') + '</answer>' + tokenizer.eos_token
        suffix = tokenizer(answer, add_special_tokens=False, return_tensors='pt').input_ids.cuda()
        ids = torch.cat([prefix, suffix], dim=1)
        labels = ids.clone()
        labels[:, :prefix.shape[1]] = -100
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                       images=images, prompts=bert, audios=None, use_cache=False)
        return output.loss

    def generate(record, arm):
        ids, images, bert, question = inputs(record, arm)
        with torch.inference_mode():
            generated = model.generate(ids, attention_mask=torch.ones_like(ids), images=images,
                prompts=bert, audios=None, do_sample=False, max_new_tokens=320,
                use_cache=True, pad_token_id=tokenizer.eos_token_id)
        return dict(raw=tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip(), prompt=question)

    def evaluate(records, tag, arm='joint'):
        model.eval()
        predictions, video_outputs = [], {}
        with (out / f'{tag}_records.jsonl').open('x') as stream:
            for number, record in enumerate(records):
                key = record['video_id']
                response = video_outputs.get(key) if arm == 'video' else None
                if response is None:
                    response = generate(record, arm)
                    if arm == 'video':
                        video_outputs[key] = response
                terms, valid = parse(response['raw'])
                row = prediction_row(record, terms, valid, **response)
                predictions.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                save(out / 'status.json', dict(phase='evaluating', panel=tag, completed=number + 1,
                    total=len(records), test_read=tag.startswith('final')))
        from exact import metrics
        counts = Counter(tuple(sorted(set(row['emotion_terms']))) for row in predictions)
        result = dict(official_ew=metric.score(predictions), exact_set=metrics([
            (set(row['required_concepts']), set(row['emotion_terms']) if row['parse_valid'] else set())
            for row in predictions]), unique_sets=len(counts), most_common_sets=counts.most_common(5),
            invalid=sum(not row['parse_valid'] for row in predictions), n=len(records))
        save(out / f'{tag}_metrics.json', result)
        return result['official_ew']['f1']

    model.eval()
    model.get_base_model().model.vision_tower.eval()
    value = loss(train[0])
    assert torch.isfinite(value)
    value.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    assert float(norm) > 0
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    save(out / 'gradient_check.json', dict(loss=float(value.detach()), gradient_norm=float(norm),
        frozen_gradients=False, test_read=False))
    model.zero_grad(set_to_none=True)
    model.eval()
    save(out / 'generation_check.json', generate(dev[0], 'joint'))
    if args.smoke:
        model.eval()
        save(out / 'complete.json', dict(phase='smoke_complete', response=generate(dev[0], 'joint')))
        return
    started = time.time()
    best, selected = evaluate(dev, 'epoch0', 'video' if args.video_only else 'joint'), 0
    torch.save(get_peft_model_state_dict(model), out / 'best.pt')
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, eps=1e-6, weight_decay=.01)
    history, update = [], 0
    for epoch in [1, 2]:
        order, losses = rng.permutation(len(train)), []
        if args.video_only:
            order = np.repeat(order, 2)
        for offset in range(0, len(order), 8):
            model.eval()
            model.get_base_model().model.vision_tower.eval()
            optimizer.zero_grad(set_to_none=True)
            group = order[offset:offset + 8]
            update += 1
            factor = update / 6 if update <= 6 else .5 * (1 + math.cos(math.pi * (update - 6) / (protocol['updates'] - 6)))
            optimizer.param_groups[0]['lr'] = 1e-5 * factor
            for index in group:
                value = loss(train[int(index)])
                assert torch.isfinite(value)
                losses.append(float(value.detach()))
                (value / len(group)).backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            save(out / 'status.json', dict(phase='training', epoch=epoch, update=update,
                total_updates=protocol['updates'], loss=float(np.mean(losses[-len(group):])), test_read=False))
        score = evaluate(dev, f'epoch{epoch}', 'video' if args.video_only else 'joint')
        if score > best:
            best, selected = score, epoch
            torch.save(get_peft_model_state_dict(model), out / 'best.pt')
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), development_f1=score))
        save(out / 'history.json', history)
    set_peft_model_state_dict(model, torch.load(out / 'best.pt', map_location='cpu', weights_only=True))
    save(out / 'selection_lock.json', dict(epoch=selected, development_f1=best,
        checkpoint_sha256=sha(out / 'best.pt'), test_read=False))
    evaluate(datasets['val'], 'selected_full560', 'video' if args.video_only else 'joint')
    final = read(ROOT / 'final_test_20261120/test_records.jsonl')
    assert len(final) == 644 and {row['subject'] for row in final} == set(range(100, 123))
    if not args.video_only:
        logs = np.load(ROOT / 'final_test_20261120/band_logpower.npy', mmap_mode='r')
    for arm in (['video'] if args.video_only else ['joint', 'video']):
        evaluate(final, f'final_{arm}', arm)
    save(out / 'complete.json', dict(phase='complete', selected_epoch=selected, seconds=time.time() - started))
    save(out / 'status.json', dict(phase='complete', test_read=True))
    gpu_lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--video-only', action='store_true')
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception:
        destination = ROOT / ('native_video_only_adaptation_v1/r1omni' if arguments.video_only else
            'additional_text_baselines_lora_v1/r1omni') / ('smoke' if arguments.smoke else 'adaptation')
        if destination.exists():
            save(destination / 'failure.json', dict(error=traceback.format_exc()))
            save(destination / 'status.json', dict(phase='failed'))
        raise
