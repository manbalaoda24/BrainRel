"""Independent InternVL language-LoRA adaptation, dev selection, then fixed evaluation."""
import argparse
import fcntl
import json
import math
import os
import ast
import pickle
import zipfile
from pathlib import Path
import time
import traceback

from common import ROOT, PROJECT, save, sha, lock_gpu, evaluator, panel
from interfaces import InternVL35
from eval_extra import source_function
from eval_erv import load_frame_reader


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), FACED_GPU_INDEX=str(args.gpu),
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    data_root = ROOT if args.dataset == 'faced' else Path('/data/model/datasets/EAV_13799131')
    assert not args.measurement_text or args.dataset == 'faced'
    assert not args.waveform or (args.dataset == 'faced' and not args.measurement_text)
    assert not args.video_only or (args.dataset == 'faced' and not args.measurement_text and not args.waveform)
    out = (data_root / 'eeg_llm_families_v1/internvl_waveform_v1' /
           ('smoke' if args.smoke else 'adaptation')) if args.waveform else (
           data_root / ('additional_text_baselines_lora_v1' if args.measurement_text else 'additional_baselines_lora_v1') / 'internvl35')
    if args.video_only:
        out = ROOT / 'native_video_only_adaptation_v1/internvl35'
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'status.json', dict(phase='loading', test_read=False))
    lock = lock_gpu()
    eav_lock = None
    if args.dataset == 'eav':
        eav_lock = (data_root / f'eav_gpu{args.gpu}.lock').open('w')
        fcntl.flock(eav_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    torch.set_num_threads(4)
    torch.manual_seed(20261119)
    rng = np.random.default_rng(20261119)
    runtime = InternVL35(ROOT / 'additional_baselines_v1/internvl35')
    if args.video_only:
        from eval_video import PROMPT as VIDEO_PROMPT
        runtime.prompt = VIDEO_PROMPT
    model = runtime.model
    model.language_model = get_peft_model(model.language_model, LoraConfig(r=16, lora_alpha=32,
        lora_dropout=0., target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        bias='none', task_type='CAUSAL_LM'))
    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert parameters and all('language_model' in name and 'lora_' in name
                              for name, parameter in model.named_parameters() if parameter.requires_grad)
    if args.dataset == 'faced':
        datasets = panel()
        train = read(ROOT / 'v2_video_joint_lora_pilot_v1/train_records.jsonl')
        dev = [row for row in datasets['val'] if row['subject'] in [80, 81]]
        assert len(train) == 240 and len(dev) == 56
        assert {row['subject'] for row in train} <= set(range(80))
        cache = ROOT / 'official200_classification_v1'
        limits_path = ROOT / 'traceable_visual_alignment_v1/measurements.npz'
        reader, epochs, rate = load_frame_reader(), 2, 1e-5
    else:
        cache = data_root / 'brainrel_adapter_v1'
        train = read(data_root / 'open_weak_asr_student_joint_v3/train_weak_records.jsonl')
        dev = read(data_root / 'open_weak_asr_student_joint_v3/val_weak_records.jsonl')
        assert len(train) == 420 and len(dev) == 63
        assert {row['subject'] for row in train} <= set(range(1, 25))
        assert {row['subject'] for row in dev} <= set(range(25, 29))
        import cv2
        reader = source_function(PROJECT / 'scripts/eav/runtime.py', 'frames', dict(np=np, cv2=cv2))
        runtime.prompt = ('Recognize the participant emotional states with unrestricted vocabulary and number. '
            'The video shows the participant, not an external stimulus movie. EEG and video cover '
            'seconds 5-15; the EEG maps summarize seconds 5-10 and 10-15. Measurements do not uniquely '
            'establish emotions. Do not invent events or unique physiological causes. ')
        runtime.video_label = 'Participant frame'
        limits_path, epochs, rate = cache / 'measurements.npz', 3, 2e-5
    assert not {row['subject'] for row in train} & {row['subject'] for row in dev}
    logs = np.load(cache / 'band_logpower.npy', mmap_mode='r')
    weights = np.load(cache / 'topology_geometry.npz')['weights'].astype('float32')
    limits = np.load(limits_path)['limits']
    atlas = source_function(PROJECT / 'scripts/faced/run.py', 'atlas',
        dict(np=np, Image=Image, ImageDraw=ImageDraw,
             BANDS=[(name, 0, 0) for name in ['delta', 'theta', 'alpha', 'beta', 'gamma']]))
    parser = source_function(PROJECT / 'scripts/mllm/eval_mllm.py', 'parse_answer', dict(json=json))
    metric, video_cache = evaluator(), {}
    if args.waveform:
        signals = np.load(cache / 'signal.npy', mmap_mode='r')
        render_waveform = source_function(PROJECT / 'scripts/mllm/train_phemo_eeg.py',
            'render_waveform', dict(np=np, Image=Image, ImageDraw=ImageDraw))
        runtime.prompt = ('Infer possible emotional experiences of the participant from the EEG waveform. '
            'Only EEG is available: no video, stimulus category, classifier, or personal metadata. '
            'Channel labels are indices, not anatomical region names. The plot covers all32channels '
            'and the middle10seconds at200Hz, with shared offset spacing and no per-channel normalization. '
            'EEG alone does not uniquely establish emotions; vocabulary and number are unrestricted. ')
    text_prompt = source_function(PROJECT / 'scripts/mllm/text_runtime.py',
        'measurement_prompt', dict(json=json)) if args.measurement_text else None
    for name, rows in [('train', train), ('selection', dev)]:
        (out / f'{name}_records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    protocol = dict(dataset=args.dataset, training_trials=len(train), selection_trials=len(dev), epochs=epochs,
        lora_rank=16, lora_alpha=32, targets=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        visual_encoder_and_merger_frozen=True, effective_batch=8, microbatch=1,
        learning_rate=rate, optimizer='AdamW', weight_decay=0.01 if args.dataset == 'faced' else 0.,
        scheduler='6/8 warmup steps then cosine decay', loss='Answer-only autoregressive CE, including end separator',
        selection='Development joint EW-F1, initial zero-shot model eligible; no test-based selection',
        initialization='Released InternVL3.5-2B; independent fresh LoRA for each dataset',
        train_manifest_sha256=sha(out / 'train_records.jsonl'), selection_sha256=sha(out / 'selection_records.jsonl'),
        seed=20261119, mixed_eeg_joint=args.dataset == 'faced', reference='Existing weak labels, not independent open gold',
        script_sha256=sha(__file__), full_dataset_adaptation=False)
    save(out / 'protocol.json', protocol)
    if args.measurement_text:
        protocol.update(input='Eight native video frames plus two five-band EEG measurement vectors as text; no EEG atlas',
            mixed_eeg_joint=False, presentations_per_epoch=len(train), updates=epochs * math.ceil(len(train) / 8),
            measurement='Log-power [10,5,30], mean windows0:5 or5:10 and channels, round4; identical to measurement_text_runtime',
            comparison_boundary='Joint-only training,60updates,not the120update mixed-text/joint budget of Qwen/OneEmo',
            development_boundary='Follow-up after previous test inspection; test not used for checkpoint selection')
        save(out / 'protocol.json', protocol)

    if args.video_only:
        protocol.update(input='Eight stimulus frames only; no EEG atlas, measurement text, audio or classifier',
            mixed_eeg_joint=False, presentations_per_epoch=480, updates=120, repeat_per_trial_per_epoch=2,
            selection='Video-only development56 EW-F1; epoch0 eligible; earliest ties',
            comparison_boundary='Dedicated video adaptation,120updates; does not match BrainRel cumulative history',
            development_boundary='Follow-up after prior test inspection, not untouched confirmatory evaluation')
        save(out / 'protocol.json', protocol)
    if args.waveform:
        protocol.update(input='One32-channel EEG waveform image; no atlas, video, text measurements, or classifier logits',
            mixed_eeg_joint=False, presentations_per_epoch=len(train), updates=epochs * math.ceil(len(train) / 8),
            selection='EEG-only development56 EW-F1, epoch0 eligible, earliest ties',
            source='PH-EMO waveform renderer reused; not an official PH-EMO or Copilot reproduction',
            model_choice='Highest recorded external video-only baseline EW-F1: released InternVL3.5-2B',
            model_choice_boundary='Architecture choice follows prior test observations; follow-up not untouched confirmation',
            waveform_renderer_sha256=sha(PROJECT / 'scripts/mllm/train_phemo_eeg.py'),
            development_boundary='Only development56 used for checkpoint selection; no test-based reselection',
            full_dataset_adaptation=False, smoke=args.smoke)
        save(out / 'protocol.json', protocol)

    def inputs(record, arm='joint'):
        if args.video_only:
            key = record['video_id']
            if key not in video_cache:
                video_cache[key] = reader(ROOT / f'stimulus_videos_v1/videos/{key}.mp4')[0]
            return video_cache[key], ()
        if args.waveform:
            image = render_waveform(signals[record['index']])
            return (), [image]
        values = np.array(logs[record['index']])
        atlases = [atlas(values[part].mean(0), weights, limits).resize((640, 192), Image.Resampling.NEAREST)
                   for part in [slice(0, 5), slice(5, 10)]] if arm != 'video' and not args.measurement_text else ()
        frames = ()
        if arm != 'eeg':
            key = record['video_id'] if args.dataset == 'faced' else record['video_path']
            if key not in video_cache:
                path = ROOT / f'stimulus_videos_v1/videos/{key}.mp4' if args.dataset == 'faced' else key
                video_cache[key] = reader(path)[0]
            frames = video_cache[key]
        return frames, atlases

    def prepare(record, arm):
        pixels, counts, question = runtime.prepare_input(*inputs(record, arm))
        if args.measurement_text:
            markers = question.split('\n' + runtime.prompt, 1)[0]
            question = markers + '\nThe video depicts the stimulus, not the participant. ' + text_prompt(np.array(logs[record['index']]))
        if args.waveform:
            question = question.replace('EEG atlas 1 (successive five-second intervals)',
                'EEG waveform (32 channels, middle10 seconds)')
        return pixels, counts, question

    def loss(record, arm):
        pixels, counts, question = prepare(record, arm)
        template = model.chat.__func__.__globals__['get_conv_template'](model.template)
        template.system_message = model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        prefix = template.get_prompt()
        for count in counts:
            prefix = prefix.replace('<image>', '<img>' + '<IMG_CONTEXT>' * model.num_image_token * count + '</img>', 1)
        tokenizer = runtime.tokenizer
        model.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        batch = tokenizer(prefix + json.dumps(record['target']) + template.sep, return_tensors='pt').to('cuda')
        prefix_ids = tokenizer(prefix, return_tensors='pt')['input_ids'].cuda()
        assert torch.equal(batch['input_ids'][:, :prefix_ids.shape[1]], prefix_ids)
        assert int((batch['input_ids'] == model.img_context_token_id).sum()) == sum(counts) * model.num_image_token
        labels = batch['input_ids'].clone()
        labels[:, :prefix_ids.shape[1]] = -100
        return model(pixel_values=pixels, **batch, labels=labels,
            image_flags=torch.ones((pixels.shape[0], 1), dtype=torch.long, device='cuda'), use_cache=False).loss

    def evaluate(records, tag, references=None, arm='joint'):
        model.eval()
        scored = []
        with (out / f'{tag}_records.jsonl').open('x') as stream:
            for number, record in enumerate(records):
                if args.measurement_text or args.waveform:
                    pixels, counts, question = prepare(record, arm)
                    with torch.inference_mode():
                        raw = model.chat(runtime.tokenizer, pixels, question,
                            dict(do_sample=False, max_new_tokens=320), num_patches_list=counts)
                    response = dict(raw=raw, image_patch_counts=counts, prompt=question)
                else:
                    response = runtime.generate(*inputs(record, arm))
                terms, valid = parser(response['raw'])
                truth = (record['required_concepts'] if 'required_concepts' in record else record['target']['emotions']) if references is None else references.get(record['sample_id'])
                row = dict(index=record['index'], subject=record['subject'],
                    emotion_terms=terms, parse_valid=valid, **response)
                if truth is not None:
                    row['required_concepts'] = truth
                    scored.append(row)
                stream.write(json.dumps(row) + '\n'); stream.flush()
                save(out / 'status.json', dict(phase='evaluating', panel=tag, completed=number + 1,
                    total=len(records), test_read=args.dataset == 'faced' and tag.startswith('final')))
        result = metric.score(scored)
        from exact import metrics
        from collections import Counter
        counts = Counter(tuple(sorted(set(row['emotion_terms']))) for row in scored)
        result['exact_set'] = metrics([(set(row['required_concepts']),
            set(row['emotion_terms']) if row['parse_valid'] else set()) for row in scored])
        result['unique_sets'] = len(counts)
        result['most_common_sets'] = counts.most_common(5)
        save(out / f'{tag}_metrics.json', result)
        return result

    if args.smoke:
        assert args.waveform, 'Smoke mode is implemented only for the waveform adapter'
        model.train()
        model.vision_model.eval()
        value = loss(train[0], 'eeg')
        assert torch.isfinite(value)
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        assert float(norm) > 0
        assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
        model.zero_grad(set_to_none=True)
        model.eval()
        pixels, counts, question = prepare(dev[0], 'eeg')
        with torch.inference_mode():
            raw = model.chat(runtime.tokenizer, pixels, question,
                dict(do_sample=False, max_new_tokens=320), num_patches_list=counts)
        save(out / 'complete.json', dict(loss=float(value.detach()), gradient_norm=float(norm),
            raw=raw, image_patch_counts=counts, test_read=False, phase='smoke_complete'))
        save(out / 'status.json', dict(phase='complete', test_read=False))
        return
    best = evaluate(dev, 'epoch0')['f1']
    selected = 0
    torch.save(get_peft_model_state_dict(model.language_model), out / 'best.pt')
    optimizer = torch.optim.AdamW(parameters, lr=rate, weight_decay=protocol['weight_decay'], eps=1e-6)
    presentations = len(train) * (2 if args.video_only or args.dataset == 'faced' and not args.measurement_text and not args.waveform else 1)
    total_updates = epochs * math.ceil(presentations / 8)
    history, update = [], 0
    for epoch in range(1, epochs + 1):
        order = [(int(index), arm) for index in rng.permutation(len(train))
                 for arm in (['video', 'video'] if args.video_only else ['eeg', 'joint'] if args.dataset == 'faced' and not args.measurement_text and not args.waveform else ['eeg' if args.waveform else 'joint'])]
        losses = []
        for offset in range(0, len(order), 8):
            model.train(); model.vision_model.eval()
            optimizer.zero_grad(set_to_none=True)
            group = order[offset:offset + 8]
            update += 1
            warmup = 6 if args.dataset == 'faced' else 8
            factor = min(1., update / warmup) if update <= warmup else .5 * (1 + math.cos(
                math.pi * (update - warmup) / (total_updates - warmup)))
            optimizer.param_groups[0]['lr'] = rate * factor
            for index, arm in group:
                value = loss(train[index], arm)
                assert torch.isfinite(value), 'Nonfinite training loss'
                losses.append(float(value.detach()))
                (value / len(group)).backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            if update == 1:
                assert float(norm) > 0
            optimizer.step()
            save(out / 'status.json', dict(phase='training', epoch=epoch, update=update,
                total_updates=total_updates, loss=float(np.mean(losses[-len(group):])), test_read=False))
        result = evaluate(dev, f'epoch{epoch}')
        if result['f1'] > best:
            best, selected = result['f1'], epoch
            torch.save(get_peft_model_state_dict(model.language_model), out / 'best.pt')
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), development=result))
        save(out / 'history.json', history)
    set_peft_model_state_dict(model.language_model, torch.load(out / 'best.pt', map_location='cpu', weights_only=True))
    references = None
    if args.dataset == 'faced':
        save(out / 'selection_lock.json', dict(epoch=selected, checkpoint_sha256=sha(out / 'best.pt'), test_read=False))
        if args.measurement_text or args.waveform or args.video_only:
            evaluate(datasets['val'], 'selected_full560')
        test = ROOT / 'final_test_20261120'
        final = read(test / 'test_records.jsonl')
        assert len(final) == 644 and {row['subject'] for row in final} == set(range(100, 123))
        if args.waveform:
            from scipy import signal
            test_signals = source_function(PROJECT / 'scripts/mllm/train_phemo_eeg.py',
                'test_signals', dict(ROOT=ROOT, sha=sha, json=json, ast=ast, np=np,
                                     zipfile=zipfile, pickle=pickle, signal=signal))
            signals = test_signals(final)
        logs = np.load(test / 'band_logpower.npy', mmap_mode='r')
        results = {arm: evaluate(final, f'final_{arm}', arm=arm)
                   for arm in (['video'] if args.video_only else ['eeg'] if args.waveform else ['joint'] if args.measurement_text else ['joint', 'eeg', 'video'])}
    else:
        final = read(cache / 'val_records.jsonl')
        references = {row['sample_id']: row['target']['emotions'] for row in
                      read(data_root / 'heldout700_reference_v1/trial_reference.jsonl')}
        assert len(final) == 700 and len(references) == 300
        assert {row['subject'] for row in final} == set(range(29, 36))
        results = dict(joint=evaluate(final, 'final_joint', references))
    save(out / 'complete.json', dict(selected_epoch=selected, development_f1=best, results=results))
    save(out / 'status.json', dict(phase='complete', training=False))
    lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['faced', 'eav'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--measurement-text', action='store_true')
    parser.add_argument('--waveform', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--video-only', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        root = ROOT if args.dataset == 'faced' else Path('/data/model/datasets/EAV_13799131')
        out = (root / 'eeg_llm_families_v1/internvl_waveform_v1' /
               ('smoke' if args.smoke else 'adaptation')) if args.waveform else (
               root / ('additional_text_baselines_lora_v1' if args.measurement_text else 'additional_baselines_lora_v1') / 'internvl35')
        if args.video_only:
            out = root / 'native_video_only_adaptation_v1/internvl35'
        if out.exists():
            save(out / 'failure.json', dict(error=traceback.format_exc()))
            save(out / 'status.json', dict(phase='failed'))
        raise
