"""FACED continuation of the official AffectGPT language adapters; no new modalities."""
import ast
import argparse
import importlib.util
import json
import math
import time
import traceback

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from common import ROOT, PROJECT, panel, prediction_row, save, sha, write_rows
from eval_mllm import parse_answer

BASE = ROOT / 'affective_zeroshot_v1'
OUT = BASE / 'affectgpt_faced_lora_v1'


def setup():
    path = BASE / 'run_affectgpt_official.py'
    spec = importlib.util.spec_from_file_location('official_affectgpt_runtime', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    prefix = []
    for node in function.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'records' for target in node.targets):
            break
        prefix.append(node)
    function.body = prefix + ast.parse('return locals()').body
    function.name = 'load_only'
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), 'exec'), module.__dict__)
    return module.load_only()


def main(video_only=False):
    assert not OUT.exists(), 'Refusing to overwrite a prior run'
    state = setup()
    model, chat = state['model'], state['chat']
    for name, parameter in model.named_parameters():
        parameter.requires_grad_('llama_model.' in name and 'lora_' in name)
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert parameters
    model.llama_model.config.use_cache = False
    model.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    datasets = panel()
    rng = np.random.default_rng(20261109)
    train = []
    for subject in range(80):
        pool = [row for row in datasets['train'] if row['subject'] == subject]
        train.extend(pool[int(index)] for index in rng.choice(len(pool), 3, replace=False))
    expected = [json.loads(line) for line in (ROOT / 'v2_video_joint_lora_pilot_v1/train_records.jsonl').read_text().splitlines()]
    assert train == expected and len(train) == 240
    selection = [row for row in datasets['val'] if row['subject'] in [80, 81]]
    assert len(selection) == 56
    write_rows(OUT / 'train_records.jsonl', train)
    write_rows(OUT / 'selection_records.jsonl', selection)
    logs = None if video_only else np.load(ROOT / 'official200_classification_v1/band_logpower.npy', mmap_mode='r')
    metric = state['evaluator']() if 'evaluator' in state else None
    if metric is None:
        from common import evaluator
        metric = evaluator()
    cache = {}

    def prompt(record, joint):
        if video_only:
            from eval_video import PROMPT
            return '###Human: <Video><FrameHere></Video>. ' + PROMPT + ' ###Assistant:'
        values = np.array(logs[record['index']])
        measurements = [values[part].mean(axis=(0, 2)).round(4).tolist() for part in [slice(0, 5), slice(5, 10)]]
        visual = ('We uniformly sample raw frames from the video: <Video><FrameHere></Video>. '
                  'The video depicts the stimulus, not the participant. ') if joint else ''
        return ('###Human: ' + visual + 'EEG log-band-power channel means '
                '(delta, theta, alpha, beta, gamma) for successive five-second windows: '
                + json.dumps(measurements) + '. These measurements do not uniquely establish emotions. '
                'Infer possible participant emotions, using unrestricted vocabulary. '
                'Return only JSON {"emotions":["emotion expression", ...]}. ###Assistant:')

    def visual_features(record):
        video_id = record['video_id']
        if video_id not in cache:
            frames, metadata = state['video_namespace']['video_frames'](ROOT / f'stimulus_videos_v1/videos/{video_id}.mp4')
            save(OUT / f'video_{video_id}.json', metadata)
            raw = torch.tensor(frames).permute(3, 0, 1, 2).unsqueeze(0).cuda()
            with torch.no_grad():
                _, cache[video_id] = model.encode_video_merge(raw, raw)
        return cache[video_id]

    def loss(record, joint):
        prefix = chat.replace_token_for_multimodal(prompt(record, joint))
        prefix_ids = chat.to_token_ids(prefix, 2000).cuda()
        answer = json.dumps(record['target']) + model.llama_tokenizer.eos_token
        target_ids = model.llama_tokenizer(answer, add_special_tokens=False, return_tensors='pt').input_ids[0].cuda()
        ids = torch.cat([prefix_ids, target_ids])
        assert len(ids) < 2000
        embedded_ids = ids.clone()
        positions = torch.where(ids == model.FRAME_PATCH_TOKEN_ID)[0]
        embedded_ids[positions] = 0
        embeddings = model.llama_model.get_input_embeddings()(embedded_ids).requires_grad_(True)
        if joint:
            assert len(positions) == chat.num_video_query_token
            features = visual_features(record)[0].detach().to(embeddings.dtype)
            embeddings = torch.cat([embeddings[:positions[0]], features, embeddings[positions[-1] + 1:]])
        else:
            assert len(positions) == 0
        labels = ids.clone()
        labels[:len(prefix_ids)] = -100
        with torch.autocast('cuda', dtype=torch.float16):
            return model.llama_model(inputs_embeds=embeddings.unsqueeze(0),
                attention_mask=torch.ones_like(ids).unsqueeze(0), labels=labels.unsqueeze(0), use_cache=False).loss

    def evaluate(records, tag, epoch, test_read=False):
        model.eval()
        scores = {}
        with torch.no_grad():
            for arm in (['video'] if video_only else ['eeg', 'joint']):
                rows = []
                with (OUT / f'{tag}_{arm}_records.jsonl').open('x') as stream:
                    for number, record in enumerate(records):
                        raw = chat.answer_sample(prompt(record, arm != 'eeg'),
                            dict(frame=visual_features(record) if arm != 'eeg' else None,
                                 face=None, audio=None, multi=None, image=None),
                            do_sample=False, max_new_tokens=320, max_length=2000)
                        terms, valid = parse_answer(raw)
                        row = prediction_row(record, terms, valid, raw=raw)
                        rows.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                        stream.flush()
                        save(OUT / 'status.json', {'phase': 'evaluating', 'epoch': epoch, 'panel': tag,
                            'arm': arm, 'completed': number + 1, 'total': len(records), 'test_read': test_read})
                scores[arm] = metric.score(rows)
                if video_only:
                    from collections import Counter
                    from exact import metrics
                    counts = Counter(tuple(sorted(set(row['emotion_terms']))) for row in rows)
                    scores[arm].update(exact_set=metrics([(set(row['required_concepts']),
                        set(row['emotion_terms']) if row['parse_valid'] else set()) for row in rows]),
                        unique_sets=len(counts), most_common_sets=counts.most_common(5))
        save(OUT / f'{tag}_metrics.json', scores)
        return scores

    protocol = {'training_trials': 240, 'selection_trials': 56, 'evaluation_trials': 560,
        'epochs': 2, 'updates': 120, 'effective_batch': 8, 'seed': 20261109,
        'initialization': 'Official AffectGPT epoch60 including original language LoRA',
        'trainable': 'Existing language LoRA only; vision, acoustic, fusion and language base frozen',
        'trainable_parameters': sum(parameter.numel() for parameter in parameters),
        'input': 'Eight stimulus frames plus same EEG measurement text as zero-shot; mixed EEG-only/joint',
        'budget_match': 'Same240train/56selection/120updates as native adapted baselines; not matched parameter count or lifetime cost',
        'optimizer': 'AdamW1e-5 weight_decay0.01,6warmup updates,cosine decay,clip1',
        'target': 'Same trial emotion JSON plus EOS; assistant-only causal CE',
        'selection': 'Joint56EW-F1 then EEG56EW-F1; epoch0 eligible; no full560 reselection',
        'generation': {'do_sample': False, 'max_new_tokens': 320}, 'test_read': False,
        'script_sha256': sha(__file__), 'runtime_sha256': sha(BASE / 'run_affectgpt_official.py'),
        'metric': metric.protocol, 'audio': None}
    if video_only:
        protocol.update(input='Eight stimulus frames only; no EEG measurements, atlases, audio or classifier',
            mix='Each training trial twice per nominal epoch;480presentations/epoch,960total,120updates',
            selection='Video-only development56 EW-F1; epoch0 eligible; earliest ties; lock before test',
            comparison_boundary='Official AffectGPT existing language LoRA;120updates,not BrainRel lifetime matching',
            development_boundary='Follow-up after prior test inspection; not untouched confirmatory evaluation')
    save(OUT / 'protocol.json', protocol)
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    started = time.time()
    history = []

    def checkpoint(epoch):
        torch.save({'lora': get_peft_model_state_dict(model.llama_model), 'epoch': epoch,
                    'protocol': protocol}, OUT / 'best.pt')

    initial = evaluate(selection, 'epoch0', 0)
    best = (initial['video']['f1'],) if video_only else (initial['joint']['f1'], initial['eeg']['f1'])
    selected_epoch = 0
    checkpoint(0)
    update = 0
    for epoch in [1, 2]:
        order = rng.permutation(len(train))
        model.llama_model.train()
        losses = []
        for offset in range(0, len(order), 4):
            optimizer.zero_grad(set_to_none=True)
            group = [(train[int(index)], joint) for index in order[offset:offset + 4]
                     for joint in ([True, True] if video_only else [False, True])]
            assert len(group) == 8
            group = [group[int(index)] for index in rng.permutation(len(group))]
            rate = (update + 1) / 6 if update < 6 else 0.5 * (1 + math.cos(math.pi * (update - 6) / 114))
            for options in optimizer.param_groups:
                options['lr'] = 1e-5 * rate
            for record, joint in group:
                value = loss(record, joint)
                assert torch.isfinite(value), 'Nonfinite loss'
                losses.append(float(value.detach()))
                scaler.scale(value / 8).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            assert torch.isfinite(norm), 'Nonfinite gradients'
            if update == 0:
                assert norm > 0
                save(OUT / 'gradient_smoke.json', {'loss': losses[-1], 'gradient_norm': float(norm),
                    'frozen_gradient_count': sum(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad)})
            scaler.step(optimizer)
            scaler.update()
            update += 1
            save(OUT / 'status.json', {'phase': 'training', 'epoch': epoch, 'update': update,
                 'total_updates': 120, 'loss': losses[-1], 'test_read': False})
        scores = evaluate(selection, f'epoch{epoch}', epoch)
        history.append({'epoch': epoch, 'train_ce': float(np.mean(losses)), 'scores': scores})
        save(OUT / 'history.json', history)
        key = (scores['video']['f1'],) if video_only else (scores['joint']['f1'], scores['eeg']['f1'])
        if key > best:
            best, selected_epoch = key, epoch
            checkpoint(epoch)
    saved = torch.load(OUT / 'best.pt', map_location='cpu', weights_only=False)
    result = set_peft_model_state_dict(model.llama_model, saved['lora'])
    assert not result.unexpected_keys
    if video_only:
        save(OUT / 'selection_lock.json', dict(epoch=selected_epoch,
            checkpoint_sha256=sha(OUT / 'best.pt'), test_read=False))
    scores = evaluate(datasets['val'], 'selected_full560', selected_epoch)
    test = None
    if video_only:
        records = [json.loads(line) for line in (ROOT / 'final_test_20261120/test_records.jsonl').read_text().splitlines()]
        assert len(records) == 644 and {row['subject'] for row in records} == set(range(100, 123))
        test = evaluate(records, 'selected_test644', selected_epoch, test_read=True)
    save(OUT / 'complete.json', {'selected_epoch': selected_epoch, 'updates': update,
        'scores': scores, 'test644': test, 'seconds': time.time() - started, 'test_read': video_only})
    save(OUT / 'status.json', {'phase': 'complete', 'test_read': video_only})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-only', action='store_true')
    args = parser.parse_args()
    if args.video_only:
        OUT = ROOT / 'native_video_only_adaptation_v1/affectgpt'
    try:
        main(video_only=args.video_only)
    except Exception:
        save(OUT / 'failure.json', {'error': traceback.format_exc(), 'test_read': False})
        raise
