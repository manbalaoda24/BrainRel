"""Isolated self-report weak-supervision pilot with visual-reading rehearsal."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, get_peft_model_state_dict

from run import ROOT, DATA, MODEL, BANDS, atlas
from regions import render, CANVAS
from weak_labels import EMOTIONS, ITEMS, FORMS, complete_target

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'encoder'))
from model import QueryFlattenHead

OUT = ROOT / 'open_self_report_sft_v1'
READER = ROOT / 'region_binding_sft_v1'
REGEX = r'model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|out_proj))'
SEED = 20261111
CLASS_NAMES = ['anger', 'disgust', 'fear', 'sadness', 'neutral', 'amusement', 'inspiration', 'joy', 'tenderness']


def save(name, value):
    path = OUT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def jsonlines(name, rows):
    (OUT / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def summarize(rows):
    true_positive = false_positive = false_negative = exact = invalid = extras = 0
    for row in rows:
        predicted = set(row['mapped_concepts'])
        target = set(row['required_concepts'])
        true_positive += len(predicted & target)
        false_positive += len(predicted - target)
        false_negative += len(target - predicted)
        exact += predicted == target and row['parse_valid'] and not row['unassessed_terms']
        invalid += not row['parse_valid']
        extras += len(row['unassessed_terms'])
    denominator = 2 * true_positive + false_positive + false_negative
    return {'n': len(rows), 'micro_f1_eight_concepts': 2 * true_positive / denominator if denominator else 0.,
            'precision': true_positive / max(1, true_positive + false_positive),
            'recall': true_positive / max(1, true_positive + false_negative),
            'strict_set_exact': exact / max(1, len(rows)), 'invalid': invalid,
            'unassessed_extra_terms': extras,
            'boundary': 'Eight self-report concepts only; extra vocabulary not judged false and not OV-MER EW score'}


def main():
    OUT.mkdir(exist_ok=False)
    start = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    source = ROOT / 'self_report_join_audit_v1/trial_self_reports.jsonl'
    self_reports = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(self_reports) == 2800 and all(row['subject'] < 100 for row in self_reports)
    teacher = [json.loads(line) for line in (ROOT / 'self_report_weak_samples_v2/samples.jsonl').read_text().splitlines()]
    optional = {(row['subject'], row['video_id']): row['accepted'] for row in teacher}
    datasets = {'train': [], 'val': []}
    full_training_targets = []
    for subject in range(100):
        pool = sorted([row for row in self_reports if row['subject'] == subject], key=lambda row: row['video_id'])
        assert len(pool) == 28
        selected = set(rng.choice(28, 3, replace=False).tolist())
        for number, row in enumerate(pool):
            scores = {name: row['score_original_order'][ITEMS.index(name)] for name in EMOTIONS}
            split = 'train' if subject < 80 else 'val'
            assert row['split'] == split
            target = complete_target(scores, optional.get((subject, row['video_id']), []) if split == 'train' else [])
            record = {'task': 'open', 'subject': subject, 'video_id': row['video_id'], 'split': split,
                      'index': subject * 84 + (row['video_id'] - 1) * 3 + 1,
                      'required_concepts': target['required_concept_ids'],
                      'target': {'emotions': [label['surface_form'] for label in target['target_labels']]},
                      'target_labels': target['target_labels']}
            if split == 'train':
                full_training_targets.append(record)
            if number in selected:
                datasets[split].append(record)
    assert len(full_training_targets) == 2240
    jsonlines('all_train_weak_targets.jsonl', full_training_targets)
    for split in datasets:
        jsonlines(split + '_open_records.jsonl', datasets[split])
    measurements = np.load(ROOT / 'traceable_visual_alignment_v1/measurements.npz')
    limits = measurements['limits']
    weights = np.load(DATA / 'topology_geometry.npz')['weights'].astype('float32')
    logs = np.load(DATA / 'band_logpower.npy', mmap_mode='r')
    assert logs.shape == (8400, 10, 5, 30)
    predictions = json.loads((READER / 'classifier_predictions.json').read_text())
    classifier_path = ROOT / 'readout_recipe_2x2_v1/full_borrowed/best.pt'
    classifier = QueryFlattenHead(64, 480).cuda().eval().requires_grad_(False)
    checkpoint = torch.load(classifier_path, map_location='cpu', weights_only=False)
    classifier.load_state_dict({key.removeprefix('classifier.'): value for key, value in checkpoint['model'].items() if key.startswith('classifier.')})
    features = np.load(ROOT / 'original480_anchor_v1/frozen_tokens.npy', mmap_mode='r')
    needed = [record['index'] for split in datasets.values() for record in split]
    with torch.inference_mode():
        for offset in range(0, len(needed), 32):
            indices = needed[offset:offset + 32]
            labels = classifier(torch.tensor(np.array(features[indices]), device='cuda')).argmax(-1).tolist()
            predictions.update({str(index): {'label': CLASS_NAMES[label]} for index, label in zip(indices, labels)})
    del classifier, checkpoint, features
    torch.cuda.empty_cache()
    for split in datasets:
        saved = [json.loads(line) for line in (READER / (split + '_records.jsonl')).read_text().splitlines()]
        replay = []
        for task in ['position', 'change']:
            for label in (['R1_HIGHER', 'R2_HIGHER', 'SIMILAR'] if task == 'position' else ['INCREASE', 'DECREASE', 'SIMILAR']):
                pool = [record for record in saved if record['task'] == task and record['target'] == label]
                count = 20 if split == 'train' else 5
                for index in rng.choice(len(pool), count, replace=False):
                    record = pool[int(index)]
                    replay.append({**record, 'required_concepts': [], 'target': {'emotion': predictions[str(record['index'])]['label'], 'evidence': record['target']}})
        datasets[split].extend(replay)
        jsonlines(split + '_mixed_records.jsonl', datasets[split])
    protocol = {'scope': 'Self-report-anchored open expression pilot, not independent unrestricted emotion truth',
                'train_open_trials': 240, 'val_open_trials': 60, 'train_rehearsal': 120, 'val_rehearsal': 30,
                'split': 'Subjects0-79 train;80-99 validation; no test. Three random trials per subject, fixed seed; middle 10s segment per trial; retrospective trial labels are weak segment supervision.',
                'target': 'Self-report >=3.5 required concepts, canonical completion; optional fixed morphological forms from audited train teacher outputs; no teacher free prose or intensity targets.',
                'inputs': 'Two five-band maps averaged across first/last five seconds of middle segment, frozen classifier prediction; no self-ratings, ground truth, subject/video ID or measured target in prompt.',
                'reading_rehearsal': '120 saved region/time tasks per epoch, original deterministic targets and predicted classifier field; not emotion causal explanations.',
                'initial_checkpoint': str(READER / 'best.pt'), 'checkpoint_sha256': hashlib.sha256((READER / 'best.pt').read_bytes()).hexdigest(),
                'trainable': 'Existing language attention LoRA rank16 only; native vision merger and all other weights frozen; original classifier untouched.',
                'epochs': 3, 'effective_batch': 8, 'microbatch': 1, 'learning_rate': 2e-5,
                'schedule': '5% warmup then cosine decay', 'selection': 'validation eight-concept micro F1; tie lower open-task validation CE; initial checkpoint included',
                'output': 'JSON emotions list with unrestricted expressions and count, empty permitted; no supplied candidate vocabulary.',
                'controls': 'before/after actual; selected checkpoint prior-only, evidence-only and cyclic mismatched images; direct frozen classifier and train-most-common-set baselines',
                'evaluation': 'Local conservative morphological mapping, not independent LLM extraction; save all raw generations for later independent extraction; extras unassessed; no OV-MER EW claim.',
                'test_read': False, 'seed': SEED, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save('protocol.json', protocol)
    save('status.json', {'phase': 'loading_model'})
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa', local_files_only=True)
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0., target_modules=REGEX, bias='none', task_type='CAUSAL_LM'))
    checkpoint = torch.load(READER / 'best.pt', map_location='cpu', weights_only=False)
    result = set_peft_model_state_dict(model, checkpoint['lora'])
    assert not result.unexpected_keys
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in checkpoint['merger']:
                parameter.copy_(checkpoint['merger'][name].to(parameter.device))
            parameter.requires_grad_('lora_' in name)
    del checkpoint
    model.eval()
    model.config.use_cache = False
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    imagecache = {}

    def pictures(record, condition):
        index = record['index']
        if condition == 'mismatch':
            pool = [item for item in datasets['val'] if item['task'] == 'open']
            offset = next(offset for offset, item in enumerate(pool) if item['index'] == index)
            index = pool[(offset + 1) % len(pool)]['index']
        if record['task'] == 'open':
            if index not in imagecache:
                values = np.array(logs[index])
                imagecache[index] = [atlas(values[part].mean(0), weights, limits).resize((640, 192), Image.Resampling.NEAREST) for part in [slice(0, 5), slice(5, 10)]]
            return imagecache[index]
        window, band = record['window'], record['band']
        panels = (np.array(logs[index]) @ weights.T).reshape(10, 5, 16, 16)
        displayed = np.clip((panels - limits[0][None, :, None, None]) / (limits[1] - limits[0])[None, :, None, None], 0, 1)
        if record['task'] == 'position':
            query = displayed[window].copy()
            if record['op'] == 'flip':
                query[band] = np.flip(query[band], 0)
            elif record['op'] == 'symmetric':
                query[band] = (query[band] + np.flip(query[band], 0)) / 2
            return [render(query, band, record['base_group'], [item[0] for item in BANDS], swapped=record['id_swapped'])[0]]
        pair = [window - 1, window] if record['op'] == 'identity' else ([window, window - 1] if record['op'] == 'swap' else [window - 1, window - 1])
        return [atlas(np.array(logs[index, part]), weights, limits).resize((640, 192), Image.Resampling.NEAREST) for part in pair]

    def batch_for(record, answer=False, condition='actual'):
        prior = predictions[str(record['index'])]['label']
        if record['task'] == 'open':
            images = [] if condition == 'prior_only' else pictures(record, condition)
            prompt = ('Predict the participant\'s emotional states from the available fallible model clues. Return only JSON {"emotions":["emotion expression", ...]}. Vocabulary and number of emotions are unrestricted; an empty list is permitted. Do not invent events, diagnoses, physiological causes, or claim EEG uniquely establishes an emotion. ')
            if images:
                prompt += 'The two fixed-scale five-band scalp maps summarize seconds0-5 and5-10 of one segment. Red is higher power and blue lower; maps are measurements, not confirmed emotional labels. '
            if condition != 'evidence_only':
                prompt += 'A separate frozen EEG classifier predicted ' + prior + '. This is a fallible model prior, not participant self-report. '
        else:
            images = pictures(record, 'actual')
            band = BANDS[record['band']][0]
            prompt = 'Fixed scales, red higher and blue lower. '
            if record['task'] == 'position':
                prompt += f'In the {band} panel compare the mean displayed power in marked regions R1 and R2. If the normalized mean difference is within0.03 reply SIMILAR, otherwise R1_HIGHER or R2_HIGHER for the higher region. '
            else:
                prompt += f'In the {band} panel compare second(query) minus first(reference) full-map mean. Within0.05 reply SIMILAR, otherwise INCREASE or DECREASE. '
            prompt += 'A separate frozen classifier supplied classifier_label: ' + prior + '. Copy it, do not infer emotion from maps. Reply JSON with emotion equal to classifier_label and evidence equal to the visual comparison label.'
        messages = [{'role': 'user', 'content': [{'type': 'image'} for image in images] + [{'type': 'text', 'text': prompt}]}]
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = processor.apply_chat_template(messages + [{'role': 'assistant', 'content': json.dumps(record['target'])}], tokenize=False, add_generation_prompt=False, enable_thinking=False) if answer else prefix
        options = {'images': images, 'min_pixels': 320 * 96, 'max_pixels': CANVAS[0] * CANVAS[1]} if images else {}
        batch = processor(text=[full], return_tensors='pt', **options)
        if answer:
            prefix_ids = processor(text=[prefix], return_tensors='pt', **options)['input_ids']
            count = prefix_ids.shape[1]
            assert torch.equal(batch['input_ids'][:, :count], prefix_ids)
            labels = batch['input_ids'].clone()
            labels[:, :count] = -100
            assert (labels != -100).any()
            batch['labels'] = labels
        base = model.get_base_model().model
        if hasattr(base, 'rope_deltas'):
            base.rope_deltas = None
        return batch.to('cuda')

    lookup = {form: concept for concept, forms in FORMS.items() for form in forms}

    @torch.no_grad()
    def evaluate(tag, condition='actual'):
        results = []
        reading = []
        ce = 0.
        model.eval()
        for record in datasets['val']:
            if record['task'] != 'open' and condition != 'actual':
                continue
            if condition == 'actual' and record['task'] == 'open':
                ce += float(model(**batch_for(record, True), use_cache=False).loss)
            batch = batch_for(record, condition=condition)
            output = model.generate(**batch, do_sample=False, max_new_tokens=128)
            raw = tokenizer.decode(output[0, batch['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            try:
                cleaned = raw.split('\n', 1)[1].rsplit('```', 1)[0] if raw.startswith('```') and raw.endswith('```') else raw
                parsed = json.loads(cleaned)
                if not isinstance(parsed, dict):
                    raise ValueError('not object')
            except (ValueError, IndexError):
                parsed = {}
            if record['task'] != 'open':
                reading.append({'task': record['task'], 'index': record['index'], 'raw': raw, 'target': record['target'],
                                'correct_evidence': parsed.get('evidence') == record['target']['evidence'],
                                'correct_copy': parsed.get('emotion') == record['target']['emotion']})
                continue
            terms = parsed.get('emotions')
            valid = isinstance(terms, list) and all(isinstance(term, str) for term in terms)
            terms = [term.strip().lower() for term in terms] if valid else []
            results.append({'index': record['index'], 'subject': record['subject'], 'video_id': record['video_id'],
                            'required_concepts': record['required_concepts'], 'raw': raw, 'parse_valid': valid,
                            'emotion_terms': terms, 'mapped_concepts': sorted({lookup[term] for term in terms if term in lookup}),
                            'unassessed_terms': [term for term in terms if term not in lookup]})
        metrics = summarize(results)
        if condition == 'actual':
            metrics['open_ce'] = ce / len(results)
            metrics['reading'] = {task: {'n': len(subset), 'accuracy': sum(item['correct_evidence'] for item in subset) / len(subset),
                                         'copy_accuracy': sum(item['correct_copy'] for item in subset) / len(subset)}
                                  for task in ['position', 'change'] if (subset := [item for item in reading if item['task'] == task])}
        jsonlines(tag + '.jsonl', results)
        jsonlines(tag + '_reading.jsonl', reading)
        save(tag + '_metrics.json', metrics)
        return metrics

    def checkpoint(epoch):
        return {'lora': get_peft_model_state_dict(model), 'epoch': epoch, 'protocol': protocol,
                'merger': {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if '.visual.merger.' in name}}

    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0., eps=1e-6)
    total = math.ceil(len(datasets['train']) / 8) * 3
    warm = max(1, math.ceil(total * .05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / warm if step < warm else .5 * (1 + math.cos(math.pi * min(1, (step - warm) / max(1, total - warm)))))
    preflight = model(**batch_for(datasets['train'][0], True), use_cache=False).loss
    preflight.backward()
    norm = sum(float(parameter.grad.float().norm()) for parameter in parameters if parameter.grad is not None)
    assert norm > 0 and all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    save('preflight.json', {'loss': float(preflight.detach()), 'lora_gradient_norm_sum': norm, 'frozen_gradients_absent': True})
    optimizer.zero_grad(set_to_none=True)
    del preflight
    save('status.json', {'phase': 'initial_validation'})
    initial = evaluate('initial_actual')
    best = (initial['micro_f1_eight_concepts'], -initial['open_ce'])
    torch.save(checkpoint(0), OUT / 'best.pt')
    history = []
    updates = 0
    for epoch in range(1, 4):
        order = np.random.default_rng(SEED + epoch).permutation(len(datasets['train']))
        loss_sum = 0.
        for offset in range(0, len(order), 8):
            group = order[offset:offset + 8]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                loss = model(**batch_for(datasets['train'][int(index)], True), use_cache=False).loss
                assert torch.isfinite(loss)
                (loss / len(group)).backward()
                loss_sum += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            updates += 1
            save('status.json', {'phase': 'training', 'epoch': epoch, 'updates': updates, 'total_updates': total,
                                 'lr': optimizer.param_groups[0]['lr'], 'seconds': time.time() - start})
        metrics = evaluate(f'epoch_{epoch:02d}_actual')
        history.append({'epoch': epoch, 'train_ce': loss_sum / len(order), 'validation': metrics, 'updates': updates})
        save('history.json', history)
        torch.save(checkpoint(epoch), OUT / 'last.pt')
        if (metrics['micro_f1_eight_concepts'], -metrics['open_ce']) > best:
            best = (metrics['micro_f1_eight_concepts'], -metrics['open_ce'])
            torch.save(checkpoint(epoch), OUT / 'best.pt')
    selected = torch.load(OUT / 'best.pt', map_location='cpu', weights_only=False)
    set_peft_model_state_dict(model, selected['lora'])
    controls = {condition: evaluate('selected_' + condition, condition) for condition in ['actual', 'prior_only', 'evidence_only', 'mismatch']}
    baseline_rows = []
    for record in datasets['val']:
        if record['task'] == 'open':
            prior = predictions[str(record['index'])]['label']
            baseline_rows.append({**record, 'mapped_concepts': [prior] if prior in EMOTIONS else [], 'parse_valid': True, 'unassessed_terms': [] if prior in EMOTIONS else [prior]})
    jsonlines('direct_classifier_baseline.jsonl', baseline_rows)
    counts = {}
    for record in full_training_targets:
        key = tuple(record['required_concepts'])
        counts[key] = counts.get(key, 0) + 1
    common = max(counts, key=counts.get)
    constant = [{**record, 'mapped_concepts': list(common), 'unassessed_terms': []} for record in baseline_rows]
    save('complete.json', {'phase': 'complete', 'initial': initial, 'best_epoch': selected['epoch'], 'controls': controls,
                           'direct_classifier': summarize(baseline_rows), 'train_most_common_set': {'concepts': list(common), **summarize(constant)},
                           'seconds': time.time() - start, 'updates': updates, 'test_read': False})
    save('status.json', {'phase': 'complete'})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if OUT.exists():
            save('status.json', {'phase': 'failed', 'traceback': traceback.format_exc()})
        raise
