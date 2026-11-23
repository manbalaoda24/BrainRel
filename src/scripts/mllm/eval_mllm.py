"""Original released MLLMs on fixed EEG-derived images, without fine-tuning."""
import argparse
import json
import time
import traceback

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from common import ROOT, PROJECT, OUTPUT, SEED, evaluator, lock_gpu, panel, prediction_row, save, sha


def parse_answer(raw):
    decoder = json.JSONDecoder()
    for offset, character in enumerate(raw):
        if character != '{':
            continue
        try:
            value, _ = decoder.raw_decode(raw[offset:])
        except json.JSONDecodeError:
            continue
        terms = value.get('emotions') if isinstance(value, dict) else None
        if isinstance(terms, list) and all(isinstance(term, str) for term in terms):
            return list(dict.fromkeys(term.strip().lower() for term in terms if term.strip())), True
    return [], False


def prompt_for(record):
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band = bands[record['description_task']['band']]
    return ('Predict the participant\'s emotional states from the available fallible model clues. Return only JSON {"emotions":["emotion expression", ...]}. Vocabulary and number of emotions are unrestricted; an empty list is permitted. Do not invent events, diagnoses, physiological causes, or claim EEG uniquely establishes an emotion. '
            f'The measurement query concerns {band}: compare the upper versus lower image halves in the first map, '
            'then the second minus first full-map mean. Normalized mean gaps within0.03 spatially and0.05 temporally count as similar. '
            'These are image regions, not anatomical lobes; measurements alone do not establish an emotion. '
            'The two fixed-scale five-band scalp maps summarize seconds0-5 and5-10 of one segment. Red is higher power and blue lower; maps are measurements, not confirmed emotional labels. ')


def main(arguments, out):
    gpu_lock = lock_gpu()
    out.mkdir(exist_ok=False)
    started = time.time()
    save(out / 'status.json', {'phase': 'loading_model', 'test_read': False})
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    records = panel()['val']
    metric = evaluator()
    from run import atlas
    model_path = PROJECT / 'oneemo/weights' if arguments.model == 'oneemo_original' else OUTPUT / 'assets/qwen25vl3b'
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(str(model_path), dtype=torch.bfloat16,
                                                      device_map={'': 'cuda'}, attn_implementation='sdpa',
                                                      local_files_only=True).eval().requires_grad_(False)
    protocol = {'model': arguments.model, 'model_path': str(model_path), 'training': False,
                'tier': 'Zero-shot original released checkpoint; not budget-matched to trained V2',
                'input': 'Same two EEG measurement maps; no independent video, classifier prior or self-report input',
                'panel': 'Same560 validation trials and middle10s as V2; no test read', 'test_read': False,
                'generation': {'do_sample': False, 'max_new_tokens': 320, 'enable_thinking': False},
                'image_resize': [640, 192], 'processor_min_pixels': 320 * 96, 'processor_max_pixels': 832 * 224,
                'prompt': 'Exact V2 image-arm emotion-only prompt including independently selected measurement query',
                'parsing': 'Explicit JSON emotions field only; no keyword matching or inferred emotions; invalid counts reported',
                'reference': 'Self-report eight-concept threshold3.5, not unrestricted independently annotated gold',
                'metric': metric.protocol, 'source_sha256': sha(__file__),
                'panel_sha256': sha(ROOT / 'numeric_evidence_filter_full_v1/val_open_records.jsonl'),
                'model_config_sha256': sha(model_path / 'config.json')}
    asset_path = OUTPUT / 'assets/qwen25vl3b.asset.json'
    if arguments.model != 'oneemo_original' and asset_path.exists():
        protocol['model_revision'] = json.loads(asset_path.read_text())
    save(out / 'protocol.json', protocol)
    measurements = np.load(ROOT / 'traceable_visual_alignment_v1/measurements.npz')
    limits = measurements['limits']
    weights = np.load(ROOT / 'official200_classification_v1/topology_geometry.npz')['weights'].astype('float32')
    logs = np.load(ROOT / 'official200_classification_v1/band_logpower.npy', mmap_mode='r')
    rows = []
    with (out / 'selected_actual.jsonl').open('x') as stream:
        for number, record in enumerate(records):
            values = np.array(logs[record['index']])
            images = [atlas(values[part].mean(0), weights, limits).resize((640, 192), Image.Resampling.NEAREST)
                      for part in [slice(0, 5), slice(5, 10)]]
            prompt = prompt_for(record)
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'image'}, {'type': 'text', 'text': prompt}]}]
            prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            batch = processor(text=[prefix], images=images, return_tensors='pt',
                              min_pixels=320 * 96, max_pixels=832 * 224).to('cuda')
            with torch.inference_mode():
                generated = model.generate(**batch, do_sample=False, max_new_tokens=320)
            tail = generated[0, batch['input_ids'].shape[1]:]
            raw = processor.decode(tail, skip_special_tokens=True)
            terms, valid = parse_answer(raw)
            row = prediction_row(record, terms, valid, raw=raw, generated_tokens=len(tail),
                                 budget_reached=len(tail) == 320)
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            save(out / 'status.json', {'phase': 'generating', 'completed': number + 1,
                                     'total': 560, 'seconds': time.time() - started, 'test_read': False})
            if number % 20 == 0:
                print('GENERATED', arguments.model, number + 1, flush=True)
    scores = metric.score(rows)
    save(out / 'selected_metrics.json', {'official_ew': scores, 'truncated_budget_count': sum(row['budget_reached'] for row in rows)})
    save(out / 'complete.json', {'phase': 'complete', 'official_ew': scores, 'seconds': time.time() - started,
                                'test_read': False, 'training': False})
    save(out / 'status.json', {'phase': 'complete', 'test_read': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['oneemo_original', 'qwen25vl3b'])
    arguments = parser.parse_args()
    destination = OUTPUT / arguments.model
    try:
        main(arguments, destination)
    except Exception as error:
        if destination.exists() and not (destination / 'complete.json').exists():
            save(destination / 'status.json', {'phase': 'failed', 'error': repr(error), 'test_read': False})
        traceback.print_exc()
        raise
