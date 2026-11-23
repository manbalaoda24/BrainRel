"""Read-only integration of native stimulus video into the trained V2 model."""
import json
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import VideoMetadata
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

from common import ROOT, PROJECT, evaluator, lock_gpu, panel, prediction_row, save, sha
from eval_mllm import prompt_for
from parser import parse_structured
from video_train import video_frames


OUT = ROOT / 'v2_stimulus_video_integration_v1'
REGEX = r'model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|out_proj))'


def main():
    gpu_lock = lock_gpu()
    OUT.mkdir(exist_ok=False)
    started = time.time()
    save(OUT / 'status.json', {'phase': 'loading_model', 'test_read': False})
    torch.set_num_threads(4)
    records = [row for row in panel()['val'] if row['subject'] in [80, 81]]
    assert len(records) == 56
    metric = evaluator()
    from run import atlas
    from nodes import NumericEvidenceFilter, measurement_nodes
    model_path = PROJECT / 'oneemo/weights'
    checkpoint_path = ROOT / 'numeric_evidence_filter_full_v1/best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={'': 'cuda'},
        attn_implementation='sdpa', local_files_only=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
                                           target_modules=REGEX, bias='none', task_type='CAUSAL_LM'))
    loaded = set_peft_model_state_dict(model, checkpoint['lora'])
    assert not loaded.unexpected_keys
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in checkpoint['merger'].items():
            assert name in parameters
            parameters[name].copy_(value.to(parameters[name].device))
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    numeric = NumericEvidenceFilter(model.get_input_embeddings().weight.shape[-1]).cuda()
    numeric.load_state_dict(checkpoint['numeric_filter'], strict=True)
    numeric.eval().requires_grad_(False)
    del checkpoint
    context = {}

    def inject(module, arguments, output):
        if not context or output.shape[1] <= int(context['positions'][-1]):
            return output
        positions = context['positions']
        delta, _, _ = numeric(context['nodes'])
        reference = output[:, positions]
        residual = NumericEvidenceFilter.bounded_residual(delta[:, -len(positions):], reference)
        updated = output.clone()
        updated[:, positions] = reference + residual.to(output.dtype)
        return updated

    model.get_input_embeddings().register_forward_hook(inject)
    protocol = {'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha(checkpoint_path),
                'source_sha256': sha(__file__), 'training': False, 'test_read': False,
                'subjects': [80, 81], 'n_per_arm': 56, 'arms': ['v2_eeg', 'v2_joint'],
                'input': 'Unchanged V2 maps and frozen numeric residual; joint additionally receives native video',
                'generation': {'do_sample': False, 'max_new_tokens': 320, 'use_cache': False},
                'video': 'Eight uniformly sampled full-clip RGB frames, no audio; original duration metadata',
                'sampling_source_sha256': sha(PROJECT / 'scripts/mllm/video_train.py'),
                'prompt': 'Exact V2 image-arm prompt; joint appends stimulus-context caveat only',
                'reference': 'Same eight self-report concepts threshold3.5, not independent open annotation',
                'boundary': 'Same known videos across disjoint subjects; video-only cannot personalize; not unseen-stimulus evaluation',
                'timing': 'Whole-video context versus middle10s EEG; joint has larger visual budget',
                'video_id_alignment': 'Number and official duration consistent; clip identity not independently semantically verified',
                'metric': metric.protocol}
    save(OUT / 'protocol.json', protocol)
    limits = np.load(ROOT / 'traceable_visual_alignment_v1/measurements.npz')['limits']
    weights = np.load(ROOT / 'official200_classification_v1/topology_geometry.npz')['weights'].astype('float32')
    logs = np.load(ROOT / 'official200_classification_v1/band_logpower.npy', mmap_mode='r')
    historical = {row['index']: row for row in (json.loads(line) for line in
                  (ROOT / 'numeric_evidence_filter_full_v1/selected_actual.jsonl').read_text().splitlines())}
    cache, scores = {}, {}
    for arm in ['v2_eeg', 'v2_joint']:
        rows = []
        with (OUT / f'{arm}_records.jsonl').open('x') as stream:
            for number, record in enumerate(records):
                values = np.array(logs[record['index']])
                images = [atlas(values[part].mean(0), weights, limits).resize(
                    (640, 192), Image.Resampling.NEAREST) for part in [slice(0, 5), slice(5, 10)]]
                prompt = prompt_for(record)
                content = [{'type': 'image'}, {'type': 'image'}]
                options = {'images': images, 'images_kwargs': {'min_pixels': 320 * 96, 'max_pixels': 832 * 224}}
                if arm == 'v2_joint':
                    video_id = record['video_id']
                    if video_id not in cache:
                        frames, metadata = video_frames(ROOT / f'stimulus_videos_v1/videos/{video_id}.mp4')
                        cache[video_id] = frames, metadata
                        save(OUT / f'video_{video_id}_sampling.json', metadata)
                    frames, metadata = cache[video_id]
                    content.append({'type': 'video'})
                    options.update(videos=[frames], videos_kwargs={'do_sample_frames': False},
                                   video_metadata=[VideoMetadata(total_num_frames=8,
                                       fps=8 / metadata['duration_seconds'], duration=metadata['duration_seconds'],
                                       frames_indices=list(range(8)))])
                    prompt += ('The video depicts the stimulus, not the participant. Use it as fallible context '
                               'without equating the depicted emotion with this participant\'s experience. ')
                content.append({'type': 'text', 'text': prompt})
                prefix = processor.apply_chat_template([{'role': 'user', 'content': content}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                batch = processor(text=[prefix], return_tensors='pt', **options).to('cuda')
                count = batch['input_ids'].shape[1]
                positions = torch.arange(count - 32, count, device='cuda')
                for name in ['image_token_id', 'video_token_id']:
                    token = getattr(model.config, name, None)
                    assert token is None or not (batch['input_ids'][:, positions] == token).any()
                context.update(nodes=torch.tensor(measurement_nodes(values, weights, limits), device='cuda').unsqueeze(0),
                               positions=positions)
                base = model.get_base_model().model
                if hasattr(base, 'rope_deltas'):
                    base.rope_deltas = None
                with torch.inference_mode():
                    generated = model.generate(**batch, do_sample=False, max_new_tokens=320)
                tail = generated[0, count:]
                raw = processor.tokenizer.decode(tail, skip_special_tokens=True).strip()
                terms, valid = parse_structured(raw)
                row = prediction_row(record, terms, valid, raw=raw, generated_tokens=len(tail),
                    input_tokens=count, budget_reached=len(tail) == 320,
                    historical_raw_equal=raw == historical[record['index']]['raw'] if arm == 'v2_eeg' else None,
                    image_grid=batch['image_grid_thw'].tolist(),
                    video_grid=batch['video_grid_thw'].tolist() if 'video_grid_thw' in batch else [],
                    prompt=prompt)
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                save(OUT / 'status.json', {'phase': 'generating', 'arm': arm, 'completed': number + 1,
                     'total': 56, 'seconds': time.time() - started, 'test_read': False})
                if arm == 'v2_eeg' and number == 3 and not all(item['historical_raw_equal'] for item in rows):
                    raise RuntimeError('V2 replay differs on initial four records; inspect before adding video')
        scores[arm] = metric.score(rows)
        save(OUT / f'{arm}_metrics.json', scores[arm])
    save(OUT / 'complete.json', {'scores': scores, 'seconds': time.time() - started, 'test_read': False})
    save(OUT / 'status.json', {'phase': 'complete', 'test_read': False})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists() and not (OUT / 'complete.json').exists():
            save(OUT / 'status.json', {'phase': 'failed', 'error': repr(error), 'test_read': False})
        traceback.print_exc()
        raise
