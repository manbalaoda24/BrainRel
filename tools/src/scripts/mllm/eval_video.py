"""Video-restricted inference with released and previously selected FACED adapters."""
import argparse
from contextlib import nullcontext
import json
import time
import traceback

import torch
from peft import set_peft_model_state_dict

from common import ROOT, lock_gpu, panel, prediction_row, save, sha
from parser import parse_structured


PROMPT = (
    'Infer the emotional states the participant may experience from the supplied stimulus video. '
    'Return only JSON {"emotions": ["emotion expression", ...]}. '
    'Use unrestricted emotion vocabulary and any number of emotions; an empty list is allowed. '
    'The video depicts the stimulus, not the participant. Treat it as fallible context, '
    'not direct evidence of this participant\'s internal state. '
    'No participant EEG or classifier prediction is supplied. Do not invent absent evidence.'
)


def main(args):
    out = ROOT / args.output
    out.mkdir(exist_ok=False)
    save(out / 'status.json', dict(phase='loading', training=False, test_read=False))
    torch.set_num_threads(4)
    torch.manual_seed(20261119)
    records = panel()['val']
    assert len(records) == 560 and len({row['video_id'] for row in records}) == 28
    frames_cache = {}

    def frames_for(record):
        video_id = record['video_id']
        if video_id not in frames_cache:
            frames_cache[video_id] = video_frames(ROOT / f'stimulus_videos_v1/videos/{video_id}.mp4')
            save(out / f'video_{video_id}_sampling.json', frames_cache[video_id][1])
        return frames_cache[video_id]

    if args.model == 'affectgpt':
        import train_affectgpt as official
        official.OUT = out
        state = official.setup()
        model, chat = state['model'], state['chat']
        video_frames = state['video_namespace']['video_frames']
        adapter = model.llama_model
        from common import evaluator
        metric = evaluator()
        checkpoint_path = ROOT / 'affective_zeroshot_v1/affectgpt_faced_lora_v1/best.pt'
        actual_prompt = '###Human: <Video><FrameHere></Video>. ' + PROMPT + ' ###Assistant:'
        visual_cache = {}

        def generate(record):
            video_id = record['video_id']
            if video_id not in visual_cache:
                frames, _ = frames_for(record)
                raw = torch.tensor(frames).permute(3, 0, 1, 2).unsqueeze(0).cuda()
                _, visual_cache[video_id] = model.encode_video_merge(raw, raw)
            raw = chat.answer_sample(actual_prompt,
                dict(frame=visual_cache[video_id], face=None, audio=None, multi=None, image=None),
                do_sample=False, max_new_tokens=320, max_length=2000)
            return raw, None

        zero_context = nullcontext
    else:
        gpu_lock = lock_gpu()
        from video_train import video_frames
        from joint_runtime import V2VideoRuntime
        from transformers.video_utils import VideoMetadata
        if args.model == 'brainrel':
            language_source = ROOT / 'v2_video_joint_lora_pilot_v1/best.pt'
            assert sha(language_source) == 'c45b420febbf787e938fdb12b31ba06ebe723080b4c144730fe99500dcd469da'
            runtime = V2VideoRuntime(out, checkpoint_path=language_source)
        else:
            runtime = V2VideoRuntime(out, native_model=args.model)
        model, adapter, metric = runtime.model, runtime.model, runtime.metric
        checkpoint_path = (language_source if args.model == 'brainrel' else
                           ROOT / f'native_mllm_adaptation_v1/{args.model}/best.pt')
        actual_prompt = PROMPT

        def generate(record):
            frames, metadata = frames_for(record)
            content = [dict(type='video'), dict(type='text', text=actual_prompt)]
            prefix = runtime.processor.apply_chat_template([dict(role='user', content=content)],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            batch = runtime.processor(text=[prefix], videos=[frames], return_tensors='pt',
                videos_kwargs=dict(do_sample_frames=False),
                video_metadata=[VideoMetadata(total_num_frames=8,
                    fps=8 / metadata['duration_seconds'], duration=metadata['duration_seconds'],
                    frames_indices=list(range(8)))]).to('cuda')
            assert 'pixel_values' not in batch and 'image_grid_thw' not in batch
            runtime.context.clear()
            base = model.get_base_model().model
            if hasattr(base, 'rope_deltas'):
                base.rope_deltas = None
            generated = model.generate(**batch, do_sample=False, max_new_tokens=320)
            tail = generated[0, batch['input_ids'].shape[1]:]
            return runtime.processor.tokenizer.decode(tail, skip_special_tokens=True).strip(), len(tail) == 320

        zero_context = model.disable_adapter

    model.eval().requires_grad_(False)
    adapter.config.use_cache = True
    save(out / 'protocol.json', dict(model=args.model, training=False, test_read=False,
        prompt=actual_prompt, validation_inputs=560, distinct_stimuli=28,
        checkpoint=str(checkpoint_path), checkpoint_sha256=sha(checkpoint_path),
        script_sha256=sha(__file__), metric=metric.protocol,
        generation=dict(do_sample=False, max_new_tokens=320),
        input='Eight stimulus frames only; no EEG atlas, measurements, numeric residual, transcript or prior',
        lora_boundary='Existing selected EEG/joint FACED adapter restricted to video inputs; not video-only training',
        zero_boundary='Released model; official pretrained AffectGPT adapters retained',
        caching='Identical video and prompt decoded once per stimulus per arm; expanded to all 20 subject references',
        reference_panel_sha256=sha(ROOT / 'numeric_evidence_filter_full_v1/val_open_records.jsonl')))
    started = time.time()
    scores = {}
    for arm in (['adapted'] if args.model == 'brainrel' else ['zeroshot', 'lora']):
        if arm == 'lora':
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            assert 'lora' in checkpoint
            loaded = set_peft_model_state_dict(adapter, checkpoint['lora'])
            assert not loaded.unexpected_keys
            save(out / 'adapter_load.json', dict(epoch=checkpoint['epoch'],
                adapter_tensors=len(checkpoint['lora']), unexpected_keys=loaded.unexpected_keys))
        predictions = []
        generated_cache = {}
        with torch.inference_mode(), (zero_context() if arm == 'zeroshot' else nullcontext()), \
                (out / f'{arm}_records.jsonl').open('x') as stream:
            for number, record in enumerate(records):
                video_id = record['video_id']
                if video_id not in generated_cache:
                    generated_cache[video_id] = generate(record)
                raw, budget_reached = generated_cache[video_id]
                terms, valid = parse_structured(raw)
                row = prediction_row(record, terms, valid, raw=raw,
                    budget_reached=budget_reached, cached_by_stimulus=True)
                predictions.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                save(out / 'status.json', dict(phase='generating', arm=arm, completed=number + 1,
                    total=560, decoded_stimuli=len(generated_cache), training=False, test_read=False))
        scores[arm] = metric.score(predictions)
        save(out / f'{arm}_metrics.json', scores[arm])
    save(out / 'complete.json', dict(scores=scores, seconds=time.time() - started,
        training=False, test_read=False))
    save(out / 'status.json', dict(phase='complete', training=False, test_read=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen25vl3b', 'oneemo', 'affectgpt', 'brainrel'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        if (ROOT / args.output).exists():
            save(ROOT / args.output / 'failure.json', dict(error=traceback.format_exc(), training=False, test_read=False))
        raise
