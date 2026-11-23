"""Final fixed-checkpoint FACED test suite. No optimizer or test selection."""
import argparse
from contextlib import nullcontext
import json
import time
import traceback

import numpy as np
import torch
from peft import set_peft_model_state_dict

from common import ROOT, lock_gpu, panel, prediction_row, save, sha
from parser import parse_structured
from eval_mllm import parse_answer
from exact import metrics
from eval_video import PROMPT as VIDEO_PROMPT


BASE = ROOT / 'final_test_20261120'


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main(args):
    out = BASE / args.model
    out.mkdir(exist_ok=False)
    save(out / 'status.json', dict(phase='loading', training=False, test_read=True))
    torch.set_num_threads(4)
    torch.manual_seed(20261109)
    records = read(BASE / 'test_records.jsonl')
    assert len(records) == 644 and {row['subject'] for row in records} == set(range(100, 123))
    assert all(row['index'] % 3 == 1 for row in records)
    logs = np.load(BASE / 'band_logpower.npy', mmap_mode='r')
    test_indices = [row['index'] for row in records]
    assert np.isfinite(logs[test_indices]).all()
    sources = []
    video_cache = {}
    if args.model == 'affectgpt':
        import train_affectgpt as official
        official.OUT = out
        state = official.setup()
        gpu_lock = state['gpu_lock']
        model, chat = state['model'], state['chat']
        adapter = model.llama_model
        from common import evaluator
        metric = evaluator()
        checkpoint_path = ROOT / 'affective_zeroshot_v1/affectgpt_faced_lora_v1/best.pt'
        video_frames = state['video_namespace']['video_frames']
        visual_cache = {}

        def generate(record, arm):
            video_id = record['video_id']
            if video_id not in visual_cache:
                frames, _ = video_frames(ROOT / f'stimulus_videos_v1/videos/{video_id}.mp4')
                raw_frames = torch.tensor(frames).permute(3, 0, 1, 2).unsqueeze(0).cuda()
                _, visual_cache[video_id] = model.encode_video_merge(raw_frames, raw_frames)
            if arm == 'video':
                prompt = '###Human: <Video><FrameHere></Video>. ' + VIDEO_PROMPT + ' ###Assistant:'
            else:
                values = np.array(logs[record['index']])
                measurements = [values[part].mean(axis=(0, 2)).round(4).tolist()
                                for part in [slice(0, 5), slice(5, 10)]]
                prompt = ('###Human: We uniformly sample raw frames from the video: <Video><FrameHere></Video>. '
                    'The video depicts the stimulus, not the participant. EEG log-band-power channel means '
                    '(delta, theta, alpha, beta, gamma) for successive five-second windows: '
                    + json.dumps(measurements) + '. These measurements do not uniquely establish emotions. '
                    'Infer possible participant emotions, using unrestricted vocabulary. '
                    'Return only JSON {"emotions":["emotion expression", ...]}. ###Assistant:')
            return chat.answer_sample(prompt, dict(frame=visual_cache[video_id], face=None,
                audio=None, multi=None, image=None), do_sample=False, max_new_tokens=320, max_length=2000), None

        zero_context = nullcontext
        tags, arms = ['zeroshot', 'lora'], ['joint', 'video']
        save(out / 'released_assets.json', json.loads((official.BASE / 'assets.json').read_text()))
    else:
        gpu_lock = lock_gpu()
        from joint_runtime import V2VideoRuntime
        from video_train import video_frames
        from transformers.video_utils import VideoMetadata
        if args.model == 'brainrel':
            from train_graph import HypergraphRuntime
            from experts import HypergraphRelationExperts
            language = ROOT / 'v2_video_joint_lora_pilot_v1/best.pt'
            checkpoint_path = ROOT / 'v2_hypergraph_experts_pilot_v1/best.pt'
            runtime = HypergraphRuntime(out, checkpoint_path=language)
            module = HypergraphRelationExperts(runtime.numeric, runtime.model.get_input_embeddings().weight.shape[-1]).cuda()
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            module.load_state_dict(checkpoint['hypergraph'], strict=True)
            runtime.numeric, runtime.filter_class = module.eval().requires_grad_(False), HypergraphRelationExperts
            sources.append(dict(path=str(language), sha256=sha(language)))
            tags = ['adapted']
        else:
            runtime = V2VideoRuntime(out, native_model=args.model)
            checkpoint_path = ROOT / f'native_mllm_adaptation_v1/{args.model}/best.pt'
            tags = ['zeroshot', 'lora']
        runtime.logs = logs
        model, adapter, metric = runtime.model, runtime.model, runtime.metric
        arms = ['eeg', 'joint', 'video']

        def generate(record, arm):
            if arm != 'video':
                batch = runtime.batch(record, joint=arm == 'joint')
            else:
                video_id = record['video_id']
                if video_id not in video_cache:
                    video_cache[video_id] = video_frames(ROOT / f'stimulus_videos_v1/videos/{video_id}.mp4')
                frames, metadata = video_cache[video_id]
                prefix = runtime.processor.apply_chat_template([dict(role='user', content=[
                    dict(type='video'), dict(type='text', text=VIDEO_PROMPT)])], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                batch = runtime.processor(text=[prefix], videos=[frames], return_tensors='pt',
                    videos_kwargs=dict(do_sample_frames=False), video_metadata=[VideoMetadata(
                        total_num_frames=8, fps=8 / metadata['duration_seconds'], duration=metadata['duration_seconds'],
                        frames_indices=list(range(8)))]).to('cuda')
                runtime.context.clear()
                base = model.get_base_model().model
                if hasattr(base, 'rope_deltas'):
                    base.rope_deltas = None
            output = model.generate(**batch, do_sample=False, max_new_tokens=320)
            tail = output[0, batch['input_ids'].shape[1]:]
            return runtime.processor.tokenizer.decode(tail, skip_special_tokens=True).strip(), len(tail) == 320

        zero_context = model.disable_adapter
    sources.append(dict(path=str(checkpoint_path), sha256=sha(checkpoint_path)))
    model.eval().requires_grad_(False)
    adapter.config.use_cache = True
    save(out / 'protocol.json', dict(model=args.model, training=False, test_read=True, n=644,
        subjects=list(range(100, 123)), arms=arms, tags=tags, checkpoints=sources,
        script_sha256=sha(__file__), manifest_sha256=sha(BASE / 'test_records.jsonl'),
        generation=dict(do_sample=False, max_new_tokens=320), metric=metric.protocol,
        selection='Pre-existing main-table checkpoint; no selection or tuning on test',
        prompt='Unchanged main-table EEG/joint prompt; unchanged video-only prompt',
        video_only='Restricted inference with existing joint/EEG adapter, not video-only retraining',
        parsing='Same existing structured parser; invalid output mapped to empty',
        strict_metrics='Lowercase/trim exact sets; eight-class macro F1; both-empty Jaccard=1',
        stimulus_boundary='Known stimulus videos, unseen participants; not unseen-video evaluation'))
    started, results = time.time(), {}
    for tag in tags:
        if tag == 'lora':
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            loaded = set_peft_model_state_dict(adapter, checkpoint['lora'])
            assert not loaded.unexpected_keys
        with torch.inference_mode(), (zero_context() if tag == 'zeroshot' else nullcontext()):
            for arm in arms:
                key = f'{tag}_{arm}'
                predictions, cached = [], {}
                with (out / f'{key}_records.jsonl').open('x') as stream:
                    for number, record in enumerate(records):
                        video_id = record['video_id']
                        if arm == 'video' and video_id in cached:
                            raw, cutoff = cached[video_id]
                        else:
                            raw, cutoff = generate(record, arm)
                            if arm == 'video':
                                cached[video_id] = raw, cutoff
                        parser = parse_answer if args.model == 'affectgpt' and arm == 'joint' else parse_structured
                        terms, valid = parser(raw)
                        row = prediction_row(record, terms, valid, raw=raw, budget_reached=cutoff)
                        predictions.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                        stream.flush()
                        save(out / 'status.json', dict(phase='generating', arm=key, completed=number + 1,
                             total=644, seconds=time.time() - started, training=False, test_read=True))
                pairs = [(set(row['required_concepts']), set(row['emotion_terms']) if row['parse_valid'] else set())
                         for row in predictions]
                results[key] = dict(official_ew=metric.score(predictions), exact_set=metrics(pairs),
                    invalid=sum(not row['parse_valid'] for row in predictions),
                    unique_sets=len({tuple(sorted(row['emotion_terms'])) for row in predictions}))
                save(out / f'{key}_metrics.json', results[key])
    save(out / 'complete.json', dict(results=results, n=644, training=False, test_read=True,
         seconds=time.time() - started))
    save(out / 'status.json', dict(phase='complete', training=False, test_read=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['brainrel', 'oneemo', 'qwen25vl3b', 'affectgpt'], required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        save(BASE / args.model / 'failure.json', dict(error=traceback.format_exc(), training=False, test_read=True))
        raise
