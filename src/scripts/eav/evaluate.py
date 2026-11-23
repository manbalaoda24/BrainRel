"""Frozen-checkpoint evaluation on all official validation inputs, without fitting."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
from generate import parse_answer, prompt_for, sha, write_json


def main(args):
    root = Path('/data/model/datasets/EAV_13799131')
    lock = (root / f'eav_gpu{args.gpu}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert int(subprocess.check_output(['nvidia-smi', f'--id={args.gpu}', '--query-gpu=memory.used',
        '--format=csv,noheader,nounits'], text=True).strip()) < 1000
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from runtime import EAVRuntime
    from wheel import OfficialEW
    torch.set_num_threads(2)
    run_root = args.run_root or root / 'heldout700_v1'
    frozen = json.loads((run_root / 'frozen.json').read_text())['models'][args.name]
    language, graph = Path(frozen['language']), Path(frozen['graph'])
    assert sha(language) == frozen['language_sha256'] and sha(graph) == frozen['graph_sha256']
    out = run_root / args.name
    out.mkdir(exist_ok=False)
    status = out / 'status.json'
    write_json(status, dict(phase='loading_model', test_read=False))
    model_path = frozen.get('model', '/home/ubuntu/work/eeg_project/oneemo/weights')
    if frozen.get('native_visual', False):
        from visual_runtime import VisualBaselineRuntime
        from peft import set_peft_model_state_dict, get_peft_model_state_dict
        runtime = VisualBaselineRuntime(root / 'brainrel_adapter_v1', model_path, language, graph)
        state = torch.load(language, map_location='cpu', weights_only=False)
        restored = set_peft_model_state_dict(runtime.model, state['lora'])
        assert not restored.unexpected_keys
        actual = get_peft_model_state_dict(runtime.model)
        assert set(actual) == set(state['lora'])
        assert all(torch.equal(actual[key].detach().cpu(), value.to(actual[key].dtype)) for key, value in state['lora'].items())
    else:
        runtime = EAVRuntime(root / 'brainrel_adapter_v1', model_path, language, graph)
    rows = [json.loads(line) for line in (root / 'brainrel_adapter_v1/val_records.jsonl').read_text().splitlines()]
    assert len(rows) == 700 and {row['subject'] for row in rows} == set(range(29, 36))
    reference_path = root / 'heldout700_reference_v1/trial_reference.jsonl'
    reference = {row['sample_id']: row['target']['emotions'] for row in
                 [json.loads(line) for line in reference_path.read_text().splitlines()]}
    assert reference and set(reference) <= {row['sample_id'] for row in rows}
    write_json(out / 'protocol.json', dict(frozen, reference_sha256=sha(reference_path),
        selection=frozen.get('selection', 'Frozen before annotation; no checkpoint/prompt tuning on this panel'),
        inputs=700, supported_references=len(reference), test_read=False,
        unknown_policy='Generate all700; unsupported references excluded, never interpreted as empty emotion truth',
        limitation='Teacher-derived ASR expressed-affect reference, not independent human gold; some official category validation was used historically'))
    predictions = []
    with torch.inference_mode(), (out / 'records.jsonl').open('x') as stream:
        for number, row in enumerate(rows):
            batch = runtime.batch(row, arm=frozen['arm'], prompt=frozen['prompt'])
            if frozen['disable_numeric']:
                runtime.context.clear()
            generated = runtime.model.generate(**batch, do_sample=False, max_new_tokens=160)
            raw = runtime.processor.tokenizer.decode(generated[0, batch['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            terms, valid = parse_answer(raw)
            record = dict(index=row['index'], sample_id=row['sample_id'], subject=row['subject'],
                emotion_terms=terms, parse_valid=valid, raw=raw, reference_supported=row['sample_id'] in reference)
            if record['reference_supported']:
                record['required_concepts'] = reference[row['sample_id']]
            predictions.append(record)
            stream.write(json.dumps(record) + '\n'); stream.flush()
            write_json(status, dict(phase='generating', completed=number + 1, total=700, test_read=False))
    scored = [row for row in predictions if row['reference_supported']]
    metrics = OfficialEW(Path('/data/model/datasets/FACED_syn50614194')).score(scored)
    write_json(out / 'complete.json', dict(phase='complete', inputs=700, scored=len(scored),
        unknown_reference=700-len(scored), metrics=metrics, test_read=False))
    write_json(status, dict(phase='complete', test_read=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--run-root', type=Path)
    main(parser.parse_args())
