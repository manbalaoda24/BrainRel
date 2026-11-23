"""One-factor, frozen-checkpoint inference sensitivity of the RMS residual cap."""
import argparse
import json
import time
import traceback

import torch

from common import ROOT, lock_gpu, panel, prediction_row, save, sha
from parser import parse_structured
from sparse import BudgetedRelationFilter
from train_graph import HypergraphRuntime
from eval_modalities import batch_for, PROMPT


CONDITIONS = {'off': 0., 'cap05': .05, 'cap10': .1, 'cap20': .2,
              'cap40': .4, 'cap80': .8, 'uncapped': None}


class CapController:
    def __init__(self):
        self.ratio = .2
        self.diagnostics = None

    def bounded_residual(self, delta, reference):
        reference_rms = reference.float().square().mean(-1, keepdim=True).sqrt().detach()
        delta_rms = delta.float().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        if self.ratio is None:
            residual = delta
            clipped = torch.zeros_like(delta_rms, dtype=torch.bool)
        else:
            cap = self.ratio * reference_rms
            residual = delta * (cap / delta_rms).clamp(max=1.)
            clipped = delta_rms > cap
        self.diagnostics = {
            'clipped_position_fraction': float(clipped.float().mean()),
            'raw_residual_rms': float(delta.float().square().mean().sqrt()),
            'bounded_residual_rms': float(residual.float().square().mean().sqrt()),
            'reference_rms': float(reference.float().square().mean().sqrt()),
        }
        if self.ratio == .2:
            torch.testing.assert_close(residual,
                BudgetedRelationFilter.bounded_residual(delta, reference), rtol=0, atol=0)
        return residual


def main(args):
    gpu_lock = lock_gpu()
    torch.set_num_threads(2)
    torch.manual_seed(20261117)
    output = ROOT / args.output
    output.mkdir(exist_ok=False)
    save(output / 'status.json', {'phase': 'loading', 'test_read': False})
    started = time.time()
    try:
        source = ROOT / 'relation_repair_sparse_v1/best.pt'
        language = ROOT / 'v2_video_joint_lora_pilot_v1/best.pt'
        runtime = HypergraphRuntime(output, checkpoint_path=language)
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        module = BudgetedRelationFilter(runtime.numeric,
            runtime.model.get_input_embeddings().weight.shape[-1], sparse=True).cuda()
        module.load_state_dict(checkpoint['hypergraph'], strict=True)
        module.epoch = int(checkpoint['epoch'])
        module.eval().requires_grad_(False)
        runtime.numeric = module
        controller = CapController()
        runtime.filter_class = controller
        runtime.model.eval().requires_grad_(False)
        records = panel()['val']
        assert len(records) == 560 and tuple(module.budgets()) == (5, 9, 2)
        save(output / 'protocol.json', {
            'experiment': 'Inference RMS-cap sensitivity; not independently retrained hyperparameters',
            'checkpoint': str(source), 'checkpoint_sha256': sha(source),
            'language_checkpoint': str(language), 'language_sha256': sha(language),
            'prompt': PROMPT, 'conditions': {name: CONDITIONS[name] for name in args.conditions},
            'arms': ['joint'], 'validation_trials': 560, 'training_updates': 0,
            'test_read': False, 'generation': {'do_sample': False, 'max_new_tokens': 320},
            'formula': 'delta * min(1, ratio * RMS(reference) / max(RMS(delta),1e-6)); per position',
            'unchanged': 'weights, node bypass, sparse budgets, input order, prompt, metric',
            'metric': runtime.metric.protocol, 'script_sha256': sha(__file__),
            'selection': 'Report all conditions; do not replace the main checkpoint or prompt',
        })
        rows = {name: [] for name in args.conditions}
        streams = {name: (output / f'{name}_records.jsonl').open('x') for name in rows}
        try:
            with torch.inference_mode():
                for number, record in enumerate(records):
                    for name in args.conditions:
                        controller.ratio = CONDITIONS[name]
                        batch = batch_for(runtime, record, '111')
                        generated = runtime.model.generate(**batch, do_sample=False, max_new_tokens=320)
                        tail = generated[0, batch['input_ids'].shape[1]:]
                        raw = runtime.processor.tokenizer.decode(tail, skip_special_tokens=True).strip()
                        terms, valid = parse_structured(raw)
                        row = prediction_row(record, terms, valid, raw=raw,
                            budget_reached=len(tail) == 320, **controller.diagnostics)
                        rows[name].append(row)
                        streams[name].write(json.dumps(row, ensure_ascii=False) + '\n')
                        streams[name].flush()
                    save(output / 'status.json', {'phase': 'evaluating', 'completed': number + 1,
                        'total': 560, 'conditions': args.conditions,
                        'seconds': time.time() - started, 'test_read': False})
            summary = {name: {'metric': runtime.metric.score(values),
                'mean_clipped_position_fraction': sum(row['clipped_position_fraction'] for row in values) / len(values),
                'parse_valid_fraction': sum(row['parse_valid'] for row in values) / len(values)}
                for name, values in rows.items()}
            save(output / 'summary.json', summary)
            save(output / 'complete.json', {'phase': 'complete', 'summary': summary,
                'seconds': time.time() - started, 'test_read': False})
            save(output / 'status.json', {'phase': 'complete', 'test_read': False})
        finally:
            for stream in streams.values():
                stream.close()
    except Exception:
        save(output / 'status.json', {'phase': 'failed', 'traceback': traceback.format_exc(),
            'test_read': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conditions', nargs='+', choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument('--output', default='rms_sensitivity_frozen_v1')
    main(parser.parse_args())
