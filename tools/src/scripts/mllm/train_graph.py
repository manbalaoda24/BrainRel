"""Frozen selected joint model with additive or integrated relation-filter pilots."""
import json
import math
import os
import time
import traceback

import numpy as np
import torch

from common import ROOT, lock_gpu, panel, prediction_row, save, sha, write_rows
from parser import parse_structured
from joint_runtime import V2VideoRuntime


INTEGRATED = os.environ.get('FACED_INTEGRATED_HYPERGRAPH') == '1'
OUT = ROOT / ('v2_hypergraph_experts_pilot_v1' if INTEGRATED else 'v2_hypergraph_relation_pilot_v1')


class HypergraphRuntime(V2VideoRuntime):
    def inject(self, module, arguments, output):
        positions = self.context.get('positions')
        if positions is None or output.shape[1] <= int(positions[-1]):
            return output
        delta, gates, attention = self.numeric(self.context['nodes'])
        reference = output[:, positions]
        residual = self.filter_class.bounded_residual(delta[:, -len(positions):], reference)
        updated = output.clone()
        updated[:, positions] = reference + residual.to(output.dtype)
        if not torch.is_grad_enabled():
            self.last_trace = (self.numeric.relation_trace() if INTEGRATED else
                               {'edge_gates': gates.detach().cpu().tolist(),
                                'slot_to_edge_attention': attention.detach().cpu().tolist()})
        return updated


def main():
    gpu_lock = lock_gpu()
    OUT.mkdir(exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(20261109)
    datasets = panel()
    train = [json.loads(line) for line in
             (ROOT / 'v2_video_joint_lora_pilot_v1/train_records.jsonl').read_text().splitlines()]
    assert len(train) == 240 and all(row in datasets['train'] for row in train)
    val = [row for row in datasets['val'] if row['subject'] in [80, 81]]
    assert len(val) == 56
    write_rows(OUT / 'train_records.jsonl', train)
    write_rows(OUT / 'selection_records.jsonl', val)
    save(OUT / 'status.json', {'phase': 'loading_model', 'test_read': False})
    source = ROOT / 'v2_video_joint_lora_pilot_v1/best.pt'
    runtime = HypergraphRuntime(OUT, checkpoint_path=source)
    from hypergraph import HypergraphEvidenceFilter
    base = runtime.numeric
    if INTEGRATED:
        from experts import HypergraphRelationExperts
        filter_class = HypergraphRelationExperts
    else:
        filter_class = HypergraphEvidenceFilter
    runtime.numeric = filter_class(base, runtime.model.get_input_embeddings().weight.shape[-1]).cuda()
    runtime.filter_class = filter_class
    runtime.model.eval().requires_grad_(False)
    parameters = [value for value in runtime.numeric.parameters() if value.requires_grad]
    assert parameters and not any(value.requires_grad for value in base.parameters())
    nodes = torch.tensor(runtime.measurement_nodes(np.array(runtime.logs[train[0]['index']]),
                         runtime.weights, runtime.limits), device='cuda').unsqueeze(0)
    with torch.no_grad():
        actual = runtime.numeric(nodes)[0]
        expected = base(nodes)[0]
        initial_difference = float((actual - expected).abs().max())
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5), 'Initial filter output mismatch'
    protocol = {'initial_checkpoint': str(source), 'initial_sha256': sha(source),
                'training_trials': 240, 'selection_trials': 56, 'epochs': 2, 'updates': 120,
                'effective_batch': 8, 'learning_rate': 1e-4, 'warmup': 6,
                'trainable': ('Integrated member scores, edge gates, existing experts and router; encoder, slot readout, '
                              'projector, OneEmo, LoRA and merger frozen' if INTEGRATED else
                              'New additive hypergraph module only; base filter, OneEmo, LoRA, visual merger frozen'),
                'selection': 'Joint EW-F1 then EEG-only EW-F1, epoch0 eligible',
                'final_evaluation': 'Selected checkpoint, all560 validation trials; no test',
                'scope': 'Continuation architecture pilot, not equal-total-budget comparison or causal attribution',
                'source_sha256': sha(__file__),
                'hypergraph_sha256': sha(__import__(filter_class.__module__).__file__),
                'integrated_experts': INTEGRATED,
                'node_attributes': 'Existing clipped displayed-map statistics; not raw EEG amplitudes',
                'question_conditioned': False, 'test_read': False}
    save(OUT / 'protocol.json', protocol)
    save(OUT / 'hypergraph_schema.json', runtime.numeric.describe())
    save(OUT / 'status.json', {'phase': 'gradient_preflight', 'test_read': False})
    batch = runtime.batch(train[0], joint=True, answer=True)
    preflight_loss = runtime.model(**batch, use_cache=False).loss
    assert torch.isfinite(preflight_loss)
    preflight_loss.backward()
    gradient = math.sqrt(sum(float(value.grad.float().square().sum()) for value in parameters
                             if value.grad is not None))
    assert gradient > 0 and math.isfinite(gradient)
    assert all(value.grad is None for value in runtime.model.parameters())
    assert all(value.grad is None for value in base.parameters())
    assert all(value.grad is None for value in runtime.numeric.parameters() if not value.requires_grad)
    save(OUT / 'preflight.json', {'initialization_within_tolerance': True, 'initial_max_abs_difference': initial_difference,
         'loss': float(preflight_loss.detach()), 'trainable_gradient_norm': gradient,
         'frozen_parameter_gradients': False, 'optimizer_steps': 0})
    runtime.numeric.zero_grad(set_to_none=True)
    del batch, preflight_loss

    def evaluate(epoch, records, prefix):
        scores = {}
        runtime.numeric.eval()
        with torch.no_grad():
            for arm in ['eeg', 'joint']:
                rows = []
                traces = []
                for number, record in enumerate(records):
                    batch = runtime.batch(record, joint=arm == 'joint')
                    generated = runtime.model.generate(**batch, do_sample=False, max_new_tokens=320)
                    tail = generated[0, batch['input_ids'].shape[1]:]
                    raw = runtime.processor.tokenizer.decode(tail, skip_special_tokens=True).strip()
                    terms, valid = parse_structured(raw)
                    rows.append(prediction_row(record, terms, valid, raw=raw, budget_reached=len(tail) == 320))
                    if number < 10:
                        traces.append({'index': record['index'], 'arm': arm, **runtime.last_trace})
                    save(OUT / 'status.json', {'phase': 'evaluating', 'panel': prefix, 'epoch': epoch,
                         'arm': arm, 'completed': number + 1, 'total': len(records), 'test_read': False})
                write_rows(OUT / f'{prefix}_{arm}_records.jsonl', rows)
                write_rows(OUT / f'{prefix}_{arm}_traces.jsonl', traces)
                scores[arm] = runtime.metric.score(rows)
        save(OUT / f'{prefix}_metrics.json', scores)
        return scores

    def checkpoint(name, epoch):
        temporary = OUT / f'{name}.tmp'
        torch.save({'hypergraph': runtime.numeric.state_dict(), 'epoch': epoch, 'protocol': protocol}, temporary)
        temporary.replace(OUT / f'{name}.pt')

    initial = evaluate(0, val, 'epoch0')
    best_key = (initial['joint']['f1'], initial['eeg']['f1'])
    best_epoch = 0
    checkpoint('best', 0)
    history = [{'epoch': 0, 'scores': initial}]
    save(OUT / 'history.json', history)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
    rng = np.random.default_rng(20261109)
    update = 0
    for epoch in range(1, 3):
        runtime.numeric.train()
        base.eval()
        order = rng.permutation(len(train)).tolist()
        losses = []
        for offset in range(0, len(order), 4):
            factor = (update + 1) / 6 if update < 6 else .5 * (1 + math.cos(math.pi * (update - 6) / 113))
            optimizer.param_groups[0]['lr'] = 1e-4 * factor
            optimizer.zero_grad(set_to_none=True)
            presentations = [(train[index], joint) for index in order[offset:offset + 4] for joint in [False, True]]
            rng.shuffle(presentations)
            for record, joint in presentations:
                batch = runtime.batch(record, joint=joint, answer=True)
                loss = runtime.model(**batch, use_cache=False).loss
                assert torch.isfinite(loss)
                losses.append(float(loss.detach()))
                (loss / 8).backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            if update == 0:
                assert float(gradient) > 0
            assert all(value.grad is None for value in runtime.model.parameters())
            assert all(value.grad is None for value in base.parameters())
            optimizer.step()
            update += 1
            save(OUT / 'status.json', {'phase': 'training', 'epoch': epoch, 'update': update,
                 'total_updates': 120, 'ce': float(np.mean(losses)), 'gradient_norm': float(gradient),
                 'learning_rate': optimizer.param_groups[0]['lr'], 'test_read': False})
            if update % 10 == 0:
                checkpoint('last', epoch)
                print('UPDATE', update, 'CE', float(np.mean(losses)), flush=True)
        scores = evaluate(epoch, val, f'epoch{epoch}')
        history.append({'epoch': epoch, 'updates': update, 'scores': scores, 'ce': float(np.mean(losses))})
        save(OUT / 'history.json', history)
        key = (scores['joint']['f1'], scores['eeg']['f1'])
        if key > best_key:
            best_key, best_epoch = key, epoch
            checkpoint('best', epoch)
        checkpoint('last', epoch)
    runtime.numeric.load_state_dict(torch.load(OUT / 'best.pt', map_location='cpu', weights_only=False)['hypergraph'])
    final = evaluate(best_epoch, datasets['val'], 'selected_full560')
    save(OUT / 'complete.json', {'phase': 'complete', 'selected_epoch': best_epoch, 'updates': update,
         'full560': final, 'seconds': time.time() - started, 'test_read': False})
    save(OUT / 'status.json', {'phase': 'complete', 'test_read': False})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            save(OUT / 'status.json', {'phase': 'failed', 'error': repr(error), 'test_read': False})
        traceback.print_exc()
        raise
