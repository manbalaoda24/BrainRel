"""Frozen active-message interventions with matched random controls."""
import argparse
from collections import Counter
import json
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as functional

from common import ROOT, lock_gpu, panel, prediction_row, save, sha
from parser import parse_structured
from sparse import BudgetedRelationFilter
from train_graph import HypergraphRuntime
from eval_modalities import batch_for, PROMPT


QUEUES = {0: ['full', 'mask_selected', 'random_0', 'random_1'],
          1: ['random_2', 'random_3', 'random_4']}


class MaskedRelations(BudgetedRelationFilter):
    def forward(self, nodes):
        encoded = self.encode(nodes.float())
        weights = self.router(encoded).softmax(-1)
        expert_outputs, members_all, gates_all, selections = [], [], [], []
        pooled_all, messages_all, responses = [], [], []
        for kind, budget in enumerate(self.budgets()):
            selected = self.edge_types == kind
            incidence = self.incidence[selected]
            logits = self.member_scores[kind](encoded).squeeze(-1).unsqueeze(1)
            members = logits.expand(-1, int(selected.sum()), -1)
            members = members.masked_fill(~incidence.unsqueeze(0), -torch.inf).softmax(-1)
            edges = members @ encoded
            raw = self.edge_gates[kind](edges)
            gate = 2 * raw.sigmoid()
            mask = torch.ones_like(raw.squeeze(-1))
            if budget < mask.shape[1]:
                order = torch.argsort(raw.squeeze(-1), dim=1, descending=True, stable=True)
                mask = torch.zeros_like(mask).scatter_(1, order[:, :budget], 1.)
            messages = edges * gate * mask.unsqueeze(-1)
            pooled_all.append(edges.detach())
            messages_all.append(messages.detach())
            edge_ids = selected.nonzero().flatten()
            dropped = getattr(self, 'drop_ids', [])
            keep = ~torch.isin(edge_ids, torch.tensor(dropped, device=edge_ids.device, dtype=torch.long))
            contexts = self.recipients[selected].T @ (messages * keep[None, :, None])
            transformed = self.experts[kind](contexts)
            expert_outputs.append(transformed)
            responses.append(transformed.detach())
            members_all.append(members)
            gates_all.append(gate.squeeze(-1))
            selections.append(mask.detach())
        experts = torch.stack(expert_outputs, dim=-2)
        filtered = encoded + (weights.unsqueeze(-1) * experts).sum(-2)
        slots, attention = self.attention(self.queries.expand(nodes.shape[0], -1, -1),
            filtered, filtered, need_weights=True)
        self.trace = {'members': torch.cat(members_all, 1).detach(),
            'gates': torch.cat(gates_all, 1).detach(), 'node_router': weights.detach(),
            'slot_to_node': attention.detach()}
        self.retrain_trace = {'selection': torch.cat(selections, 1), 'budgets': self.budgets()}
        self.content_trace = {'pooled': torch.cat(pooled_all, 1),
            'messages': torch.cat(messages_all, 1), 'expert_response': torch.stack(responses, 1)}
        return self.project(slots) * self.gate_logit.sigmoid(), weights, attention


def mask_sets(module, index):
    trace = module.relation_trace()
    mass = np.asarray(trace['mean_selected_edge_routing_mass'])[0]
    active = np.asarray(trace['selection'])[0].astype(bool)
    kinds = module.edge_types.cpu().numpy()
    member_counts = module.incidence.sum(1).cpu().numpy()
    recipient_counts = module.recipients.sum(1).cpu().numpy()
    core = []
    for kind, count in enumerate([2, 4, 1]):
        available = np.flatnonzero(active & (kinds == kind))
        core.extend(sorted(available.tolist(), key=lambda edge: (-mass[edge], edge))[:count])
    strata = Counter((int(kinds[edge]), int(member_counts[edge]), int(recipient_counts[edge]))
        for edge in core)
    result = {'full': [], 'mask_selected': core}
    for seed in range(5):
        generator = np.random.default_rng(np.random.SeedSequence([20261114, index, seed]))
        chosen = []
        for (kind, members, recipients), count in sorted(strata.items()):
            candidates = np.flatnonzero(active & (kinds == kind)
                & (member_counts == members) & (recipient_counts == recipients))
            assert len(candidates) >= count
            chosen.extend(generator.choice(candidates, count, replace=False).tolist())
        result[f'random_{seed}'] = sorted(chosen)
    assert all(active[edge] for edges in result.values() for edge in edges)
    return result, trace


def main(args):
    gpu_lock = lock_gpu()
    torch.set_num_threads(2)
    torch.manual_seed(20261114)
    out = ROOT / f'selected_relation_masks_v1_gpu{args.queue}'
    out.mkdir(exist_ok=False)
    save(out / 'status.json', {'phase': 'loading', 'test_read': False})
    source = ROOT / 'relation_repair_sparse_v1/best.pt'
    language = ROOT / 'v2_video_joint_lora_pilot_v1/best.pt'
    runtime = HypergraphRuntime(out, checkpoint_path=language)
    module = MaskedRelations(runtime.numeric,
        runtime.model.get_input_embeddings().weight.shape[-1], sparse=True).cuda()
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    module.load_state_dict(checkpoint['hypergraph'], strict=True)
    module.epoch = int(checkpoint['epoch'])
    module.eval().requires_grad_(False)
    runtime.model.eval().requires_grad_(False)
    records = panel()['val']
    assert len(records) == 560 and tuple(module.budgets()) == (5, 9, 2)
    save(out / 'protocol.json', {'checkpoint': str(source), 'sha256': sha(source),
        'language_checkpoint': str(language), 'language_sha256': sha(language),
        'prompt': PROMPT, 'conditions': QUEUES[args.queue], 'training_updates': 0,
        'test_read': False, 'samples': 560, 'selected_counts': [2, 4, 1],
        'ranking': 'Original unsigned structural routing mass among active edges',
        'random_seeds': [0, 1, 2, 3, 4], 'random_matching': 'Active edges; type/member count/recipient count',
        'mask': 'Zero gated selected messages before recipient aggregation; no refill or renormalization',
        'unchanged': 'Node bypass, node-router, original sparse membership, visual/text inputs',
        'downstream': 'Slot attention and RMS cap recompute normally',
        'reference': runtime.metric.protocol, 'script_sha256': sha(__file__),
        'generation': {'do_sample': False, 'max_new_tokens': 320}})
    baseline = BudgetedRelationFilter(runtime.numeric,
        runtime.model.get_input_embeddings().weight.shape[-1], sparse=True).cuda()
    baseline.load_state_dict(checkpoint['hypergraph'], strict=True)
    baseline.epoch = module.epoch
    baseline.eval().requires_grad_(False)
    runtime.numeric = module
    started = time.time()
    rows = {condition: [] for condition in QUEUES[args.queue]}
    streams = {condition: (out / f'{condition}_records.jsonl').open('x') for condition in rows}
    traces = (out / 'baseline_traces.jsonl').open('x') if args.queue == 0 else None
    pooled, messages, responses = [], [], []
    try:
        with torch.inference_mode():
            for number, record in enumerate(records):
                module.drop_ids = []
                batch = batch_for(runtime, record, '111')
                nodes = runtime.context['nodes']
                original = module(nodes)[0].clone()
                if number == 0:
                    expected = baseline(nodes)[0]
                    torch.testing.assert_close(original, expected, rtol=1e-5, atol=1e-6)
                    save(out / 'preflight.json', {'unmasked_matches_original': True,
                        'max_abs_difference': float((original - expected).abs().max())})
                choices, trace = mask_sets(module, record['index'])
                original_messages = module.content_trace['messages'].clone()
                original_selection = module.retrain_trace['selection'].clone()
                positions = runtime.context['positions']
                reference = functional.embedding(batch['input_ids'][:, positions],
                    runtime.model.get_input_embeddings().weight)
                original_bounded = module.bounded_residual(original, reference).clone()
                if traces is not None:
                    trace_row = {'index': record['index'], 'subject': record['subject'],
                        'video_id': record['video_id'], 'required_concepts': record['required_concepts'],
                        'masks': choices, 'trace': trace,
                        'gates': module.trace['gates'][0].cpu().tolist(),
                        'node_router': module.trace['node_router'][0].cpu().tolist()}
                    traces.write(json.dumps(trace_row) + '\n')
                    traces.flush()
                    pooled.append(module.content_trace['pooled'][0].cpu().numpy())
                    messages.append(original_messages[0].cpu().numpy())
                    responses.append(module.content_trace['expert_response'][0].cpu().numpy())
                for condition in rows:
                    module.drop_ids = choices[condition]
                    batch = batch_for(runtime, record, '111')
                    actual = module(nodes)[0]
                    assert torch.equal(original_selection, module.retrain_trace['selection'])
                    bounded = module.bounded_residual(actual, reference)
                    diagnostics = {'removed_message_norm': float(original_messages[:, module.drop_ids].norm()),
                        'residual_l2_change': float((actual - original).norm()),
                        'bounded_residual_l2_change': float((bounded - original_bounded).norm()),
                        'bounded_residual_rms': float(bounded.float().square().mean().sqrt()),
                        'residual_rms': float(actual.float().square().mean().sqrt()),
                        'reference_rms': float(reference.float().square().mean().sqrt()),
                        'mask_ids': module.drop_ids,
                        'core_overlap': len(set(module.drop_ids) & set(choices['mask_selected']))}
                    generated = runtime.model.generate(**batch, do_sample=False, max_new_tokens=320)
                    tail = generated[0, batch['input_ids'].shape[1]:]
                    raw = runtime.processor.tokenizer.decode(tail, skip_special_tokens=True).strip()
                    terms, valid = parse_structured(raw)
                    row = prediction_row(record, terms, valid, raw=raw,
                        budget_reached=len(tail) == 320, **diagnostics)
                    rows[condition].append(row)
                    streams[condition].write(json.dumps(row, ensure_ascii=False) + '\n')
                    streams[condition].flush()
                    del generated
                if number % 10 == 0:
                    save(out / 'status.json', {'phase': 'evaluating', 'completed': number + 1,
                        'total': 560, 'conditions': list(rows), 'seconds': time.time() - started,
                        'test_read': False})
            summary = {condition: runtime.metric.score(values) for condition, values in rows.items()}
            save(out / 'summary.json', summary)
            if traces is not None:
                np.savez_compressed(out / 'baseline_content.npz',
                    indices=np.asarray([record['index'] for record in records]),
                    pooled=np.stack(pooled), messages=np.stack(messages),
                    expert_responses=np.stack(responses))
                save(out / 'edge_schema.json', module.describe())
            save(out / 'complete.json', {'phase': 'complete', 'summary': summary,
                'seconds': time.time() - started, 'test_read': False, 'training_updates': 0})
            save(out / 'status.json', {'phase': 'complete', 'test_read': False})
    finally:
        for stream in streams.values():
            stream.close()
        if traces is not None:
            traces.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=int, choices=[0, 1], required=True)
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        folder = ROOT / f'selected_relation_masks_v1_gpu{arguments.queue}'
        if folder.exists():
            save(folder / 'status.json', {'phase': 'failed', 'error': repr(error), 'test_read': False})
        traceback.print_exc()
        raise
