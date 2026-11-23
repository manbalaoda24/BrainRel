"""Hyperedge aggregation inside the original three relation experts."""
import torch
from torch import nn

from nodes import NumericEvidenceFilter
from hypergraph import build_hyperedges


class HypergraphRelationExperts(NumericEvidenceFilter):
    def __init__(self, base, hidden_size):
        super().__init__(hidden_size, slots=base.queries.shape[1])
        self.load_state_dict(base.state_dict(), strict=True)
        self.requires_grad_(False)
        self.experts.requires_grad_(True)
        self.router.requires_grad_(True)
        self.coordinates, self.edge_metadata, incidence = build_hyperedges()
        types = ['spatial', 'temporal', 'cross_band']
        edge_types = torch.tensor([types.index(edge['type']) for edge in self.edge_metadata])
        recipients = incidence.clone()
        for number, edge in enumerate(self.edge_metadata):
            if edge['type'] == 'temporal':
                center = (number - 50) // 10
                recipients[number] &= torch.tensor([coord[0] == center for coord in self.coordinates])
        for kind in range(3):
            assert (recipients[edge_types == kind].sum(0) == 1).all()
        self.register_buffer('incidence', incidence)
        self.register_buffer('recipients', recipients.float())
        self.register_buffer('edge_types', edge_types)
        self.member_scores = nn.ModuleList([nn.Linear(64, 1) for _ in types])
        self.edge_gates = nn.ModuleList([nn.Linear(64, 1) for _ in types])
        for layer in [*self.member_scores, *self.edge_gates]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.disabled_edge = None
        self.trace = None

    def forward(self, nodes):
        encoded = self.encode(nodes.float())
        weights = self.router(encoded).softmax(-1)
        if self.disabled_expert is not None:
            if self.disabled_expert not in {0, 1, 2}:
                raise ValueError('Unknown relation expert')
            keep = torch.ones(3, device=weights.device)
            keep[self.disabled_expert] = 0
            weights = weights * keep
        expert_outputs, edge_weights, gates = [], [], []
        for kind in range(3):
            selected = self.edge_types == kind
            membership = self.incidence[selected]
            logits = self.member_scores[kind](encoded).squeeze(-1).unsqueeze(1)
            logits = logits.expand(-1, int(selected.sum()), -1)
            members = logits.masked_fill(~membership.unsqueeze(0), -torch.inf).softmax(-1)
            edges = members @ encoded
            gate = 2 * self.edge_gates[kind](edges).sigmoid()
            if self.disabled_edge is not None:
                if not 0 <= self.disabled_edge < len(self.edge_metadata):
                    raise ValueError('Unknown hyperedge')
                edge_numbers = selected.nonzero().flatten()
                gate = gate * (edge_numbers != self.disabled_edge)[None, :, None]
            contexts = self.recipients[selected].T @ (edges * gate)
            expert_outputs.append(self.experts[kind](contexts))
            edge_weights.append(members)
            gates.append(gate.squeeze(-1))
        experts = torch.stack(expert_outputs, dim=-2)
        filtered = encoded + (weights.unsqueeze(-1) * experts).sum(-2)
        selected, attention = self.attention(self.queries.expand(nodes.shape[0], -1, -1),
                                             filtered, filtered, need_weights=True)
        self.trace = {'members': torch.cat(edge_weights, dim=1).detach(),
                      'gates': torch.cat(gates, dim=1).detach(),
                      'node_router': weights.detach(), 'slot_to_node': attention.detach()}
        return self.project(selected) * self.gate_logit.sigmoid(), weights, attention

    def relation_trace(self):
        trace = self.trace
        routed = trace['node_router'][:, :, self.edge_types]
        paths = torch.einsum('bsn,bne,en->bse', trace['slot_to_node'], routed, self.recipients)
        summaries = []
        for batch in range(paths.shape[0]):
            mass = paths[batch].mean(0)
            top = mass.topk(10).indices.tolist()
            summaries.append([{'edge_id': number, 'type': self.edge_metadata[number]['type'],
                               'members': self.edge_metadata[number]['members'],
                               'member_weights': trace['members'][batch, number, self.incidence[number]].cpu().tolist(),
                               'edge_gate': float(trace['gates'][batch, number]),
                               'routing_mass': float(mass[number])} for number in top])
        return {'node_router': trace['node_router'].cpu().tolist(),
                'slot_to_node': trace['slot_to_node'].cpu().tolist(),
                'slot_to_edge_routing_mass': paths.cpu().tolist(), 'top_relations': summaries,
                'interpretation': 'Unsigned routing paths, not signed residual contributions or emotion attribution'}

    def describe(self):
        return {'node_order': ['window', 'band', 'image_half'], 'coordinates': self.coordinates,
                'edges': [{**edge, 'recipients': self.recipients[number].nonzero().flatten().tolist()}
                          for number, edge in enumerate(self.edge_metadata)],
                'temporal_readout': 'Each node reads only its own centered temporal edge; other members supply context',
                'normalization': 'Within-edge softmax, one recipient edge per type/node; gate=2*sigmoid(logit)',
                'initialization': 'Uniform members and unit gates reproduce original neighbor means to numerical tolerance',
                'region_definition': 'Two image halves, not anatomical regions',
                'scope': 'Single numerical residual interface; no additive parallel graph; no causal interpretation'}
