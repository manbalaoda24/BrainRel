"""Explicit measurement hyperedges with traceable membership, not causal attribution."""
import torch
from torch import nn
from nodes import NumericEvidenceFilter


def build_hyperedges():
    coordinates = [(window, band, region) for window in range(10)
                   for band in range(5) for region in range(2)]
    edges = []
    for window in range(10):
        for band in range(5):
            members = [index for index, coord in enumerate(coordinates)
                       if coord[:2] == (window, band)]
            edges.append({'type': 'spatial', 'members': members})
    for center in range(10):
        for band in range(5):
            for region in range(2):
                members = [index for index, coord in enumerate(coordinates)
                           if coord[1:] == (band, region) and abs(coord[0] - center) <= 1]
                edges.append({'type': 'temporal', 'members': members})
    for window in range(10):
        for region in range(2):
            members = [index for index, coord in enumerate(coordinates)
                       if coord[0] == window and coord[2] == region]
            edges.append({'type': 'cross_band', 'members': members})
    incidence = torch.zeros(len(edges), len(coordinates), dtype=torch.bool)
    for number, edge in enumerate(edges):
        incidence[number, edge['members']] = True
    return coordinates, edges, incidence


class HypergraphEvidenceFilter(nn.Module):
    def __init__(self, base, hidden_size, slots=32):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.coordinates, self.edge_metadata, incidence = build_hyperedges()
        self.register_buffer('incidence', incidence)
        types = ['spatial', 'temporal', 'cross_band']
        self.register_buffer('edge_types', torch.tensor([types.index(edge['type'])
                                                       for edge in self.edge_metadata]))
        self.encode = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.LayerNorm(64))
        self.member_score = nn.Linear(64, 1)
        self.type_embedding = nn.Embedding(3, 64)
        self.edge_transform = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.LayerNorm(64))
        self.edge_gate = nn.Linear(64, 1)
        self.queries = nn.Parameter(torch.randn(1, slots, 64) * .02)
        self.attention = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
        self.project = nn.Linear(64, hidden_size)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        self.gate_logit = nn.Parameter(torch.tensor(-2.))

    def forward(self, nodes):
        baseline, _, _ = self.base(nodes)
        encoded = self.encode(nodes.float())
        logits = self.member_score(encoded).squeeze(-1).unsqueeze(1)
        logits = logits.expand(-1, self.incidence.shape[0], -1)
        members = logits.masked_fill(~self.incidence.unsqueeze(0), -torch.inf).softmax(-1)
        edges = members @ encoded + self.type_embedding(self.edge_types).unsqueeze(0)
        edges = self.edge_transform(edges)
        gates = self.edge_gate(edges).sigmoid()
        selected, attention = self.attention(self.queries.expand(nodes.shape[0], -1, -1),
                                             edges * gates, edges * gates, need_weights=True)
        delta = self.project(selected) * self.gate_logit.sigmoid()
        return baseline + delta, gates.squeeze(-1), attention

    def describe(self):
        return {'node_order': ['window', 'band', 'image_half'],
                'coordinates': self.coordinates, 'edges': self.edge_metadata,
                'region_definition': 'Upper/lower halves of interpolated 16x16 maps; not anatomical regions',
                'temporal_definition': 'Centered radius-one windows, clipped at boundaries',
                'interpretation': 'Membership is exact provenance; gates and attention are not causal explanations'}

    bounded_residual = staticmethod(NumericEvidenceFilter.bounded_residual)
