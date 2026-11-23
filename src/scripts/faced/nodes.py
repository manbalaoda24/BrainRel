"""Relation experts and bounded auxiliary residuals, preserving the image path."""
import numpy as np
import torch
from torch import nn


def measurement_nodes(values, weights, limits):
    assert values.shape == (10, 5, 30)
    panels = (values @ weights.T).reshape(10, 5, 16, 16)
    displayed = np.clip((panels - limits[0][None, :, None, None]) /
                        (limits[1] - limits[0])[None, :, None, None], 0, 1)
    nodes = []
    for window in range(10):
        for band in range(5):
            for region in range(2):
                patch = displayed[window, band, region * 8:(region + 1) * 8]
                previous = displayed[max(0, window - 1), band, region * 8:(region + 1) * 8]
                nodes.append([patch.mean(), patch.std(), patch.min(), patch.max(),
                              patch.mean() - previous.mean(), window / 9., band / 4., float(region)])
    result = np.asarray(nodes, dtype='float32')
    assert result.shape == (100, 8) and np.isfinite(result).all()
    return result


class NumericEvidenceFilter(nn.Module):
    def __init__(self, hidden_size, slots=32):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.LayerNorm(64))
        coordinates = torch.tensor([(window, band, region) for window in range(10) for band in range(5) for region in range(2)])
        window, band, region = coordinates.T
        masks = [((window[:, None] == window) & (band[:, None] == band)),
                 ((band[:, None] == band) & (region[:, None] == region) & ((window[:, None] - window).abs() <= 1)),
                 ((window[:, None] == window) & (region[:, None] == region))]
        for number, mask in enumerate(masks):
            matrix = mask.float()
            self.register_buffer(f'relation_{number}', matrix / matrix.sum(-1, keepdim=True))
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(64, 64), nn.GELU()) for _ in masks])
        self.router = nn.Linear(64, 3)
        self.queries = nn.Parameter(torch.randn(1, slots, 64) * .02)
        self.attention = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
        self.project = nn.Linear(64, hidden_size)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        self.gate_logit = nn.Parameter(torch.tensor(-2.))
        self.disabled_expert = None

    def forward(self, nodes):
        encoded = self.encode(nodes.float())
        weights = self.router(encoded).softmax(-1)
        if self.disabled_expert is not None:
            if self.disabled_expert not in {0, 1, 2}:
                raise ValueError('Unknown relation expert')
            keep = torch.ones(3, device=weights.device, dtype=weights.dtype)
            keep[self.disabled_expert] = 0
            weights = weights * keep
        experts = torch.stack([expert(getattr(self, f'relation_{number}') @ encoded)
                               for number, expert in enumerate(self.experts)], dim=-2)
        filtered = encoded + (weights.unsqueeze(-1) * experts).sum(-2)
        queries = self.queries.expand(nodes.shape[0], -1, -1)
        selected, attention = self.attention(queries, filtered, filtered, need_weights=True)
        return self.project(selected) * self.gate_logit.sigmoid(), weights, attention

    @staticmethod
    def bounded_residual(delta, reference):
        cap = .2 * reference.float().square().mean(-1, keepdim=True).sqrt().detach()
        rms = delta.float().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        return delta * (cap / rms).clamp(max=1.)
