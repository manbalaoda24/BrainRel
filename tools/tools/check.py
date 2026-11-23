"""Synthetic CPU contracts for archived modules; no participant data required."""
import argparse
import ast
import json
import sys
from pathlib import Path


def main(args):
    root = args.source_root.resolve()
    sys.path[:0] = [str(root / 'scripts' / name) for name in [
        'mllm', 'faced']]
    import numpy as np
    import torch
    from nodes import NumericEvidenceFilter, measurement_nodes
    from experts import HypergraphRelationExperts
    from parser import parse_structured
    from exact import metrics

    torch.manual_seed(123)
    rng = np.random.default_rng(123)
    values = rng.normal(size=(10, 5, 30)).astype('float32')
    geometry = rng.uniform(size=(256, 30)).astype('float32')
    geometry /= geometry.sum(-1, keepdims=True)
    limits = np.stack([np.full(5, -2), np.full(5, 2)]).astype('float32')
    nodes = measurement_nodes(values, geometry, limits)
    assert nodes.shape == (100, 8) and np.isfinite(nodes).all()
    base = NumericEvidenceFilter(80).eval()
    graph = HypergraphRelationExperts(base, 80).eval()
    assert graph.incidence.shape == (170, 100)
    assert [int((graph.edge_types == kind).sum()) for kind in range(3)] == [50, 100, 20]
    for number, edge in enumerate(graph.edge_metadata):
        recipients = graph.recipients[number].nonzero().flatten().tolist()
        if edge['type'] == 'temporal':
            center = (number - 50) // 10
            assert all(graph.coordinates[index][0] == center for index in recipients)
    with torch.no_grad():
        tensor = torch.tensor(nodes).unsqueeze(0)
        original = base(tensor)[0]
        actual, router, attention = graph(tensor)
        torch.testing.assert_close(actual, original, rtol=1e-5, atol=1e-6)
        assert actual.shape == (1, 32, 80)
        assert router.shape == (1, 100, 3) and attention.shape == (1, 32, 100)
        delta = torch.randn_like(actual) * 100
        reference = torch.randn_like(actual)
        bounded = graph.bounded_residual(delta, reference)
        assert torch.all(bounded.square().mean(-1).sqrt() <= .2 * reference.square().mean(-1).sqrt() + 1e-6)
        assert torch.equal(graph.bounded_residual(delta, torch.zeros_like(reference)), torch.zeros_like(delta))
        trace = graph.relation_trace()
        assert 'slot_to_node' in trace and 'top_relations' in trace
    assert parse_structured('{"emotions":["joy"]}') == (['joy'], True)
    assert parse_structured('not JSON') == ([], False)
    empty = metrics([(set(), set())])
    assert empty['jaccard'] == 1 and empty['micro_f1'] == 0
    path = root / 'scripts/faced/run.py'
    tree = ast.parse(path.read_text())
    atlas = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'atlas')
    from PIL import Image, ImageDraw
    namespace = dict(np=np, Image=Image, ImageDraw=ImageDraw,
                     BANDS=[(name, 0, 0) for name in ['delta', 'theta', 'alpha', 'beta', 'gamma']])
    exec(compile(ast.Module(body=[atlas], type_ignores=[]), str(path), 'exec'), namespace)
    image = namespace['atlas'](values[0], geometry, limits)
    assert image.size == (320, 96)
    print(json.dumps(dict(passed=True, device='cpu', synthetic_only=True,
        checks=['100x8_nodes', '170_hyperedges', 'temporal_center_recipients',
                '32_slot_readout', 'router_and_attention_shapes', 'initial_equivalence',
                'RMS_cap', 'zero_reference', 'trace_schema', 'parser', 'empty_set_policy', 'atlas_shape'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    main(parser.parse_args())
