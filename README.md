# BrainRel

EEG atlases, numerical relations and video context for emotion generation.
This is a local release candidate, not a published repository.

## Code

- `src/scripts/faced/nodes.py`: numerical nodes and residual interface.
- `src/scripts/faced/hypergraph.py`: relation membership.
- `src/scripts/faced/experts.py`: relation experts and node routing.
- `src/scripts/mllm/train_graph.py`: relational adaptation.
- `src/scripts/eav/`: EAV preparation and adaptation.
- `tools/infer.py`: frozen FACED inference.
- `tools/check.py`: synthetic CPU checks.

```bash
python tools/verify.py
python tools/tables.py --check
python tools/check.py --source-root src
python tools/infer.py --data-root /YOUR/FACED --model /YOUR/ONEEMO \
  --output /NEW/OUTPUT --arm joint --limit 1 --compare-archived
```

Use `--limit 0` for all 644 FACED test inputs. The default is one smoke sample.
Read [training](docs/training.md), [data access](docs/access.md),
[reproduction](docs/reproduce.md) and [third-party notices](docs/third_party.md).
The observed environment is not a complete dependency lock. Legacy training
scripts still require the documented upstream code and authorized data caches.

## Scope

Renaming changes imports and source paths, not model parameters, prompts,
checkpoint keys, dataset filenames, or archived scores. `manifests/names.json`
maps each source back to the sealed snapshot. Historical source revisions are
not all recovered; exact from-scratch retraining is not promised.

FACED uses held-out subjects with shared stimulus videos. EAV scores 300 supported
weak references among 700 validation inputs, not an independently annotated test.
The historical curriculum and fresh-start ablations have different budgets;
selected checkpoint prefixes must not be described as completed schedules.
Recorded paths are computational provenance, not physiological causality.

No data, participant predictions, weights, credentials or third-party metric
assets are included. Licenses and redistribution permissions need review before
public release. No training or automatic downloading is started by packaging.
