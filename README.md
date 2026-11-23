# BrainRel

Official implementation of **"BrainRel: Capturing Traceable Physiological Relations for Multimodal Open-Vocabulary Emotion Recognition"**.

BrainRel is a decoupled multimodal framework that pairs deterministic EEG topographic atlases and source-indexed numerical hypergraph relations with video context for open-vocabulary emotion generation.

## Repository Structure

- `src/scripts/faced/nodes.py`: Structured numerical EEG nodes and bounded residual injection interface.
- `src/scripts/faced/hypergraph.py`: Spatial, temporal, and cross-band hypergraph topology.
- `src/scripts/faced/experts.py`: Hypergraph relation experts and sample-adaptive node routing.
- `src/scripts/mllm/train_graph.py`: Progressive relational adaptation scripts.
- `src/scripts/eav/`: EAV preprocessing and cross-scenario adaptation scripts.
- `tools/infer.py`: Inference script for FACED evaluation.
- `tools/check.py`: Lightweight CPU sanity checks for core modules.

## Quick Start

### 1. Sanity Checks
```bash
python tools/verify.py
python tools/check.py --source-root src
```

### 2. FACED Inference
```bash
python tools/infer.py \
  --data-root /PATH/TO/FACED \
  --model /PATH/TO/CHECKPOINT \
  --output /PATH/TO/OUTPUT \
  --arm joint \
  --limit 1
```
Use `--limit 0` to evaluate all 644 held-out FACED test trials (default `--limit 1` runs a single smoke test).

## Documentation

- [Training Protocol](docs/training.md)
- [Data Access & Preparation](docs/access.md)
- [Reproduction Notes](docs/reproduce.md)
- [Third-Party Notices](docs/third_party.md)

## Evaluation Notes & Data Policy

- **Datasets**: No raw EEG/video data, participant predictions, pretrained weights, or credentials are distributed in this repository. Please obtain FACED and EAV directly from their official providers under their respective licenses (see `docs/access.md`).
- **Evaluation Scope**: FACED evaluates 23 held-out test participants with shared stimulus videos. EAV evaluates 300 ASR-supported weak references among 700 validation interactions. Recorded forward paths provide computational provenance for the numerical residual rather than physiological causality.
- **Third-Party Assets**: Upstream backbones, baseline models, and emotion-wheel evaluation tools remain subject to their original licenses (see `docs/third_party.md`).
