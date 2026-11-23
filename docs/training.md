# Training lineage and interpretation

## Historical main model

The historical preparation is a chain of continuations, not one clean five-stage run.
S0 comprises native visual reading, paired reading, classifier/reading multitask preparation,
and region-binding preparation. The classifier checkpoint was an auxiliary preparation dependency;
it is not a classifier prior supplied to final emotion generation. S1 comprises label-only
open-description preparation followed by image-only continuation. S2 adds the numerical filter;
S3 adapts the inherited LoRA using EEG-only and joint examples; S4 refines integrated dense
hypergraph experts. Do not replace these with the fresh-start comparison below.

Inference restores the released base plus the S3 checkpoint (including inherited merger and
numerical state), then the selected S4 hypergraph state. S4 is not a complete standalone model.
All ten discovered checkpoint files exist. A filename, selected epoch and hash are recorded for each.
Classifier dependency edges are distinct from language/merger initialization edges.

### Recorded dependency edges

- `v2_video_joint_lora_pilot_v1/best.pt` → `v2_hypergraph_experts_pilot_v1/best.pt` (initial_checkpoint)
- `numeric_evidence_filter_full_v1/best.pt` → `v2_video_joint_lora_pilot_v1/best.pt` (initial_checkpoint)
- `open_input_image_v1/best.pt` → `numeric_evidence_filter_full_v1/best.pt` (initial_checkpoint)
- `open_description_ew_label_v1/best.pt` → `open_input_image_v1/best.pt` (initial_checkpoint)
- `region_binding_sft_v1/best.pt` → `open_description_ew_label_v1/best.pt` (initial_checkpoint)
- `classifier_visual_multitask_v1/last.pt` → `region_binding_sft_v1/best.pt` (initial_checkpoint)
- `readout_recipe_2x2_v1/full_borrowed/best.pt` → `region_binding_sft_v1/best.pt` (classifier_checkpoint)
- `traceable_visual_reader_paired_v2/best.pt` → `classifier_visual_multitask_v1/last.pt` (initial_checkpoint)
- `readout_recipe_2x2_v1/full_borrowed/best.pt` → `classifier_visual_multitask_v1/last.pt` (classifier_checkpoint)
- `traceable_visual_reader_sft_v1/best.pt` → `traceable_visual_reader_paired_v2/best.pt` (initial_checkpoint)

The snapshot contains current source bytes. Some recorded historical source hashes may not match
the current scripts; consult `manifests/historical_source_audit.json`. An unmatched source hash
is an explicit limitation for exact from-scratch replay, not permission to rewrite history.
Epochs in `stage_lineage.json` are selected checkpoint epochs, not the allocated training duration.

## Fresh common-initialization comparison

All three runs were allocated 150 updates and 1,200 presentations, with selection at 0/50/100/150.
Mixed/staged schedules contain the same complete task-batch multiset and planned module updates.
Their selected update-100 prefixes differ: mixed LoRA/merger/core/relation = 59/20/18/41;
staged = 70/30/30/30. The selected staged model has not yet run final relation refinement.
Direct joint selected update150, with 150 updates per adaptable group. This is not a pure ordering control.
The full historical model is a different-budget reference, not a fourth matched training arm.

## EAV

EAV uses its own S0–S4 v4 continuation chain and EAV-fitted geometry/display limits.
The reported `heldout700_v1/small_joint` reference points to that chain, not the later full v5 run.
The archived evaluation is the 700-input validation panel, with only supported references scored;
do not silently rename it an independent test set. See `manifests/inputs_and_eav.json`.
The selected main run generated700 inputs and scored300 supported references; its unchanged
61.56% score is not an emotion score over700 independent reference annotations.
