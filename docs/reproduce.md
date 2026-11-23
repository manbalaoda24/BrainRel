# Non-drift policy and acceptance levels

1. Original data directories, selected weights and prediction files remain read-only.
2. Source, checkpoint, input, prompt and metric fingerprints are separate. The package manifest
   checks bytes; matching metrics alone does not prove matching predictions.
3. Stored-prediction regression re-runs the original pinned EW evaluator and exact-set diagnostics.
   All compared fields must agree within1e-12. Only display tables round to two decimals.
4. Synthetic CPU checks cover100×8nodes,170edges,50/100/20expert counts, temporal central recipients,
   32slot readout, initialization equivalence,20%RMS cap, parser and empty-set policy.
5. A new GPU replay must compare raw generations, parsed terms and validity against the archived
   records. A mismatch is reported as a new reproduction result, never used to replace the old score.
6. Matching input token IDs, image/video grids, per-position embeddings and residuals is the next
   diagnostic level if output differs. No test-driven threshold/prompt/checkpoint changes are allowed.

Observed dependency versions are not a complete install lock. CUDA build, native model support,
fonts, video decoding and upstream custom code can affect exact replay. Do not assume that a modern
default font, interpolation rule or processor budget is equivalent. Archived source is never edited
to hide these differences. `results/synthetic_cpu_smoke.json` records the performed CPU checks.

Training-budget accounting distinguishes allocated schedules, completed runs and selected prefixes.
Do not describe the update100 staged checkpoint as having completed all five tasks. Historical S1
contains more than the final135-update continuation. Historical preparation included classifier-related
tasks, even though the final prompt has no classifier prior. These facts must remain in documentation.
