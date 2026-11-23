# Data, model and upstream access

FACED and EAV must be obtained through their approved dataset access procedures. The package does
not provide credentials, authenticated links, raw signals, stimulus/facial videos, audio, self-ratings,
ASR transcripts or reference-label exports. A derived graph/image/label is not automatically cleared
for redistribution. Synthetic examples are the default public examples.

FACED expects the original relative cache paths recorded in `configs/faced_main.json`. Limits must
remain training-fitted; do not refit on validation/test. The main runtime requires training-era
`official200_classification_v1/band_logpower.npy` during initialization and swaps to the separately
prepared test cache before inference. Pixel conversion, nearest-neighbor resizing, frame sampling,
and the independently assigned measurement-query band are part of the input contract.

OneEmo base weights and tokenizer/processor files are supplied by the user. The source hash ledger
identifies the observed base files without redistributing them. The S3 and S4 incremental files are
also user-supplied until model/data permissions are reviewed. Only load trusted original checkpoints:
the archived PyTorch loaders use weights_only=False.

Official EW assets are not bundled. Obtain the exact upstream revision and verify all asset hashes
listed in `results/score_regression.json`, preserving its mapping and empty-reference behavior.
No network downloader runs automatically. Missing assets fail instead of substituting a new mapping.

EAV source uses participant-face video, not external stimulus video. It uses EAV-specific channel
geometry and training-fitted normalization. Check acquisition-order assumptions before anatomy claims.
Its weak ASR-derived references are not participant self-reports or independently verified human gold.
