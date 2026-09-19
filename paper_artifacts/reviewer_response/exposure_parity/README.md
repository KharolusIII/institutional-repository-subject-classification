# Exposure-parity reviewer artifacts

This directory contains the privacy-safe aggregate evidence for the post-hoc
validation experiment that restricts BM25 and fine-tuned LaBSE to identical raw
full-text spans. Start with [`REPORT.md`](REPORT.md) for the interpretation and
paste-ready reviewer/manuscript text.

## Contents

- `tables/metrics_by_seed.csv`: complete threshold-selection (`calibration` in
  the implementation) and validation metrics.
- `tables/metrics_aggregate.csv`: run-instance summaries; LaBSE uncertainty is
  the sample SD across three independent training seeds, while the thresholded
  BM25 F1 values are identical across the three run contexts.
- `tables/paired_validation_differences.csv`: BM25-minus-LaBSE differences.
- `tables/epoch_history.csv` and `tables/convergence_summary.csv`: the complete
  early-stopping record and selected checkpoints.
- `tables/exposure_totals.csv` and `tables/exposure_distributions.csv`: aggregate
  common-span coverage, with no document identifiers.
- `tables/per_label_validation_by_seed.csv` and
  `tables/per_label_validation_aggregate.csv`: 37-label validation summaries.
- `tables/thresholds_by_seed.csv`: decision thresholds selected on the
  internally named `calibration` partition, in the frozen label order.
- `configs/`: path-neutral configurations reconstructed from each resolved run.
- `provenance.json`, `environment.json` and `source_artifact_checksums.csv`:
  curated reproducibility records.
- `MANIFEST.sha256`: byte-level checksums for every other file in this directory.

No test inference or evaluation was performed in this follow-up; historical
test results from the earlier study had already been inspected. No source text,
repository handle, item-level prediction, split membership, NumPy array,
model/checkpoint, cache, log, credential or personal/cloud path is included.

The `code_commit` recorded in `provenance.json` identifies the historical
execution revision. This aggregate evidence release does not claim to
distribute the restricted corpus or the private training workspace.

## Rebuild

From the repository root, point the builder to the three complete private run
directories:

```bash
python scripts/build_exposure_parity_reviewer_package.py \
  --seed13-dir /path/to/seed13-validation-run \
  --seed42-dir /path/to/seed42-validation-run \
  --seed73-dir /path/to/seed73-validation-run \
  --overwrite
```

The builder validates run compatibility, reconstructs aggregate per-label
metrics in memory, scans its outputs for private payloads, creates the
deterministic ZIP, and refreshes the top-level paper-artifact manifest.
