# Synthetic examples

This directory contains the only record-level example distributed with the
public repository. Every value is generated and does not describe a real
repository item.

## Input

`input/dummy_metadata.csv` demonstrates the expected metadata fields, multiple
labels, multilingual abstracts, keywords and inline full text. Regenerate it
with:

```bash
python scripts/generate_dummy_data.py
```

The example uses inline full text so it can run without a mapping file, Drive,
or external corpus. The README under `data/` documents the alternative mapping
CSV and TXT/Parquet interfaces for authorized datasets.

## Output

`output/` is a curated successful smoke-run result. It illustrates:

- dataset and label statistics;
- the validation grid and frozen test result;
- thresholds, confidence intervals and per-label metrics;
- the generated English evaluation report; and
- two example figures.

Operational logs, environments, Git paths, manifests, item-level predictions,
and Parquet files are intentionally omitted from the public example.

Results on this toy corpus demonstrate execution only and must not be compared
with the paper's results. Aggregate evidence for the paper appears separately
under `paper_artifacts/runs/`.
