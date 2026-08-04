# Final manuscript consistency audit

Manuscript reviewed: `CACIC_2026_Beyond_Embeddings_v3_FINAL (1).docx`.

Evidence used:

- main v3 run `20260802T165746Z_full_20k_v3_segmented_finetuned_2026`;
- focused audit `20260804T102544Z_convergence_audit_20k_2026`;
- source commits `7181fcceb7a62a155e589b53015a9de7a31dc083` and
  `6e46e5c2af0249591d1f59b07b82ad43125bd0a2`;
- resolved configurations, aggregate CSVs, logs, prediction artifacts and the
  implementation of ingestion, language processing, vectorization,
  fine-tuning, calibration, selection and reporting.

## Verdict

The manuscript's four tables and its principal numerical claims agree with the
stored evidence. The corpus counts, split, language audit, coverage, validation
leaders, family test results, final metrics, confidence intervals, learning
curves and per-subject examples are correctly reported to the shown precision.

Two embedded workflow figures must be replaced before submission. They are
legacy diagrams and are inconsistent with the final protocol.

## Required figure corrections

The current dataset diagram contains `20,000`, `fulltext (100K)`, and a
three-way train/validation/test split. The final experiment instead uses:

- 19,710 analytical documents;
- a fair 400,000-character full-text budget per item;
- train 12,836, calibration 1,927, validation 1,988 and test 2,959;
- preserved multiple abstract/full-text units; and
- exact-full-text duplicate grouping before splitting.

The current experimental diagram omits BoW, the dedicated calibration split,
the separate frozen/fine-tuned chunk strategies and the distinction between
validation selection and held-out evaluation. Its internal figure numbering is
also reversed relative to its placement in the manuscript.

Use these generated replacements without an internal `Fig.` label:

- `figures/dataset_construction_workflow.png` for manuscript Fig. 1;
- `figures/experimental_protocol_workflow.png` for manuscript Fig. 2.

## Verified manuscript tables

### Table 1

All 12 reported corpus and split values match `dataset_statistics.csv`,
`text_unit_statistics.csv`, `fulltext_coverage.csv` and the frozen split.

### Table 2

The two agreement rates and the historical-reference accuracy match the stored
language artifacts. `Agreement` is correctly distinguished from `accuracy`.

### Table 3

All seven family rows match frozen validation selection and held-out family
evaluation. The BM25 row correctly uses the convergence-resolved final model.
The text also correctly reports the separate validation Micro-F1 leader:
BM25 + all fields + logistic regression (`0.777362`).

### Table 4

All final metrics and both 95% bootstrap intervals match the convergence-audit
test output and its 1,000-resample bootstrap.

## Verified methodological terminology

- `One-vs-Rest`, `LinearSVC`, `BCEWithLogitsLoss`, `AdamW`, Macro-F1,
  Micro-F1, Hamming loss and iterative multi-label stratification are used
  correctly.
- Frozen embeddings use overlapping token chunks and length-weighted
  hierarchical pooling.
- Fine-tuning uses non-overlapping uniformly selected chunks, at most eight per
  text unit, and weighted aggregation of logits.
- Per-label thresholds use calibration support of at least 20; otherwise the
  global threshold is inherited.
- The formula shown is the implemented BM25 variant.
- `83.45%` is correctly described as represented-character coverage and not as
  transformer token coverage.

## Recommended final additions to the manuscript

Add a short code/data availability paragraph, for example:

> Code, the executable Colab notebook, frozen experiment configurations, and
> aggregate reviewer artifacts are available at
> https://github.com/KharolusIII/institutional-repository-subject-classification.
> Repository metadata and full text are not redistributed; the public notebook
> runs end to end with synthetic data, and the supplementary package contains
> no item-level records or credentials.

Refer explicitly to `SUPPLEMENTARY_MATERIAL.md` after the reproducibility
paragraph or in the submission system. Also confirm whether the venue requires
numeric LNCS citations rather than the manuscript's current author-date style.
