# Runs and provenance

## Main v3 experiment

| Property | Value |
|---|---|
| Run | `20260802T165746Z_full_20k_v3_segmented_finetuned_2026` |
| Source commit | `7181fcceb7a62a155e589b53015a9de7a31dc083` |
| Configuration | `configs/run_full_20k_v3_2026.yaml` and inherited profiles |
| Cohort | 19,710 documents; 37 labels; 24,326 assignments |
| Split | 12,836 train; 1,927 threshold selection (internal name: `calibration`); 1,988 validation; 2,959 test |
| Search | 189 sparse + 42 fixed sentence-embedding + 2 supervised sequence-classifier candidates |
| Primary criterion | Validation Macro-F1 |
| Initial winner | language stopwords + full text + BM25 + LinearSVC |

This run established the corpus, frozen split, preprocessing comparison,
representation-family winners, transformer learning curves and initial test
comparison. It also generated the frozen family predictions used for paired
analysis.

The immutable run identifiers `sbert*` refer to DistilUSE. Fixed candidates use
the complete Sentence-Transformers pipeline; supervised candidates initialize
`AutoModelForSequenceClassification` from the checkpoint's Transformer module.
See `tables/model_nomenclature.csv` and `docs/MODEL_NOMENCLATURE.md`.

## Focused convergence audit

| Property | Value |
|---|---|
| Run | `20260804T102544Z_convergence_audit_20k_2026` |
| Source commit | `6e46e5c2af0249591d1f59b07b82ad43125bd0a2` |
| Configuration | `configs/run_convergence_audit_20k_2026.yaml` |
| Reused evidence | Identical cohort, split fingerprint and prepared corpus |
| Search | BM25/TF-IDF × four controlled LinearSVC settings = 8 fits |
| Retained setting | BM25, `C=0.5`, stored `tol=0.0005` |
| Validation | Macro-F1 0.749656; Micro-F1 0.773168 |
| Final test | Macro-F1 0.752228; Micro-F1 0.768458 |

For BM25 with `C=1.0`, Humanidades reached `max_iter=200000` under both
tolerances. Both `C=0.5` variants converged for all 37 labels and tied on every
stored validation metric. The implementation returned `tol=0.0005` first; no
performance superiority over `tol=0.0001` is claimed. All 37 final test refits
converged.

## Cross-run paired comparison

Final-audit predictions were aligned by handle with the frozen main-run family
predictions. The published `tables/paired_bootstrap_updated.csv` uses 1,000
paired item-level resamples, seed 42, and reports candidate minus final BM25.
The resampling requires item identifiers internally, but the public export
contains only aggregate differences and intervals.

## Environment reported by the runs

| Component | Version |
|---|---|
| Python | 3.12.13 |
| scikit-learn | 1.6.1 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.13.1 |
| Sentence Transformers | 5.6.0 |
| stopwordsiso | 0.7.1 |

Elapsed times are not used for model comparison because the work was resumed
across heterogeneous Colab sessions. Session and stage timings are retained as
operational provenance only.
