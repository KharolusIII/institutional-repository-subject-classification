# Post-run checklist: segmented and supervised Transformer protocol v3

This checklist converts the limitations observed in the completed 20k run
(`000b9d2`) into acceptance criteria for the next experiment.

| Finding from the completed run | v3 resolution | Evidence artifact |
|---|---|---|
| Thresholds were not comparable during validation | Dedicated calibration split and per-configuration, per-label thresholds | `results_validation.csv`, `thresholds.json` |
| Declared language was exported as `und` | Values from `dc.language*` and `sedici2003.idioma[es]` are parsed; suffixes identify metadata-field language | `language_agreement*.csv` |
| LangID disagreement audit was unavailable | Consolidated silver/gold abstract ground truth is matched by handle and text hash | `abstract_language_ground_truth_*.csv` |
| Multiple abstracts were concatenated | DSpace `||` values and language-specific columns remain independent text units | `text_unit_statistics.csv`, prepared v3 Parquet |
| “No posee” was treated as content | Explicit absence markers are removed and counted | `text_unit_statistics.csv` |
| Multiple full texts were concatenated before truncation | Source files remain separate and share a fair handle-level character budget | segmented full-text v3 Parquet |
| Stopwords could be wrong inside multilingual items | Abstracts and full-text preprocessing windows are detected independently; keywords retain all words | prepared v3 Parquet |
| Frozen dense pooling compressed heterogeneous documents directly | Hierarchical pooling uses chunks → text unit → handle with normalized field weights | `embedding_coverage_*.csv` |
| Dense coverage described only the last call | Coverage is reported separately for train, calibration, and validation | `embedding_coverage_*.csv` |
| Fixed dense baselines alone cannot adapt to the task | DistilUSE- and LaBSE-backbone sequence classifiers jointly optimize the Transformer and a new 37-output head with BCE loss; this is not sentence-embedding fine-tuning | `finetuning/results_validation.csv` |
| Fine-tuning could be lost after a Colab interruption | Checkpoints are written every configured batch interval and every epoch | `finetuning/*/checkpoint.pt` |
| A fixed epoch count could underfit or retain an overfit final state | Validation is measured each epoch; training stops with patience and restores the best Macro-F1 checkpoint | `finetuning/*/epoch_history.csv`, `finetuning/*/best_checkpoint.pt` |
| A resumed run could recompute different splits after Drive availability changed | The usable cohort and four-way split are frozen once; every validation and fine-tuning checkpoint is bound to their SHA-256 fingerprint | `cohort_manifest.csv`, `split_manifest.csv`, `split_context.json` |
| Runtime projection counted only the last resumed session | Active elapsed time is accumulated across sessions | `timing_sessions.csv`, `timing.csv`, `scaling_estimate.csv` |
| A post-hoc transformer test could alter the winner | All candidates compete on validation; the global winner is frozen as the sole confirmatory result before any test metric is computed | `results_validation.csv`, `results_test.csv` |
| A single test result would not compare representation families | The best BoW, TF-IDF, BM25, fixed DistilUSE/LaBSE embedding, and DistilUSE-/LaBSE-backbone sequence-classifier candidates are selected by a pre-specified validation-only rule; they receive secondary test evaluation without changing the global winner | `selected_test_models_frozen.csv`, `results_test_comparative.csv`, `paired_bootstrap_vs_global.csv` |
| Items with more files/chunks could dominate training | Unit/chunk weights sum to one per handle | fine-tuning configuration and tests |
| Legacy caches silently mixed protocols | v3 uses dedicated segmented full-text, prepared-dataset, and embedding cache namespaces | notebook configuration |
| Label overlap was difficult to interpret | Per-label confusion counts, co-occurrence, rankings, bootstrap intervals, and English figures/reports remain mandatory | final run artifacts |

The historical ground truth is hybrid by design: rows where the human label
and historical LangID agreed are marked `human_langid_agreement`; manually
resolved disagreements are marked `manual_audit`. Results must be reported both
overall and with those provenance counts. Current LangID predictions drive
preprocessing; ground truth is used only for language evaluation.
