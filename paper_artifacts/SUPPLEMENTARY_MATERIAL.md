# Supplementary material

## S1. Scope and public-data boundary

This supplement supports the reported 2026 SEDICI subject-classification
experiment. It contains aggregate statistics, frozen configurations, model
metrics and publication figures. It excludes repository texts, handles,
item-level predictions, split manifests, caches, model weights, credentials and
personal filesystem paths.

The executable notebook defaults to a synthetic corpus. Real repository data
are not required to test the software and are not redistributed.

## S2. Corpus construction

The analytical cohort contains 19,710 documents and 24,326 assignments over 37
subjects. It preserves 24,411 abstract segments and 20,750 full-text files;
4,365 items contain multiple abstract units and 463 contain multiple full-text
units. Eight explicit absence markers were removed, and 115 cohort items have
no usable abstract.

Source full text totals 1,212,509,366 characters. Fair per-item allocation under
the 400,000-character ceiling represents 1,011,823,166 characters (83.4487%).
The ceiling affects 510 documents (2.5875%).

Exact normalized/case-folded full-text duplicates are assigned one content
group before iterative multi-label splitting. The split fingerprint is
`d8a0b22f1fd9a1d54481c5b5227165c8ecb1dca5f9658ea7f7a83c77b3e3c24e`.

## S3. Language processing

Abstract and full-text units are detected independently with langid.py.
Detected content language drives sparse stopword removal. Declared metadata is
retained for agreement auditing but does not override the detector. Keywords
are normalized and retain all terms.

Declared/detected agreement is 14,940/15,166 (98.5098%) for abstracts and
18,897/19,675 (96.0457%) for full text. Against 23,609 matched historical
abstract segments, accuracy is 99.8687%; 23,055 reference decisions derive from
human/historical-detector agreement and 554 disagreements were manually
audited.

## S4. Model grid

The 233 primary validation candidates comprise:

- 189 sparse candidates: 3 preprocessing modes × 7 field combinations ×
  3 representations × 3 classifiers;
- 42 frozen dense candidates: 2 embedding models × 7 field combinations ×
  3 classifiers; and
- 2 fine-tuned transformer candidates.

Macro-F1 is the primary selection metric. The highest validation Macro-F1 in
the v3 grid is BM25/full-text/LinearSVC (`0.748348`). The highest validation
Micro-F1 is BM25/all-fields/logistic-regression (`0.777362`, Macro-F1
`0.738822`). The former advanced because Macro-F1 was prespecified.

## S5. Transformer adaptation

Frozen SBERT and LaBSE encode overlapping token chunks and aggregate them by a
length-weighted mean from chunks to text units and then to handles. Fine-tuning
uses `AutoModelForSequenceClassification`, non-overlapping chunks sampled
uniformly across each unit, at most eight chunks per unit, a length of 256,
class-weighted binary cross-entropy, AdamW at `2e-5`, batch 16, evaluation batch
64, gradient accumulation 4 and gradient checkpointing.

Both models selected epoch 5. SBERT validation Macro-F1 progresses from 0.5967
to 0.6663 monotonically. LaBSE progresses 0.6596, 0.6895, 0.7078, 0.7077 and
0.7148. Fine-tuned LaBSE is the strongest transformer family on test
(Macro-F1 0.7069; Micro-F1 0.7279).

## S6. Convergence and final evaluation

The focused audit evaluates BM25 and TF-IDF under four LinearSVC settings. Both
BM25 `C=1.0` fits fail to converge only for Humanidades. Both `C=0.5` fits
converge for every label and tie on stored validation metrics. The retained
stored setting is `tol=0.0005` without a claim of superiority over `0.0001`.

The final 2,959-item test estimate is:

| Metric | Value |
|---|---:|
| Subset accuracy | 0.586347 |
| Hamming loss | 0.016414 |
| Micro-F1 | 0.768458 |
| Macro-F1 | 0.752228 |
| Sample-F1 | 0.763147 |
| Precision@1 / Recall@1 | 0.827982 / 0.726203 |
| Precision@3 / Recall@3 | 0.380985 / 0.931959 |
| Precision@5 / Recall@5 | 0.237715 / 0.965979 |

The 95% bootstrap intervals are `[0.737224, 0.764313]` for Macro-F1 and
`[0.756687, 0.779731]` for Micro-F1.

## S7. Subject-level results

The strongest labels include Odontología (`0.9481`), Ciencias Informáticas
(`0.9420`) and Educación Física (`0.9396`). The lowest are Biología (`0.3009`),
Ciencias Sociales (`0.4590`) and Urbanismo (`0.5542`). Humanidades has test
support 227 and F1 `0.6571`, illustrating that support alone does not explain
difficulty.

See `tables/per_label_test.csv`, `tables/multilabel_confusion_matrices.csv`, and
the per-label figures for the complete audit.

## S8. Artifact map

| Manuscript claim | Public evidence |
|---|---|
| Corpus and split | `runs/v3_main/dataset_statistics.csv`, `text_unit_statistics.csv`, `fulltext_coverage.csv`, `label_coverage_by_split.csv` |
| Language audit | `runs/v3_main/language_agreement_summary.csv`, `abstract_language_ground_truth_summary.csv`, language distributions |
| Complete 233-candidate validation comparison | `runs/v3_main/results_validation_grid.csv` |
| Fine-tuning | `tables/transformer_validation.csv`, `sbert_epoch_history.csv`, `labse_epoch_history.csv` |
| Family results | `tables/family_test_comparison.csv` |
| Convergence audit | `runs/convergence_audit/results_validation.csv`, `tables/classifier_convergence.csv` |
| Final metrics and CIs | `tables/final_test_metrics.csv`, `runs/convergence_audit/bootstrap_ci.csv` |
| Paired differences | `tables/paired_bootstrap_updated.csv` |
| Subject behavior | `tables/per_label_test.csv`, confusion/error tables and figures |

## S9. Reproduction

Run the public notebook with its default synthetic profile for an end-to-end
functional check. To reproduce the paper experiment with authorized data, use
`configs/run_full_20k_v3_2026.yaml`, followed by
`configs/run_convergence_audit_20k_2026.yaml`, retaining the same cohort and
split fingerprint. Outputs are resumable and bound to their split context.
