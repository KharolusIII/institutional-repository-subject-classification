# Final evaluation report

## Selected configuration

preprocessing=raw, feature_set=abstract, representation=bow, classifier=sgd

## Overall test metrics

| Metric | Value |
|---|---:|
| Macro F1 | 0.8434 |
| Micro F1 | 0.8261 |
| Micro precision | 0.7917 |
| Micro recall | 0.8636 |
| Subset accuracy | 0.5789 |
| Hamming loss | 0.1053 |

## Best classified subject labels

| Subject label | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Ciencias Informáticas | 6 | 1.000 | 0.833 | 0.909 |
| Química | 6 | 1.000 | 0.833 | 0.909 |
| Historia | 5 | 1.000 | 0.800 | 0.889 |
| Educación | 5 | 0.500 | 1.000 | 0.667 |

## Most difficult subject labels

| Subject label | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Educación | 5 | 0.500 | 1.000 | 0.667 |
| Historia | 5 | 1.000 | 0.800 | 0.889 |
| Ciencias Informáticas | 6 | 1.000 | 0.833 | 0.909 |
| Química | 6 | 1.000 | 0.833 | 0.909 |

## Interpretation note

Per-label results should be interpreted together with test support. Labels with
very small support have high metric uncertainty.

## Pre-specified secondary representation-family comparison
These test results are secondary analyses. The global confirmatory winner was frozen from validation before any test metric was computed and is not replaced by this ranking.
| Role | Family | Representation | Classifier | Validation Macro F1 | Test Macro F1 | Test Micro F1 |
|---|---|---|---|---:|---:|---:|
| family_secondary | tfidf | tfidf | linear_svc | 0.9222 | 0.9268 | 0.9268 |
| family_secondary | bm25 | bm25 | logreg | 0.9222 | 0.9268 | 0.9268 |
| family_secondary|global_confirmatory | bow | bow | sgd | 0.9530 | 0.8434 | 0.8261 |
