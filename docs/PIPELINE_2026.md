# Pipeline 2026

## Methodological contract

Train fits representations and model parameters. Calibration estimates
decision thresholds independently for each candidate. Validation selects
preprocessing, representation, classifier, hyperparameters, pooling and the
supervised-training epoch. Test remains isolated until all selection decisions
are frozen.

## Historical versus 2026

| Component | 2025 | 2026 | Reason |
|---|---|---|---|
| Code organization | Stateful Colab cells with duplicated generations | Installable modules plus thin notebook | Testing and reuse |
| Paths | Personal Drive paths | YAML and five notebook variables | Portability |
| Runs | Parameter-derived directory could overwrite | Unique UTC `run_id` | Traceability |
| Target schema | Four fields merged | Explicit fields plus source/support report | Auditability |
| Language metadata | Not systematically verified | Declared and detected stored separately | DSpace suffixes can be wrong |
| Field language | Not detected | Abstract/fulltext independent; keywords best-effort | Mixed-language records |
| Stopwords | No systematic language-aware ablation | Detected-language-specific, sparse only | Fair empirical comparison |
| Text normalization | Whitespace joining | NFKC and technical cleanup; accents preserved | Robustness |
| Sparse features | TF-IDF and BM25 | BoW, TF-IDF, BM25 and word/char studies | Stronger baselines |
| BM25 | Custom formula | Exact `legacy` variant retained | Reproducibility |
| Embeddings | Direct encoding / implicit truncation | `legacy_truncated` and chunked pooling | Long documents |
| Embedding cache | None | Content/config keyed cache | Resume expensive runs |
| Estimators | Instances reused | Factory plus `sklearn.clone` | Isolation |
| Randomness | Partial | Central seed | Repeatability |
| Metrics | Accuracy, micro-F1, macro-F1 | Named subset accuracy, multilabel and ranking metrics | Cataloger-oriented evaluation |
| Thresholds | Estimator defaults | Global/per-label on dedicated calibration | Actionable suggestions without validation leakage |
| Test | Split created, not evaluated | Frozen final evaluation | Unbiased estimate |
| Timing | Classifier fit + predict called `time_sec` | Stage-separated timing | Honest resource accounting |
| Artifacts | CSVs on Drive | Manifests, statistics, metrics, predictions | Reproducibility/integration |

## Language-aware sparse flow

Each field is handled independently:

```text
abstract → detect language → normalize → field-language stopwords
keywords → best-effort detect → normalize → stopwords only if reliable
fulltext → detect language → normalize → field-language stopwords
                                      ↓
                            concatenate selected fields
                                      ↓
                              BoW / TF-IDF / BM25
```

If language is `und`, stopwords are not removed. Character n-grams must not
receive stopword removal.

## Dense and supervised Transformer flows

Transformer text receives NFKC, whitespace normalization, and removal of
unambiguous technical corruption only. It does not receive stopword removal,
stemming, or mandatory lemmatization.

The fixed embedding route loads the complete DistilUSE and LaBSE
Sentence-Transformers pipelines without parameter updates. `legacy_truncated`
reproduces direct `SentenceTransformer.encode`. Chunked
modes query `model.max_seq_length`, tokenize, create configurable overlaps,
embed chunks, and apply mean, length-weighted mean, or optional max pooling.
Pooling is selected on validation, never test.

The supervised route is architecturally distinct. It initializes
`AutoModelForSequenceClassification` from the Transformer module distributed
with each checkpoint, creates a new 37-output head, and jointly optimizes the
backbone and head using class-weighted `BCEWithLogitsLoss`. It uses up to eight
uniformly distributed, non-overlapping chunks per text unit and aggregates
chunk logits to document scores. No contrastive sentence-embedding objective
is used. Historical `sbert*` identifiers denote DistilUSE; see
[`MODEL_NOMENCLATURE.md`](MODEL_NOMENCLATURE.md).

## Current execution boundary

The CLI automatically executes sparse experiments. Dense infrastructure,
chunking, pooling, and caching are implemented in `embeddings.py` and can be
used from the Colab dense section. Dense grids are intentionally not run by
default because they download large models and are costly.

## Execution scale

Scale is a configuration choice, not a hardcoded limitation:

- `smoke`: at most 1,000 eligible items, for debugging and input validation;
- `20k`: the historical operational scale, retained for comparable E0/E1 runs;
- `full`: all eligible handles after target filtering.

All three use the same leakage controls. A full-corpus run is scientifically
preferable for the final model when resources permit, but it does not replace
the 20k reproduction: dataset size itself changes the experiment. Dense
full-corpus runs should follow a successful sparse full-corpus run because
embedding and cache requirements can be substantial.

## Operational logging

`run.log` is created before ingestion and receives console messages, warnings,
stage transitions, dataset/split counts, validation results, the frozen
configuration, final test metrics, completion time, and failure tracebacks.
It lives beside the resolved YAML, environment, commit, artifacts, and
stage-separated `timing.csv`.
