# Institutional Repository Subject Classification

[![Tests](https://github.com/KharolusIII/institutional-repository-subject-classification/actions/workflows/tests.yml/badge.svg)](https://github.com/KharolusIII/institutional-repository-subject-classification/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KharolusIII/institutional-repository-subject-classification/blob/main/notebooks/01_public_reproducibility_colab.ipynb)

Reproducible multi-label subject classification for institutional repositories,
developed around SEDICI (Universidad Nacional de La Plata). The system suggests
subjects to catalogers; it does not replace professional cataloging decisions.

## Reviewer quick access

- [Supplementary material (PDF)](paper_artifacts/CACIC_2026_Supplementary_Material.pdf)
- [Supplementary material (DOCX)](paper_artifacts/CACIC_2026_Supplementary_Material.docx)
- [Complete 233-configuration validation table](paper_artifacts/runs/v3_main/results_validation_grid.csv)
- [Final held-out metrics](paper_artifacts/tables/final_test_metrics.csv)
- [Complete per-subject results](paper_artifacts/tables/per_label_test.csv)
- [Runs and provenance](paper_artifacts/RUNS_AND_PROVENANCE.md)
- [Exposure-parity reviewer response and validation evidence](paper_artifacts/reviewer_response/exposure_parity/REPORT.md)
- [Downloadable exposure-parity artifact package](paper_artifacts/packages/exposure_parity_validation_artifacts_2026-09-15.zip)

The public evidence package contains aggregate results only. Restricted texts,
document identifiers, item-level predictions, private notebooks, credentials,
and local paths are not distributed.

## What the pipeline does

It links DSpace metadata, Assetstore IDs, handles, and full text; measures
coverage before expensive reads; constructs a multi-label dataset; detects
language from content; creates iterative stratified splits; compares sparse and
dense representations; selects configurations on validation; and evaluates a
frozen configuration on test.

Scientific logic lives in `src/ir_subject_classification/`. The Colab notebook
is an interface to the same package.

## Architecture

```text
metadata + mapping + TXT/Parquet
  → handle-level dataset
  → language identification per field
  → label selection and iterative split
  → sparse preprocessing OR natural transformer text
  → representation + One-vs-Rest classifier
  → decision-threshold selection on the dedicated threshold-selection split
  → configuration and checkpoint selection on validation
  → isolated final test
  → versioned reports and predictions
```

The current package is the maintained implementation. Historical experiments
are documented in [`docs/LEGACY_2025_AUDIT.md`](docs/LEGACY_2025_AUDIT.md),
but are not required to run or evaluate the 2026 pipeline.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
pytest
```

For every optional backend:

```bash
python -m pip install -e .[all]
```

## Smoke test with synthetic data

```bash
python scripts/generate_dummy_data.py
python scripts/run_pipeline.py --config configs/dummy.yaml
```

The generated data are synthetic and contain no repository records.

## Google Colab

The public notebook defaults to synthetic data, anonymous Git access, temporary
Colab storage, and no Google Drive mount. To use an authorized corpus, set
`USE_DUMMY_DATA=False` and change:

- `BASE_DIR`
- `META_CSV`
- `MAP_CSV`
- `FULLTEXT_SOURCE`
- `CONFIG_FILE`
- `EXECUTION_PROFILE`

The notebook clones and installs the package, validates inputs, runs the
pipeline, and displays the main artifacts. Drive mounting is opt-in through
`USE_GOOGLE_DRIVE=True`. With the public defaults, no private input data or
credentials are required.

## Public repository layout

- `src/ir_subject_classification/`: maintained pipeline implementation.
- `notebooks/01_public_reproducibility_colab.ipynb`: anonymous, executable
  Colab interface with generic paths and synthetic-data defaults.
- `examples/input/`: synthetic input illustrating the public schema.
- `examples/output/`: curated outputs from a successful synthetic smoke run.
- `paper_artifacts/runs/v3_main/`: aggregate evidence from the main paper run.
- `paper_artifacts/runs/convergence_audit/`: aggregate convergence-audit and
  final-test evidence.
- `paper_artifacts/figures/` and `paper_artifacts/tables/`: publication-ready
  figures and compact reviewer tables.

Private notebooks, authorized corpora, item-level predictions, caches,
checkpoints, local paths, and credentials are intentionally excluded.

### Final reproducibility configurations

These are the maintained configurations for reproducing or extending the final
2026 protocol:

| Profile | Configuration | Intended use |
|---|---|---|
| Synthetic smoke | `configs/dummy.yaml` | Fast public end-to-end check without external data |
| Protocol smoke v3 | `configs/run_protocol_smoke_1k_v3_2026.yaml` | Segmentation, decision-threshold selection and supervised sequence-classification acceptance test on approximately 1,000 items |
| Paper experiment v3 | `configs/run_full_20k_v3_2026.yaml` | Frozen 20k paper cohort; 233 validation candidates and family-level tests |
| Convergence audit | `configs/run_convergence_audit_20k_2026.yaml` | Eight controlled BM25/TF-IDF LinearSVC fits and confirmatory held-out test |
| Full-corpus v3 | `configs/run_full_corpus_v3_2026.yaml` | Final segmented protocol with a dedicated threshold-selection partition over every eligible item |

The paper results come from **Paper experiment v3**, followed by the
**Convergence audit** using the same frozen cohort and split fingerprint.

### Historical and scale-control profiles

The following profiles remain available for continuity, timing studies and
earlier ablations. Their results must not be mixed with the final v3 protocol.

| Profile | Configuration | Intended use |
|---|---|---|
| `smoke` | `configs/run_smoke_2026.yaml` | Up to 1,000 items; input/schema and fast Colab checks |
| `full-smoke` | `configs/run_full_smoke_1k_2026.yaml` | Up to 1,000 items; sparse plus fixed DistilUSE and LaBSE embedding timing |
| `20k` | `configs/run_20k_2026.yaml` | Historical-scale comparison and E0/E1 continuity |
| `full-20k` | `configs/run_full_20k_2026.yaml` | All 231 sparse and dense combinations on a stratified 20,000-item sample |
| `full-20k-v2` | `configs/run_full_20k_v2_2026.yaml` | Calibrated 20k protocol plus resumable DistilUSE- and LaBSE-backbone sequence classifiers |
| `full-final` | `configs/run_full_final_2026.yaml` | All 231 sparse and dense combinations on every eligible item and 37 labels |
| `full` | `configs/run_full_corpus_2026.yaml` | Every eligible item; resumable sparse plus dense execution |

Validation combinations, downloaded Drive texts, and embeddings are
checkpointed. An interrupted run keeps a `RUN_INCOMPLETE` marker and is resumed
on the next execution instead of starting over.

Every run now freezes `cohort_manifest.csv` and `split_manifest.csv` before any
model is selected. Resumed sessions restore those exact handles and assignments
instead of recomputing a split from newly available files. Validation and
fine-tuning checkpoints are bound to `split_context.json` by a SHA-256
fingerprint; incompatible legacy artifacts fail closed and must never be mixed
with a new test partition. Active session time is accumulated in
`timing_sessions.csv`, so scaling estimates include resumed work rather than
only the final Colab session.

### Focused convergence audit

Before reporting the final sparse winner, run
`configs/run_convergence_audit_20k_2026.yaml`. This profile reuses the
segmented 20k dataset, skips transformer work, and compares BM25 and TF-IDF
with four controlled Linear SVC settings. Selection remains isolated to
validation and only the resulting global winner is evaluated on test.
`classifier_convergence.csv` records iterations and convergence separately for
every subject label and fit phase. The audit has completed; aggregate evidence
is available under [`paper_artifacts/`](paper_artifacts/).

The `full-final` profile can be executed as five resumable Colab stages by
setting `EXECUTION_STAGE` in the notebook:

1. `prepare` downloads and consolidates the selected full text, detects
   languages, fixes the split, and writes compressed Parquet datasets;
2. `sparse` evaluates BoW, TF-IDF, and BM25;
3. `sbert` creates/reuses fixed, chunked DistilUSE vectors and evaluates all linear classifiers;
4. `labse` does the same for fixed LaBSE sentence embeddings;
5. `finalize` selects the validation winner and performs the isolated test
   evaluation.

`all` executes the same phases in one invocation. For separate Colab sessions,
use the five named stages in order. The validation CSV is updated atomically
after every completed model combination. Full text and prepared records are
stored as Zstandard-compressed Parquet, while dense vectors use atomic,
batch-sharded NumPy caches keyed by model, pooling configuration, and document
contents. A disconnection can therefore lose at most the currently encoded
document batch rather than an entire representation.

The `full-20k-v2` profile also trains supervised sequence classifiers
initialized from the Transformer modules distributed with the DistilUSE and
LaBSE checkpoints. It uses binary cross-entropy over uniformly sampled chunks
from each text unit. It
checkpoints every 500 training batches and after every epoch. Model selection
uses validation, while per-label decision thresholds use a separate
threshold-selection split (named `calibration` in the implementation); no
probability or score calibration is performed. The global validation winner is
frozen before any test result is computed.
Protocol v3 trains for at most five epochs, evaluates after every epoch, stops
after two epochs without a material Macro-F1 improvement, and restores the
best-validation checkpoint. The 1k smoke intentionally remains a one-epoch
functional test.

The global validation winner remains the single confirmatory result. Protocol
v3 also pre-specifies the best-validation selection rule for each representation family
(BoW, TF-IDF, BM25, fixed DistilUSE/LaBSE embeddings, and DistilUSE-/LaBSE-
backbone sequence classifiers) for a
secondary test comparison. Those results cannot replace the global winner and
include paired bootstrap differences on the same test documents.

Protocol v3 preserves every abstract value and every mapped full-text file as
an independent text unit. Sparse preprocessing detects language per abstract
and per full-text window. Fixed dense embeddings and supervised sequence
classifiers aggregate hierarchically
from chunks to text units and then to handles, with field weights normalized so
that an item with more files does not receive more total weight. Explicit
absence markers such as `No posee` are removed and audited. See
[`docs/PROTOCOL_V3_CHECKLIST.md`](docs/PROTOCOL_V3_CHECKLIST.md).

### Dense-model terminology

Historical identifiers such as `sbert`, `sbert_frozen` and
`sbert_finetuned` are retained in configurations, caches and result tables for
reproducibility. They refer specifically to DistilUSE
(`distiluse-base-multilingual-cased-v1`). The fixed route uses the complete
Sentence-Transformers embedding pipelines. The supervised route instead
initializes `AutoModelForSequenceClassification` from the corresponding
Transformer module, creates a 37-output head, and jointly updates backbone and
head with class-weighted binary cross-entropy. It is therefore supervised
sequence-classification fine-tuning, not contrastive sentence-embedding
fine-tuning. See [`docs/MODEL_NOMENCLATURE.md`](docs/MODEL_NOMENCLATURE.md).

The 20k profile is retained as an experimental control, not as a permanent
limit. Results across scales must be reported separately because their
training distributions and statistical power differ. Start with `smoke`, then
`20k`, and only then `full` after checking coverage, runtime, RAM, and storage.

### Logs

Every run writes `run.log` inside its unique output directory and mirrors the
same messages to the console. It records the resolved profile, corpus counts,
language distributions, split coverage, every validation experiment, the
frozen winner, test metrics, elapsed time, Python warnings, and full tracebacks
on failure. `config_resolved.yaml`, `environment.txt`, `git_commit.txt`, and
`timing.csv` provide the surrounding reproducibility context.

The final Colab cell prints the last 80 lines and includes an optional
`files.download(...)` call for downloading the complete log.

### Public Colab paths

The notebook uses generic temporary paths by default:

```text
Workspace:
/content/ir_subject_classification_workspace/

User-provided inputs:
/content/ir_subject_classification_data/
  metadata.csv
  fulltext_mapping.csv
  fulltext_txt/
```

The Git checkout is created on temporary Colab storage at
`/content/institutional-repository-subject-classification`. Outputs and caches
are written under `BASE_DIR`. Users who explicitly enable Drive may point these
generic variables to their own authorized Drive locations.
Input metadata, mapping, and TXT files are read in place and are never moved or
written by the pipeline.

The TXT directory contains too many entries for reliable enumeration through
the mounted Drive filesystem. Real-data profiles therefore authenticate with
Google Drive API, list file metadata once, cache it as
`cache/drive_txt_index.csv`, map IDs to handles, and download only the
full texts selected by the current execution profile.

The public notebook clones this public repository anonymously. It contains no
token lookup, secret name, credential helper, or authenticated Git remote.
Private working notebooks must be stored outside the repository and must never
be committed.

## Input format

The metadata CSV needs a handle or URI, configured target columns, abstract and
keyword columns. Full text can be:

- an inline `fulltext` column (useful for testing);
- a directory of `<internal_id>.txt` plus a mapping CSV;
- Parquet shards through the ingestion API.

Targets are never inferred silently. The source columns are explicit in YAML
and `target_schema_report.csv` records label, source field, and support.

## Language policy

DSpace suffixes such as `[es]` identify the language of the metadata field,
not necessarily the document. Values stored in `dc.language*` and legacy
`sedici2003.idioma[es]` are retained as declared document languages for audit,
not treated as ground truth. LangID content detection drives preprocessing;
abstract and full text are detected independently. Keyword detection is
best-effort and becomes `und` when evidence is insufficient.

Language-specific stopword removal is available only for BoW, TF-IDF, and
BM25, and is applied independently to abstracts and full text before
concatenation. Keywords are normalized but never stopword-filtered. Lists come from
[Stopwords ISO](https://github.com/stopwords-iso/stopwords-iso) through the
pinned `stopwordsiso` package and use ISO 639-1 language codes. Unsupported or
undetermined languages remain unchanged. Transformers retain natural text and
receive only Unicode/whitespace/obvious-noise cleanup.
Declared-versus-detected comparisons are called **agreement**, never accuracy.

Dense long-document experiments use a length-weighted mean over all
overlapping chunks inside the configured character ceiling. Full profiles use
up to 200,000 characters and write `fulltext_coverage.csv` plus
`embedding_coverage_*.csv` artifacts for auditing.

Final runs also produce an English `evaluation_report.md`, a subject
performance ranking, per-label multilabel confusion matrices, and
publication-ready figures. Original subject label names are never translated.

## Historical baseline and 2026 experiments

`configs/baseline_2025.yaml` captures the audited E0 behavior: 37 labels,
TF-IDF/BM25/DistilUSE/LaBSE, seven feature sets, and One-vs-Rest LogReg,
LinearSVC, and SGD. Historical values are reference evidence, not new results.

The planned sequence is:

- E0: historical reproduction;
- E1: sparse preprocessing ablation;
- E2: word/character n-grams;
- E3: legacy-truncated versus chunked embeddings;
- E4: focused tuning;
- E5: validation-only threshold optimization and final test;
- E6: Top 37/60/100 and long-tail studies.

See [`docs/PIPELINE_2026.md`](docs/PIPELINE_2026.md) and [`PLAN.md`](PLAN.md).

## Outputs and reproducibility

Every run creates `outputs/<UTC timestamp>_<experiment>/` and never overwrites
another run. Depending on the experiment it includes resolved configuration,
environment, Git commit, statistics, language reports, split coverage,
validation/test metrics, thresholds, timings, per-label results, bootstrap
confidence intervals, and ranked test predictions.

Vocabulary, IDF, and BM25 are fitted on train only. Representation,
preprocessing, classifier, pooling, hyperparameters, and thresholds are
selected on validation. Test is evaluated only after the global winner and
the pre-registered secondary family representatives are frozen.

## Paper artifacts

[`paper_artifacts/`](paper_artifacts/) contains the compact public evidence
package: final and family-level metrics, validation leaders, per-subject
results, language agreement, transformer epoch histories, convergence records,
paired-bootstrap comparisons, and publication-ready figures. These are
aggregate outputs only. Item-level predictions, handles, texts, manifests,
caches, and model weights are deliberately excluded.

The [complete validation grid](paper_artifacts/runs/v3_main/results_validation_grid.csv)
contains all 233 candidates with preprocessing, fields, representation,
classifier, selected decision thresholds, multilabel metrics and transformer training
metadata. It is the machine-readable scientific export of the resumable
validation checkpoint.

## Privacy and data distribution

Real metadata/full text are not committed. `.gitignore` excludes private data,
credentials, model caches, embeddings, and generated runs. Review repository
and copyright policies before processing or sharing exported content.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The software
is released under the [MIT License](LICENSE).
