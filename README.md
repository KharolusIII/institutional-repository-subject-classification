# Institutional Repository Subject Classification

[![Tests](https://github.com/KharolusIII/institutional-repository-subject-classification/actions/workflows/tests.yml/badge.svg)](https://github.com/KharolusIII/institutional-repository-subject-classification/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KharolusIII/institutional-repository-subject-classification/blob/main/notebooks/01_sedici_subject_classification_colab.ipynb)

Reproducible multi-label subject classification for institutional repositories,
developed around SEDICI (Universidad Nacional de La Plata). The system suggests
subjects to catalogers; it does not replace professional cataloging decisions.

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
  → validation selection and thresholds
  → isolated final test
  → versioned reports and predictions
```

The historical material is preserved unchanged in `Pipe 2025 OR/`. Its audit
is in [`docs/LEGACY_2025_AUDIT.md`](docs/LEGACY_2025_AUDIT.md).

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

Open the badge above. At the beginning change only:

- `BASE_DIR`
- `META_CSV`
- `MAP_CSV`
- `FULLTEXT_SOURCE`
- `CONFIG_FILE`
- `EXECUTION_PROFILE`

The notebook clones and installs the package, mounts Drive, validates inputs,
runs the pipeline, and displays the main artifacts. For a first check, enable
`USE_DUMMY_DATA=True`; no Drive data are then required.

### Execution scale profiles

The same scientific code can be run with four explicit profiles:

| Profile | Configuration | Intended use |
|---|---|---|
| `smoke` | `configs/run_smoke_2026.yaml` | Up to 1,000 items; input/schema and fast Colab checks |
| `full-smoke` | `configs/run_full_smoke_1k_2026.yaml` | Up to 1,000 items; sparse plus SBERT and LaBSE timing |
| `20k` | `configs/run_20k_2026.yaml` | Historical-scale comparison and E0/E1 continuity |
| `full` | `configs/run_full_corpus_2026.yaml` | Every eligible item; resumable sparse plus dense execution |

Validation combinations, downloaded Drive texts, and embeddings are
checkpointed. An interrupted run keeps a `.incomplete` marker and is resumed
on the next execution instead of starting over.

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

### SEDICI Colab paths

The notebook defaults intentionally separate the new repository workspace from
the historical, read-only inputs:

```text
Repository/workspace:
MyDrive/A___Maestria_en_ID/Tareas_PLN/
  001_1_Clasificador_Materias_SEDICI_Texto_Completo/

Historical inputs:
MyDrive/A___Maestria_en_ID/Tareas_PLN/100_datos_thesis_maestria/
  SEDICIpoblacion.csv
  Mapeo_SEDICI_Rafa_data-1758643353469.csv
  Datos_SEDICI/SEDICI_FullText_TXT/
```

The Git checkout is created on fast, temporary Colab storage at
`/content/institutional-repository-subject-classification`. Outputs and the
Google Drive file index cache are persisted under the exact `BASE_DIR`.
Historical metadata, mapping, and TXT files are read from their existing
locations and are never moved or written by the pipeline.

The TXT directory contains too many entries for reliable enumeration through
the mounted Drive filesystem. Real-data profiles therefore authenticate with
Google Drive API, list file metadata once, cache it as
`cache/drive_txt_index.csv`, map IDs to handles, and download only the
full texts selected by the current execution profile.

For GitHub access, create a private Colab secret named:

```text
GITHUB_TOKEN_IR_SUBJECT_CLASSIFICATION
```

Enable notebook access for that secret. The notebook retrieves it with
`google.colab.userdata`, uses a temporary `GIT_ASKPASS` helper for clone/pull,
and removes the helper and token from its temporary environment immediately
afterward. The token is never embedded in `REPOSITORY_URL`, Git remotes,
configuration files, or run logs.

## Input format

The metadata CSV needs a handle or URI, configured target columns, abstract and
keyword columns. Full text can be:

- an inline `fulltext` column (useful for testing);
- a directory of `<internal_id>.txt` plus a mapping CSV;
- Parquet shards through the ingestion API.

Targets are never inferred silently. The source columns are explicit in YAML
and `target_schema_report.csv` records label, source field, and support.

## Language policy

DSpace suffixes such as `[es]` are audit metadata, not ground truth.
Abstract and full text are detected independently. Keyword detection is
best-effort and becomes `und` when evidence is insufficient.

Language-specific stopword removal is available only for BoW, TF-IDF, and
BM25, and is applied per field before concatenation. Lists come from
[Stopwords ISO](https://github.com/stopwords-iso/stopwords-iso) through the
pinned `stopwordsiso` package and use ISO 639-1 language codes. Unsupported or
undetermined languages remain unchanged. Transformers retain natural text and
receive only Unicode/whitespace/obvious-noise cleanup.
Declared-versus-detected comparisons are called **agreement**, never accuracy.

Dense long-document experiments use a length-weighted mean over all
overlapping chunks inside the configured character ceiling. Full profiles use
up to 200,000 characters and write `fulltext_coverage.csv` plus
`embedding_coverage_*.csv` artifacts for auditing.

## Historical baseline and 2026 experiments

`configs/baseline_2025.yaml` captures the audited E0 behavior: 37 labels,
TF-IDF/BM25/SBERT/LaBSE, seven feature sets, and One-vs-Rest LogReg,
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
selected on validation. Test is evaluated only after the configuration is
frozen.

## Privacy and data distribution

Real metadata/full text are not committed. `.gitignore` excludes private data,
credentials, model caches, embeddings, and generated runs. Review repository
and copyright policies before processing or sharing exported content.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The software
is released under the [MIT License](LICENSE).
