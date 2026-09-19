# Exposure-parity validation report

## Decision for the present revision

The existing experiment is sufficient to answer the reviewer's *content
exposure* objection as a clearly labelled post-hoc robustness analysis. It is
not necessary to add the hierarchical long-document experiment to the current
paper. That experiment asks a different question—whether increasing and
learning the aggregation of long-document evidence improves LaBSE—and is better
reserved for a subsequent publication.

The control does not make BM25 and LaBSE algorithmically identical. It isolates
the point raised by the reviewer: both routes receive evidence from exactly the
same raw source spans. Model-specific tokenization, normalization and inductive
biases necessarily remain different and are stated as such.

## Paste-ready response to the reviewer

> We thank the reviewer and agree that the original comparison did not isolate
> model family from document-content exposure. We therefore added a post-hoc,
> validation-only exposure-parity analysis using the 16,751 non-test records
> from the unchanged 19,710-document, 37-label cohort and frozen four-way
> partition. The LaBSE fast tokenizer first
> defines exact raw-source offsets for up to eight uniformly spaced,
> non-overlapping 256-token sequences (254 content tokens plus [CLS]/[SEP]) per
> original full-text unit. BM25 is then fitted only on those same raw character
> spans, after the established language-aware sparse preprocessing; it receives
> no lexical content outside the Transformer-selected spans. The frozen LangID
> assignments for the original preprocessing windows are reused only to choose
> BM25 stopword lists. Items containing several full-text files retain the
> historical per-unit allocation, so they may contain more than eight chunks in
> total. The LaBSE backbone and its new 37-label head are fine-tuned end-to-end
> for up to 20 epochs. Decision thresholds are selected exclusively on the
> frozen threshold-selection partition (named `calibration` in the
> implementation); no probability or score calibration is performed.
> Checkpoints are selected by validation Macro-F1 under an
> early-stopping rule frozen before the three reported runs (patience=2,
> minimum delta=0.002), and LaBSE is repeated with seeds 13, 42 and 73. No test
> inference or evaluation was performed in this follow-up; historical test
> results from the earlier study had already been inspected. On validation,
> BM25 obtained Macro-F1
> 0.757330 and Micro-F1 0.770086; exposure-matched
> LaBSE obtained Macro-F1 0.739794 ± 0.001979
> and Micro-F1 0.753722 ± 0.001700 across seeds.
> BM25 led in all three runs, by 0.017535 Macro-F1
> and 0.016364 Micro-F1 on average. Thus, the
> BM25-versus-LaBSE ranking persists when raw content exposure is held constant,
> although we explicitly limit the conclusion to these two routes, this corpus,
> this split and this post-hoc validation control. The result does not establish
> exposure parity for SBERT or for every Transformer configuration.

## Paste-ready paragraph for the paper

> **Post-hoc exposure-parity analysis.** To examine whether the sparse model's
> advantage was attributable to access to more document content, we constructed
> a validation-only matched-exposure control on the frozen cohort and split.
> Exact raw-source spans were selected through the pinned LaBSE fast tokenizer
> (up to eight uniformly spaced, non-overlapping sequences of 256 model tokens,
> including two special tokens, per original full-text unit). The LaBSE
> sequence classifier consumed those token sequences, while BM25 consumed only
> the corresponding raw character spans after its model-appropriate,
> language-aware preprocessing. Frozen LangID assignments derived for the
> original full-text preprocessing windows selected BM25 stopword lists, but no
> unselected lexical content entered its feature matrix. Across the
> 16,751 training, threshold-selection and validation items,
> the common spans represented
> 136,758,120 of 866,897,087
> capped source characters (15.78%).
> With decision-threshold fitting restricted to the threshold-selection
> partition (named `calibration` in the implementation) and validation-only
> checkpoint selection, BM25 achieved Macro-F1 0.757330 and Micro-F1
> 0.770086. Across three LaBSE fine-tuning seeds, the corresponding
> values were 0.739794 ± 0.001979 and
> 0.753722 ± 0.001700, respectively. BM25 led
> for every seed. These findings show that the observed ranking is not explained
> solely by unequal raw-content exposure; they do not establish equivalence of
> the models' internal representations or preprocessing. No test inference or
> evaluation was performed for this post-hoc control.

## Compact table for the paper

| Seed | Selected LaBSE epoch | BM25 Macro-F1 | BM25 Micro-F1 | LaBSE Macro-F1 | LaBSE Micro-F1 | BM25 − LaBSE Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | 14 | 0.757330 | 0.770086 | 0.741890 | 0.755102 | 0.015440 |
| 42 | 9 | 0.757330 | 0.770086 | 0.739537 | 0.751823 | 0.017793 |
| 73 | 9 | 0.757330 | 0.770086 | 0.737956 | 0.754240 | 0.019373 |
| **Mean** | — | **0.757330** | **0.770086** | **0.739794** | **0.753722** | **0.017535** |

Values are validation results. The `±` values reported in the prose are sample
standard deviations across LaBSE seeds. The thresholded BM25 F1 values are
identical in all three run contexts; BM25 was refitted in each context as an
integrity control. No inferential test is claimed from three seeds on one fixed
split.

## Protocol and interpretation audit

- Frozen split: 12,836 train, 1,927 threshold selection (implementation name:
  `calibration`), 1,988 validation and 2,959 test; all 37 labels remain in the
  fixed schema.
- Exposure allocation: historical `per_unit`; a maximum of eight sequences per
  source full-text unit, not a global maximum of eight per repository item.
- Chunk geometry: 256 model tokens, 254 content tokens, no overlap, uniformly
  spaced over each unit's available windows.
- LaBSE: end-to-end fine-tuning with a 37-output chunk classifier and weighted
  mean aggregation. Maximum 20 epochs; early stopping uses Macro-F1, patience 2
  and minimum delta 0.002.
- BM25: the same raw spans only, language-aware stopword preprocessing, train-only
  vocabulary, 100,000 maximum features, and one-vs-rest Linear SVC.
- Language side information: frozen LangID assignments from the original
  full-text preprocessing windows select BM25 stopword lists; language is not
  redetected from short spans, and no unselected lexical content enters BM25.
- Decision thresholds: selected only on the dedicated threshold-selection
  partition (internally named `calibration`). Epochs: selected only on validation.
  No test inference or evaluation was performed in the new analysis; historical
  test results from the earlier study had already been inspected.
- Historical execution source revision: `afb1df90fe967e41386878d8f28b27a2f7ff15f5`; pinned LaBSE revision:
  `836121a0533e5664b21c7aacc5d22951f2b8b25b`.

## Scope limitation

This analysis supports a narrow and useful conclusion: within the frozen
validation protocol, BM25's lead over the tested LaBSE route persists when both
models are restricted to identical raw-source spans. It does not establish the
same result for SBERT or all Transformer configurations, nor determine whether
a hierarchical Transformer exposed to substantially more text would outperform
either matched route. The latter is the planned follow-up experiment and should
not be implied by the present table.
