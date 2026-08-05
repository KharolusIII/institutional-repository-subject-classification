# Dense-model nomenclature

The repository preserves several `sbert*` identifiers because they are part of
completed-run CSVs, cache keys, checkpoints and configuration compatibility.
In this project, those identifiers refer specifically to
`sentence-transformers/distiluse-base-multilingual-cased-v1` (DistilUSE), not
to an unspecified SBERT model.

| Internal identifier | Scientific name | What is executed |
|---|---|---|
| `sbert`, `sbert_frozen` | fixed DistilUSE sentence embeddings | The complete Sentence-Transformers pipeline (DistilBERT, mean pooling and 768→512 Dense layer) remains fixed; overlapping chunk embeddings are pooled hierarchically and passed to a linear classifier. |
| `labse`, `labse_frozen` | fixed LaBSE sentence embeddings | The complete Sentence-Transformers pipeline (BERT, CLS pooling, Dense layer and normalization) remains fixed; overlapping chunk embeddings are pooled hierarchically and passed to a linear classifier. |
| `sbert_finetuned` | DistilUSE-backbone sequence classifier | `AutoModelForSequenceClassification` is initialized from the Transformer module distributed with the DistilUSE checkpoint. The Transformer and new 37-output classification head are jointly optimized. |
| `labse_finetuned` | LaBSE-backbone sequence classifier | `AutoModelForSequenceClassification` is initialized from the Transformer module distributed with the LaBSE checkpoint. The Transformer and new 37-output classification head are jointly optimized. |

The supervised route is **not sentence-embedding fine-tuning**: it does not
optimize the complete Sentence-Transformers pipeline with a contrastive,
triplet or sentence-similarity loss. It is end-to-end supervised multi-label
sequence-classification fine-tuning with class-weighted
`BCEWithLogitsLoss`. It uses non-overlapping chunks, retains at most eight
uniformly distributed chunks **per text unit**, and obtains document scores by
weighted aggregation of chunk logits.

Comparisons between the supervised sequence classifiers and fixed embedding
baselines are descriptive family-level comparisons, not controlled causal
estimates of the isolated effect of fine-tuning: the architecture, training
objective, chunk handling and validation-selected field set can differ.

