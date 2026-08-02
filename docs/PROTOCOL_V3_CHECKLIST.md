# Post-run checklist: segmented and fine-tuned protocol v3

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
| SBERT and LaBSE were frozen | Both encoders receive supervised multi-label fine-tuning with BCE loss | `finetuning/results_validation.csv` |
| Fine-tuning could be lost after a Colab interruption | Checkpoints are written every configured batch interval and every epoch | `finetuning/*/checkpoint.pt` |
| A post-hoc transformer test would reuse test | All classical, frozen, and fine-tuned candidates compete on validation; only the global winner is evaluated on test | `results_validation.csv`, `results_test.csv` |
| Items with more files/chunks could dominate training | Unit/chunk weights sum to one per handle | fine-tuning configuration and tests |
| Legacy caches silently mixed protocols | v3 uses dedicated segmented full-text, prepared-dataset, and embedding cache namespaces | notebook configuration |
| Label overlap was difficult to interpret | Per-label confusion counts, co-occurrence, rankings, bootstrap intervals, and English figures/reports remain mandatory | final run artifacts |

The historical ground truth is hybrid by design: rows where the human label
and historical LangID agreed are marked `human_langid_agreement`; manually
resolved disagreements are marked `manual_audit`. Results must be reported both
overall and with those provenance counts. Current LangID predictions drive
preprocessing; ground truth is used only for language evaluation.
