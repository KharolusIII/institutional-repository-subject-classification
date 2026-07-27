# Data

Real SEDICI metadata and full text are not distributed because they may be
large or subject to repository and privacy constraints.

Expected inputs:

- metadata CSV: one row per item (or rows aggregatable by handle);
- mapping CSV: Assetstore/bitstream ID and handle;
- full text: a directory of `<internal_id>.txt` files or Parquet shards with
  `file_id` and `text`;
- target fields: configured explicitly in YAML.

`data/sample/` contains a small synthetic corpus used only for tests and
demonstration. It does not represent real SEDICI records.

