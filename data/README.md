# Data

Real SEDICI metadata and full text are not distributed because they may be
large or subject to repository and privacy constraints.

Expected inputs:

- metadata CSV: one row per item (or rows aggregatable by handle);
- mapping CSV: Assetstore/bitstream ID and handle;
- full text: a directory of `<internal_id>.txt` files or Parquet shards with
  `file_id` and `text`;
- target fields: configured explicitly in YAML.

`examples/input/` contains the small synthetic corpus used for tests and
demonstration. `examples/output/` contains a curated synthetic result example.
Neither directory contains real SEDICI records.

In Colab, very large TXT directories use `fulltext_format: gdrive_api` because
the mounted filesystem may return an empty listing. The Drive API index is
cached; text content is downloaded only for selected handles.

Downloaded files are cached individually under the persistent
`cache/fulltext` directory. After a Colab timeout, rerunning a resumable
profile downloads only files that are still missing.

