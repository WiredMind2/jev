# Local data

Raw downloads and converted JSONL stay **out of git**. This directory is
the on-disk layout the rest of the docs assume.

```text
data/
  README.md                 # this file
  banking77/jsonl/          # converted Choice rows
  sst5/jsonl/
  boolq/jsonl/
  clinc150/jsonl/
  wikispeedia/              # SNAP archives + jsonl (see jevlike script)
  synthetic/                # jevlike-data synthetic
```

Conversion must emit
[`../schemas/training-example.schema.json`](../schemas/training-example.schema.json)
and record `metadata.domain`, `metadata.group_id`, and a frozen
`metadata.schema_version` for the criteria text.

See [Datasets](../docs/10-datasets.md) for sources, licenses, and the v0
mix. Do not commit customer data or TypeSafe API dumps.
