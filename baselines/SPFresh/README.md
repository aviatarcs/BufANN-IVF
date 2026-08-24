# Running SPFresh workloads in BufANN

This directory contains the bundled SPFresh source. BufANN runs SPFresh through the launcher scripts under [`benchmark/SPFresh/`](../../benchmark/SPFresh/). The original upstream documentation is preserved in [`README.old.md`](README.old.md).

Run all commands below from the root of the BufANN repository.

## Before running a workload

Configure the selected dataset under `benchmark/datasets/` and follow the top-level [`README.md`](../../README.md) to prepare its converted vectors and deep ground truth. Initialize and build SPFresh's bundled dependencies and set up RocksDB by following the original instructions in [`README.old.md`](README.old.md). The launcher scripts configure and build the required SPFresh executables when needed. They expect RocksDB to be installed in the benchmark dependency prefix (`../baselines_env` by default); set `SPFRESH_ROCKSDB_DIR` if its CMake package is installed elsewhere.

## Prepare, then run

Each workload has a preparation script and a run script. The preparation step creates the workload tree, derives the appropriate ground truth, and builds or restores the native SPFresh index. The run step consumes that prepared tree.

```bash
# Query
bash benchmark/SPFresh/sift/query_prepare.sh
bash benchmark/SPFresh/sift/query.sh

# Insert
bash benchmark/SPFresh/sift/insert_prepare.sh
bash benchmark/SPFresh/sift/insert.sh

# Delete
bash benchmark/SPFresh/sift/delete_prepare.sh
bash benchmark/SPFresh/sift/delete.sh

# Mixed insert/delete update
bash benchmark/SPFresh/sift/update_prepare.sh
bash benchmark/SPFresh/sift/update.sh
```

The default dataset is `sift1m`. Select another configured profile on both commands of a pair:

```bash
DATASET=sift10m bash benchmark/SPFresh/sift/query_prepare.sh
DATASET=sift10m bash benchmark/SPFresh/sift/query.sh
```

The workloads exercise SPFresh as follows:

- `query` searches an index built from the configured base portion.
- `insert` builds the base index, inserts vectors from the following portion, and queries the resulting index.
- `delete` builds the base index, deletes vectors from its tail, and queries the resulting index.
- `update` performs mixed insert and delete batches and evaluates the updated index.

Insert, delete, and update mutate their prepared index. Run the corresponding `*_prepare.sh` again before repeating one of those workloads.

## Files produced by the scripts

Prepared, workload-specific files live under:

```text
eval_tempfiles/<dataset>/SPFresh/<workload>/
```

Reusable native indexes are stored under:

```text
eval_cached/<dataset>/SPFresh/
```

Each run writes its console output and evaluation results to `run.log` inside the workload directory.

Common overrides include `UPDATE_PERCENT`, `INSERT_POINTS`, `DELETE_POINTS`, and `UPDATE_BATCH`. Thread counts can be adjusted with `SPFRESH_QUERY_THREADS`, `SPFRESH_INSERT_THREADS`, and `SPFRESH_DELETE_THREADS`.
