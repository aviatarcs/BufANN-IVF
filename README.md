# BufANN

This repository contains BufANN and the scripts used to prepare and run its query-only, insertion, deletion, and mixed-update workloads.

## 1. Compile BufANN and the DiskANN build helpers

```bash
bash benchmark/scripts/build.sh
```

The BufANN driver is written to `build/tests/bufann_driver`. The DiskANN helpers are written to `baselines/DiskANN/build/tests/`.

## 2. Configure a dataset

Edit the appropriate file under `benchmark/datasets/` and configure its source vectors, dataset path, and index path. The scripts handle conversion to `.bin` and ground-truth generation.

### Dataset downloads

Download the 128-dimensional float32 SIFT1M [base vectors](https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_base.fvecs?download=true) and [query vectors](https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_query.fvecs?download=true) in `fvecs` format.

SIFT1B, also called BIGANN, is available from the [Big ANN Benchmarks dataset page](https://big-ann-benchmarks.com/neurips21.html) as separate [base](https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin), [query](https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin), and [learning](https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/learn.100M.u8bin) vector files. These are 128-dimensional uint8 `u8bin` inputs, i.e. they don't need conversion from `fvecs` to `bin`.

### Dataset configuration

Datasets usually come in two dominant formats: the `.fvecs` or `.bvecs` format, which stores duplicated dimension count per vector, and the `.bin` (or `.u8bin` or other variants), which states the vector count and dimension once upfront, followed by all vectors packed. BufANN uses `.bin` as input, but if the `.bin` input path doesn't exist, it looks for configured `FVECS` or `BVECS` paths and convert from them.

For SIFT1M, set `DATA_FVECS` and `QUERY_FVECS` to the downloaded `sift_base.fvecs` and `sift_query.fvecs` files. The scripts convert them to float `.bin` files when needed.

For SIFT10M and SIFT100M, create your own sample from SIFT1B and supply the resulting `.bin` file.

The first `NPTS_BASE` vectors form the initial index. The remaining `UPDATE_POINTS` vectors are reserved for insertion and mixed-update workloads. Verify that `DATA_TYPE`, `DIM`, `NPTS_FULL`, `NPTS_BASE`, and `UPDATE_POINTS` match the configured files.

## 3. Generate deep ground truth

BufANN evaluation frequently needs ground truth for a database that contains only a particular ID range from the full dataset file. A deep ground-truth file stores the top **K=100** nearest-neighbor IDs and distances for every query against the full dataset. The evaluation scripts filter those candidates on the fly to obtain the ground truth for the active ID range without repeating the expensive full nearest-neighbor computation.

Generate this file once for the configured dataset:

```bash
bash benchmark/gen_deep_gt.sh sift1m
```

The output is written to `GT_BIN_KCACHED` from the dataset configuration.

## 4. Build PQ data

Train PQ pivots on the full dataset and generate PQ codes for the first `NPTS_BASE` vectors:

```bash
bash benchmark/scripts/build_pq.sh sift1m
```

The default is 32 PQ chunks (i.e. 32 bytes per vector) and can be overridden with `PQ_CHUNKS`.

## 5. Build the DiskANN graph

Build the graph over the first `NPTS_BASE` vectors:

```bash
bash benchmark/scripts/build_diskann_index.sh sift1m
```

Configure the graph degree with `DISKANN_R` in the dataset profile (default =64). The build-time search-list size and memory budget default to `BUILD_L=100` and `BUILD_RAM_GB=200`. Override the latter two when invoking the graph builder, for example:

```bash
BUILD_L=200 BUILD_RAM_GB=64 bash benchmark/scripts/build_diskann_index.sh sift1m
```

The builder uses all available CPU cores by default; set `BUILD_THREADS` to override that behavior.

(Note: The DiskANN index is sometimes referred to as the "Canonical" graph / index in this repo.)

## 6. Run BufANN

Each workload has a preparation script followed by a run script. The scripts handle conversion from the DiskANN graph to BufANN's heap format and derive workload-specific ground truth from the deep K=100 file.

`eval_tempfiles/` contains the mutable files for the current evaluation run and can be recreated by running the preparation script again. `eval_cached/` stores reusable, expensive-to-produce artifacts such as converted indexes and derived dataset slices so later preparation runs can restore them. If you are using a server with many disk partitions, you might want to use symbolic links to manage where these two folders go, e.g. to enable running very large datasets.

```bash
bash benchmark/BufANN/sift/query_prepare.sh
bash benchmark/BufANN/sift/query.sh

bash benchmark/BufANN/sift/insert_prepare.sh
bash benchmark/BufANN/sift/insert.sh

bash benchmark/BufANN/sift/delete_prepare.sh
bash benchmark/BufANN/sift/delete.sh

bash benchmark/BufANN/sift/update_prepare.sh
NTHREADS=16 MERGE_THREADS=8 bash benchmark/BufANN/sift/update.sh
```

The mixed-update example assumes a 16-thread server and assigns eight threads to maintenance, leaving eight threads for concurrent queries. The workloads are query-only, insert-only, delete-only, and mixed updates. Each run writes `result.json` and `run.log` under `eval_tempfiles/<dataset>/BufANN/<workload>/`.
