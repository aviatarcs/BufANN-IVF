#!/bin/bash
# Downloads the raw files for the high-dim comparison, one directory each.
D=/tmpdata/work/data/highdim
get() { # url out
  [[ -s $2 ]] && return 0
  curl -fsSL --retry 8 --retry-delay 5 -o $2.part "$1" && mv $2.part $2 || { echo "FAIL $1"; return 1; }
}
mkdir -p $D/laion512/raw $D/cohere768/raw $D/dbpedia1536/raw
(
  LU=https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb
  for i in 0 1 2 3 4 5 6 7 8 9 10 405; do get $LU/img_emb_$i.npy $D/laion512/raw/img_emb_$i.npy & 
    (( i % 4 == 3 )) && wait; done; wait; echo LAION512_DONE
) &
(
  CU=https://assets.zilliz.com/benchmark/cohere_large_10m
  get $CU/test.parquet $D/cohere768/raw/test.parquet
  get $CU/neighbors.parquet $D/cohere768/raw/neighbors.parquet
  for i in 00 01 02 03 04 05 06 07 08 09; do get $CU/train-$i-of-10.parquet $D/cohere768/raw/train-$i-of-10.parquet & 
    [[ $i == 04 ]] && wait; done; wait; echo COHERE_DONE
) &
(
  for f in $(curl -s https://huggingface.co/api/datasets/KShivendu/dbpedia-entities-openai-1M | python3 -c "import json,sys; [print(s['rfilename']) for s in json.load(sys.stdin)['siblings'] if s['rfilename'].endswith('.parquet')]"); do
    get https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M/resolve/main/$f $D/dbpedia1536/raw/$(basename $f)
  done; echo DBPEDIA_DONE
) &
wait; echo FETCH_ALL_DONE
