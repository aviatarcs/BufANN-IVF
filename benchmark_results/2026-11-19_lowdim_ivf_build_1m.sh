#!/bin/bash
# IVF-PQ indexes (agent/lowdim-gap = agent/rerank-batches code), untimed.
B=/tmpdata/work/wt-lowdim/build/tests/ivf_pq_build_index
O=/tmpdata/work/data/ivf L=/tmpdata/work/logs/lowdim
b() { # name file dim nlist chunks
  mkdir -p $O/$1
  [[ -f $O/$1/done ]] && return
  /usr/bin/time -v $B --data_type float --data_file $2 --index_prefix $O/$1/n$4 --dim $3 --ivf_nlist $4 --ivf_pq_chunks $5 > $L/ivf_build_$1.log 2>&1
  echo "$1 exit=$?"; touch $O/$1/done
}
b sift1m /local/data/sift/sift_base.fbin 128 1024 32
b gist1m /local/data/gist/gist_base.fbin 960 1024 120
echo ALL_DONE
