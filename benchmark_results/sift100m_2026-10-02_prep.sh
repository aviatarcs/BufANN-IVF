#!/bin/bash
# Prepare the SIFT100M (90M base) IVF-PQ vs BufANN comparison, all on the
# NVMe-backed /tmpdata. Each step logs to /tmpdata/bench100m/<step>.log.
set -x
L=/tmpdata/bench100m
IVF=/tmpdata/BufANN-IVF/build/tests
CS=/tmpdata/BufANN-CS395T
export DATASET=sift100m SIFT100M_ROOT=/tmpdata/sift100m SIFT100M_INDEX_DIR=/tmpdata/indexes/sift100m BUILD_RAM_GB=100

# 1. IVF-PQ indexes (index file v5: one copy of the PQ codes).
for n in 131072 262144; do
  /usr/bin/time -v $IVF/ivf_pq_build_index --data_type uint8 --data_file /tmpdata/sift100m/base.90M.bin \
    --index_prefix /tmpdata/ivf_bench/sift100m_v5/n$n --dim 128 --ivf_nlist $n --ivf_pq_chunks 32 > $L/ivf_build_n$n.log 2>&1
  echo "ivf build n$n exit=$?"
done

# 2. base.100M.bin = base.90M rows then tail.10M rows (the shuffled first
#    100M of SIFT1B, per make_sift100m.py), under a 100M-row header.
B=/tmpdata/sift100m/base.100M.bin
if [[ ! -f $B ]]; then
  python3 -c "import struct,sys; sys.stdout.buffer.write(struct.pack('<ii', 100000000, 128))" > $B.tmp &&
  tail -c +9 /tmpdata/sift100m/base.90M.bin >> $B.tmp && tail -c +9 /tmpdata/sift100m/tail.10M.bin >> $B.tmp &&
  [[ $(stat -c %s $B.tmp) == 12800000008 ]] && mv $B.tmp $B
fi
echo "base.100M exit=$? size=$(stat -c %s $B 2>/dev/null)"

# 3-5. BufANN's own pipeline.
cd $CS
bash benchmark/gen_deep_gt.sh sift100m > $L/gen_deep_gt.log 2>&1;            echo "deep gt exit=$?"
bash benchmark/scripts/build_diskann_index.sh sift100m > $L/diskann_build.log 2>&1; echo "diskann build exit=$?"
bash benchmark/BufANN/sift/query_prepare.sh > $L/bufann_query_prepare.log 2>&1;   echo "bufann prepare exit=$?"

# 6. Warmup queries as ensure_bufann_warmup_queries samples them.
python3 benchmark/sample_base_warmup.py $B uint8 1000 $L/warmup_90M_n1000_seed0.bin --seed 0 --npts-limit 90000000 \
  > $L/warmup.log 2>&1; echo "warmup exit=$?"
echo ALL_DONE
