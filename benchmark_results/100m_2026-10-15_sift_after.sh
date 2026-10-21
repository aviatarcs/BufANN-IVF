#!/bin/bash
# After orchestrate.sh (DEEP/SPACEV/Turing): SIFT100M rerun on this node
# (data made by make_sift100m_nvme.py: shuffled first 100M of SIFT1B, 90M
# base), with the same steps as the other three, then the Turing wider-beam
# test. One thing at a time, so every query run has a quiet node.
set -x
L=/nvme/bench100m
D=/nvme/100m/sift100m
GT=$D/sift100m_90000000_gt100.bin
# 1. exact top-100 over the 90M base, backed up with the others.
[[ -f $GT ]] || /usr/bin/time -v $L/exact_gt uint8 $D/base.90M.bin $D/query.10K.u8bin 100 $GT > $L/gt_sift100m.log 2>&1 || exit 1
G=/proj/ut-data-PG0/evpeng/gt100m/sift100m
mkdir -p $G && cp $GT $D/query.10K.u8bin $L/warmup_sift100m_90M_n1000_seed0.bin $G/ &&
  (cd /proj/ut-data-PG0/evpeng/gt100m && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
# 2. IVF-PQ index, nlist 131072 (SIFT100M's best on Oct 10-11), 32 chunks.
P=/nvme/ivf_bench/sift100m/n131072
mkdir -p $(dirname $P)
[[ -f ${P}_ivf_pq_index.bin ]] || /usr/bin/time -v /tmpdata/BufANN-IVF/build/tests/ivf_pq_build_index --data_type uint8 \
  --data_file $D/base.90M.bin --index_prefix $P --dim 128 --ivf_nlist 131072 --ivf_pq_chunks 32 > $L/ivf_build_sift100m.log 2>&1 || exit 1
# 3. IVF-PQ query runs, the ivf_runs.sh grid.
source <(sed -n '/^B=/,/^}/p' $L/ivf_runs.sh)
for P in 32 64 128 256 512 1024; do run sift100m_fixed_p$P sift100m uint8 u8bin $P 1.25 0 100 0:0,10:0,10:1,10:2; done
for C in 160 320 640 1280; do run sift100m_adapt_c$C sift100m uint8 u8bin $C 1 1.3,1.4,1.5,1.6,1.7 100 0:0,10:2; done
# 4. BufANN: canonical DiskANN build (orchestrate.sh's steps), then its runs.
source <(sed -n '/^BIN=/,/^}/p' $L/orchestrate.sh)
build sift100m uint8 u8bin 0; echo "sift100m diskann build exit=$? $(date +%T)"
$L/bufann_one.sh sift100m uint8 u8bin 128 > $L/bufann_sift100m.log 2>&1; echo "sift100m bufann exit=$? $(date +%T)"
# 5. Turing: is its recall ceiling the centroid beam? Beams 2 and 3 x nprobe.
run turing100m_beam_p1024 turing100m float fbin 1024 2,3 0 100 0:0
run turing100m_beam_p2048 turing100m float fbin 2048 2,3 0 100 0:0
run turing100m_beam_c1280 turing100m float fbin 1280 2,3 1.4,1.6 100 0:0
echo ALL_DONE
