#!/bin/bash
# Copies everything the high-dim search phase reads from /tmpdata/work/data/highdim
# to /nvme/highdim (same layout), at low CPU/IO priority, then checks it:
# sizes of every file, and SHA-256 of 3 x 64 MB slices (start, middle, end) of
# every file (whole file when smaller). highdim.sh search reads it with
# HD_ROOT=/nvme/highdim.
set -u
S=/tmpdata/work/data/highdim; D=/nvme/highdim
LOG=/tmpdata/work/logs/highdim/stage_nvme.log
say() { echo "$(date '+%F %T') [stage] $*" | tee -a /tmpdata/work/logs/highdim/status.log; }
cp1() { mkdir -p "$(dirname "$D/$1")"; nice -n 10 ionice -c3 rsync -a --inplace "$S/$1" "$D/$1"; }

say "start"
for ds in gist960 dbpedia1536 laion512 cohere768; do
  for f in base.fbin query.10K.fbin gt100.bin gt10.bin warmup_n1000_seed0.bin; do cp1 $ds/$f || say "FAIL $ds/$f"; done
  cp1 $ds/ivf/ || say "FAIL $ds/ivf"
  cp1 $ds/bufann/ || say "FAIL $ds/bufann"
  cp1 rabitq/indexes/$ds/ || say "FAIL rabitq/indexes/$ds"
  # RaBitQ-SSD's data dir: symlinks to the dataset's own files, now on /nvme.
  R=$D/rabitq/data/$ds; mkdir -p $R
  ln -sfn $D/$ds/base.fbin $R/train.fbin; ln -sfn $D/$ds/query.10K.fbin $R/test.fbin
  ln -sfn $D/$ds/gt100.bin $R/gt10; ln -sfn $D/$ds/gt100.bin $R/gt100
  say "$ds copied"
done

slice_sums() {  # file -> sha256 of 3 slices
  local f=$1 n; n=$(stat -c %s "$f")
  if (( n <= 3 * 64 * 1024 * 1024 )); then sha256sum < "$f" | cut -c1-16; return; fi
  for off in 0 $(( n / 2 / 1048576 )) $(( n / 1048576 - 64 )); do
    dd if="$f" iflag=direct bs=1M skip=$off count=64 status=none | sha256sum | cut -c1-16
  done | tr '\n' ' '
}
bad=0
cd $S
while IFS= read -r f; do
  [[ -f $D/$f ]] || { echo "MISSING $f" >> $LOG; bad=$((bad+1)); continue; }
  [[ $(stat -c %s "$f") == $(stat -c %s "$D/$f") ]] || { echo "SIZE $f" >> $LOG; bad=$((bad+1)); continue; }
  [[ "$(slice_sums "$f")" == "$(slice_sums "$D/$f")" ]] || { echo "SUM $f" >> $LOG; bad=$((bad+1)); }
done < <(find gist960 dbpedia1536 laion512 cohere768 rabitq/indexes -type f \
          \( -path '*/ivf/*' -o -path '*/bufann/*' -o -path 'rabitq/indexes/*' -o -name '*.fbin' -o -name 'gt*.bin' -o -name 'warmup*' \) \
          ! -path '*/diskann/*' ! -path '*/raw/*')
say "verify done: $bad problems ($(du -sh $D | cut -f1) on /nvme)"
