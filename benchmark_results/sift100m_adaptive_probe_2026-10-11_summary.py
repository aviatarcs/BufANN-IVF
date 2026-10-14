# Best q/s at recall thresholds, fixed vs adaptive, per nlist; and each fixed
# point against the fastest adaptive run of at least its recall.
import json, glob, os, re
rows = []
for f in glob.glob("/tmpdata/bench100m/results_adaptive/*.log  # the run script's logs"):
    for line in open(f):
        if line.startswith('{"baseline"'):
            r = json.loads(line); r["tag"] = os.path.basename(f)[:-4]; rows.append(r)
for n in (131072, 262144):
    R = [r for r in rows if r["nlist"] == n]
    fixed = sorted([r for r in R if r["probe_tau"] == 0], key=lambda r: r["nprobe"])
    adapt = [r for r in R if r["probe_tau"] > 0]
    print(f"nlist {n}")
    for t in (90, 95, 97, 98, 99):
        bf = max([r["query_qps"] for r in fixed if r["recall"] >= t], default=0)
        ba = max([r["query_qps"] for r in adapt if r["recall"] >= t], default=0)
        print(f"  recall >= {t}: fixed {bf:7.0f}  adaptive {ba:7.0f}  {'%+.0f%%' % (100*(ba/bf-1)) if bf and ba else ''}")
    for r in fixed:
        c = [a for a in adapt if a["recall"] >= r["recall"] - 0.005]
        if not c: print(f"  fixed nprobe {r['nprobe']}: no adaptive run reaches recall {r['recall']:.3f}"); continue
        a = max(c, key=lambda a: a["query_qps"])
        print(f"  fixed nprobe {r['nprobe']:4d} recall {r['recall']:.3f} {r['query_qps']:7.0f} q/s p99 {r['query_lat_p99_us']/1e3:5.1f} ms {r['codes_avg']/1e3:4.0f}K codes"
              f" | adaptive cap {a['nprobe']} tau {a['probe_tau']:.2f} recall {a['recall']:.3f} {a['query_qps']:7.0f} q/s ({100*(a['query_qps']/r['query_qps']-1):+.0f}%)"
              f" p99 {a['query_lat_p99_us']/1e3:5.1f} ms, {a['probes_avg']:.0f} lists {a['codes_avg']/1e3:4.0f}K codes")
