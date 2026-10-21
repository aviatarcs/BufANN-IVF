"""Best q/s at recall thresholds per dataset and re-rank setting, with the winning
setting's per-query time breakdown (us, summed over phases; IVFPQSearchTimes)."""
import glob, json, sys
from collections import defaultdict

R = sys.argv[1] if len(sys.argv) > 1 else "/nvme/bench100m/results_ivf"
rows = defaultdict(list)
for path in glob.glob(f"{R}/*.log"):
    ds = path.split("/")[-1].split("_")[0]
    for l in open(path):
        if l.startswith("{"):
            r = json.loads(l)
            r["mode"] = "adaptive" if r["probe_tau"] > 0 else "fixed"
            rows[ds].append(r)

PH = ["t_centroid_us", "t_table_us", "t_scan_us", "t_select_us", "t_fetch_us", "t_rerank_us"]
for ds in sorted(rows):
    print(f"\n## {ds}")
    groups = defaultdict(list)
    for r in rows[ds]:
        groups[(r["rerank_m"], r["rerank_batch"], r["rerank_patience"])].append(r)
    for key in sorted(groups):
        line = []
        for th in (90, 95, 98, 99):
            ok = [r for r in groups[key] if r["recall"] >= th]
            line.append("-" if not ok else f"{max(ok, key=lambda r: r['query_qps'])['query_qps']:.0f}")
        print(f"rerank_m {key[0]:3d} sched {key[1]}:{key[2]}  best q/s at recall >= 90/95/98/99: {' / '.join(line)}"
              f"   (max recall {max(r['recall'] for r in groups[key]):.2f})")
    print("breakdown of the best setting at recall >= 95 (default schedule), us per query:")
    for key in sorted(groups):
        if key[1] != 0: continue
        ok = [r for r in groups[key] if r["recall"] >= 95]
        if not ok: continue
        b = max(ok, key=lambda r: r["query_qps"])
        tot = sum(b[p] for p in PH)
        parts = "  ".join(f"{p[2:-3]} {b[p]:.0f} ({100*b[p]/tot:.0f}%)" for p in PH)
        print(f"  rerank_m {key[0]}: {b['mode']} P{b['nprobe']} tau {b['probe_tau']:.2f}: q/s {b['query_qps']:.0f}, recall "
              f"{b['recall']:.2f}, p99 {b['query_lat_p99_us']/1e3:.1f} ms, {b['codes_avg']:.0f} codes, "
              f"{b['misses_per_query']:.0f} misses/q\n    {parts}")
