"""IVF-PQ vs BufANN, best q/s (and its p99) at recall thresholds, per dataset."""
import glob, json, re
from collections import defaultdict
TH = (90, 95, 98, 99)
ivf, buf = defaultdict(list), defaultdict(list)
for p in glob.glob("results_ivf/*.log"):
    ds = p.split("/")[-1].split("_")[0]
    for l in open(p):
        if l.startswith("{"):
            r = json.loads(l)
            if r["rerank_batch"] == 0: ivf[ds].append(r)
for p in glob.glob("results_bufann/*_L*.json"):
    ds = p.split("/")[-1].split("_")[0]
    r = json.load(open(p)); r["L"] = int(re.search(r"_L(\d+)", p).group(1)); buf[ds].append(r)
def best(rows, th):
    ok = [r for r in rows if r["recall"] >= th]
    return max(ok, key=lambda r: r["query_qps"]) if ok else None
for ds in ("sift100m", "deep100m", "spacev100m", "turing100m"):
    print(f"\n{ds}: max recall IVF-PQ {max(r['recall'] for r in ivf[ds]):.2f}, BufANN {max(r['recall'] for r in buf[ds]):.2f}")
    for th in TH:
        a, b = best(ivf[ds], th), best(buf[ds], th)
        fa = f"{a['query_qps']:7.0f} p99 {a['query_lat_p99_us']/1e3:5.1f} (P{a['nprobe']} tau{a['probe_tau']:.2f} beam{a['centroid_L']} m{a['rerank_m']})" if a else "      -"
        fb = f"{b['query_qps']:7.0f} p99 {b['query_lat_p99_us']/1e3:5.1f} (L{b['L']})" if b else "      -"
        ratio = f"{a['query_qps']/b['query_qps']:.2f}x" if a and b else ""
        print(f"  >= {th}: IVF-PQ {fa:55s} BufANN {fb:30s} {ratio}")
    rss_i = max(r["rss_mb"] for r in ivf[ds]); rss_b = max(r["peak_rss_mb"] for r in buf[ds])
    print(f"  RSS IVF-PQ {rss_i:.0f} MB, BufANN {rss_b:.0f} MB")
