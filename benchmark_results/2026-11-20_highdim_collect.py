#!/usr/bin/env python3
"""Collects the high-dim runs into one jsonl and prints, per set, dataset and
memory budget, each system's best q/s at recall@10 >= 90/95/98/99, with the
SSD page reads per query and the device reads/s at that point.

  collect.py <out.jsonl>

Two sets, in the `set` field:
  tmpdata  first pass on /tmpdata (LVM stripe over the SATA disk + 4 NVMe,
           ~0.46M 4 KB random reads/s), pre-fix binaries, IVF-PQ through a
           1 GB buffer pool (io_mode pool). Superseded by nvme.
  nvme     /nvme (4-NVMe stripe, ~1.8M reads/s), binaries from agent/highdim
           efd6743 (pin-count fix + async re-rank), IVF-PQ with uncached
           O_DIRECT re-rank reads (io_mode direct) at pipeline depth 0 and 2.
"""
import csv, glob, json, os, re, sys

W = "/tmpdata/work"
SETS = {"tmpdata": (f"{W}/logs/highdim", f"{W}/logs/highdim/rabitq_merged", ""),
        "nvme": (f"{W}/logs/highdim_nvme", f"{W}/highdim-src/RaBitQ-SSD/results/main_exp/merged", "_nvme")}
DS = {  # name: (N, D, pq_small, pq_large, memdim_small, memdim_large)
    "gist960": (1000000, 960, 64, 120, 512, 960),
    "dbpedia1536": (990000, 1536, 96, 192, 768, 1536),
    "laion512": (10000000, 512, 32, 64, 256, 512),
    "cohere768": (9990000, 768, 48, 96, 384, 768),
}
rows = []
for st, (logs, rq, tag) in SETS.items():
    for ds, (N, D, cs, cl, ms, ml) in DS.items():
        d = f"{logs}/{ds}"
        for lg in sorted(glob.glob(f"{d}/ivf_pq*.log")):
            for line in open(lg, errors="replace"):
                if line.startswith('{"baseline":"IVF-PQ"'):
                    r = json.loads(line)
                    r.setdefault("io_mode", "pool")       # the tmpdata bench predates the field
                    r.setdefault("pipeline_depth", 0)
                    rows.append({"set": st, "dataset": ds, **r, "run": os.path.basename(lg)[:-4]})
        for js in sorted(glob.glob(f"{d}/bufann_pq*_L*.json")):
            m = re.search(r"bufann_pq(\d+)_L(\d+)\.json", js)
            r = json.load(open(js))
            r = r[0] if isinstance(r, list) else r
            flat = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
            for k, v in r.items():
                if isinstance(v, dict):
                    flat.update({kk: vv for kk, vv in v.items() if not isinstance(vv, (dict, list))})
            rows.append({"set": st, "dataset": ds, "baseline": "BufANN", "N": N, "pq_bytes": int(m[1]),
                         "search_L": int(m[2]), **flat})
        for f in sorted(glob.glob(f"{rq}/{ds}_C*/query_results.csv")):
            cfg = os.path.basename(os.path.dirname(f))
            if ("_nvme_" in cfg) != bool(tag):
                continue
            for r in csv.DictReader(open(f)):
                # *_after_exbits: the A-B-A repeat after the IVF-PQ runs (drift check), not a system point
                rows.append({"set": st, "dataset": ds, "N": N, "baseline": "RaBitQ-SSD", "config": cfg,
                             "aba_repeat": "_after_" in cfg, **r})

with open(sys.argv[1], "w") as out:
    for r in rows:
        out.write(json.dumps(r) + "\n")


def points(st, ds, budget):
    N, D, cs, cl, ms, ml = DS[ds]
    pq, md = (cs, ms) if budget == 0 else (cl, ml)
    sysp = {}
    for r in rows:
        if r["set"] != st or r["dataset"] != ds:
            continue
        if r["baseline"] == "IVF-PQ" and r["pq_chunks"] == pq:
            k = f"IVF-PQ {r['io_mode']} d{r['pipeline_depth']}"
            sysp.setdefault(k, []).append((r["recall"], r["query_qps"], r["misses_per_query"]))
        elif r["baseline"] == "BufANN" and r["pq_bytes"] == pq and "recall" in r:
            sysp.setdefault("BufANN", []).append(
                (float(r["recall"]), float(r["query_qps"]), float(r["page_fault_total"]) / float(r["query_count"])))
        elif r["baseline"] == "RaBitQ-SSD" and f"memdim{md}_" in r["config"]:
            k = "RaBitQ raw" if r["config"].endswith("_raw") else "RaBitQ exbits"
            if r["aba_repeat"]:
                k += " (repeat after IVF)"
            sysp.setdefault(k, []).append(
                (100 * float(r["recall_at_topk"]), float(r["qps"]), float(r["io_pages_mean"])))
    return pq, md, dict(sorted(sysp.items()))


print("best q/s at recall@10 >= target, as q/s (SSD page reads per query, M reads/s);\n"
      "- = not reached; max recall in brackets")
for st in SETS:
    for ds in DS:
        for budget in (0, 1):
            pq, md, sysp = points(st, ds, budget)
            if not sysp:
                continue
            print(f"\n[{st}] {ds}  PQ {pq} B / RaBitQ memdim {md} bits")
            print(f"  {'system':18s}" + "".join(f"{t:>22d}" for t in (90, 95, 98, 99)) + "   max recall")
            for s, p in sysp.items():
                cells = []
                for t in (90, 95, 98, 99):
                    q = [(qps, pg) for rec, qps, pg in p if rec >= t]
                    if q:
                        qps, pg = max(q)
                        cells.append(f"{f'{qps:.0f} ({pg:.0f}, {qps * pg / 1e6:.2f}M)':>22s}")
                    else:
                        cells.append(f"{'-':>22s}")
                print(f"  {s:18s}" + "".join(cells) + f"   [{max(r for r, _, _ in p):.2f}]")
