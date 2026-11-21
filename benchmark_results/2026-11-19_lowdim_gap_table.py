import json, sys, collections
d = sys.argv[1]
rows = collections.defaultdict(list)
for l in open(f'{d}/100m_2026-10-15.jsonl'):
    r = json.loads(l)
    if r.get('baseline') == 'IVF-PQ':
        rows[(r['dataset'], 'IVF-PQ')].append((r['recall'], r['query_qps'], r))
    elif r.get('recall') is not None and 'query_qps' in r:
        rows[(r['dataset'], r.get('baseline','BufANN'))].append((r['recall'], r['query_qps'], r))
for l in open(f'{d}/rabitq_ssd_100m_2026-10-20.jsonl'):
    r = json.loads(l)
    rows[(r['dataset'], 'RQ:' + r['config'][:22])].append((100*float(r['recall_at_topk']), float(r['qps']), r))
T = [90, 95, 98, 99]
for ds in ['sift100m', 'deep100m', 'spacev100m', 'turing100m']:
    print(ds)
    for (dd, sysn), pts in sorted(rows.items()):
        if dd != ds: continue
        best = []
        for t in T:
            c = [p for p in pts if p[0] >= t]
            best.append(max(c, key=lambda p: p[1]) if c else None)
        print('  %-30s' % sysn + ' '.join('%7.0f' % b[1] if b else '      -' for b in best))
        if '-v' in sys.argv:
            for t, b in zip(T, best):
                if not b: continue
                r = b[2]
                if sysn == 'IVF-PQ':
                    print('     r>=%d rec %.2f nprobe %s tau %s probes %.0f codes %d rerank %s mpq %.1f lat %.0f cent %.0f scan %.0f fetch %.0f rr %.0f' % (t, b[0], r['nprobe'], r.get('probe_tau'), r['probes_avg'], r['codes_avg'], r['rerank_m'], r['misses_per_query'], r['query_lat_avg_us'], r['t_centroid_us'], r['t_scan_us'], r['t_fetch_us'], r['t_rerank_us']))
                elif sysn.startswith('RQ'):
                    print('     r>=%d rec %.2f nprobe %s pages %s bytes %s lat %s rss %s' % (t, b[0], r['op_param'], r['io_pages_mean'], r['io_bytes_mean'], r['lat_mean_us'], r['peak_rss_mb']))
