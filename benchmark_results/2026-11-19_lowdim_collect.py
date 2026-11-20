"""Collect this investigation's runs into one JSONL and print the comparison.

RaBitQ-SSD: its query CSV (OUTPUT_SCHEMA.md) joined with the per-nprobe
'Scan candidate-insertion stats' table from the run log (probed = codes
scanned, cand_ins = passed the in-RAM bound, iopages = pages read,
boosts = records refined/re-ranked). IVF-PQ: ivf_pq_query_bench JSON lines.
usage: collect.py <out.jsonl>
"""
import csv, glob, json, os, re, sys
R = '/tmpdata/work/rabitq-ssd/results/main_exp'
L = '/tmpdata/work/logs/lowdim'
rows = []
for d in sorted(glob.glob(f'{R}/*_*')):
    if d.endswith('merged'): continue
    tag = os.path.basename(d)
    for c in glob.glob(f'{d}/csv/*.csv'):
        log = c.replace('/csv/', '/logs/').replace('.csv', '.log')
        stats = {}
        txt = open(log).read()
        m = txt.split('=== Scan candidate-insertion stats (avg per query) ===')
        if len(m) > 1:
            lines = m[1].strip().splitlines()
            hdr = lines[0].split()
            for ln in lines[1:]:
                v = ln.split()
                if len(v) != len(hdr): break
                stats[v[0]] = dict(zip(hdr, map(float, v)))
        for r in csv.DictReader(open(c)):
            o = {'system': 'RaBitQ-SSD', 'dataset': r['dataset'], 'config': tag.split('drainA0.4_')[-1],
                 'run_id': r['run_id'], 'threads': int(r['threads']), 'nprobe': int(float(r['op_param'])),
                 'recall': 100 * float(r['recall_at_topk']), 'qps': float(r['qps']), 'lat_mean_us': float(r['lat_mean_us']),
                 'lat_p99_us': float(r['lat_p99_us']), 'pages_per_query': float(r['io_pages_mean']),
                 'eff_iops': float(r['eff_iops']), 'cpu_util_pct': r['cpu_util_pct'], 'peak_rss_mb': r['peak_rss_mb'],
                 'index_ssd_bytes': r['index_ssd_bytes']}
            s = stats.get(r['op_param']) or stats.get(str(o['nprobe']))
            if s:
                o.update(codes_scanned=s['probed'], passed_bound=s['cand_ins'], records_refined=s['boosts'],
                         records_examined=s['recs_exam'])
            rows.append(o)
for f in sorted(glob.glob(f'{L}/ivf_*.log')):
    if 'build' in f: continue
    name = os.path.basename(f)[4:-4]
    for ln in open(f):
        if not ln.startswith('{'): continue
        r = json.loads(ln)
        r.update(system='IVF-PQ', run=name, dataset=name.split('_')[0], qps=r['query_qps'])
        rows.append(r)
with open(sys.argv[1], 'w') as out:
    for r in rows: out.write(json.dumps(r) + '\n')
T = [90, 95, 98, 99]
groups = {}
for r in rows:
    key = (r['dataset'], r['system'], r.get('config') or r['run'])
    groups.setdefault(key, []).append(r)
for key in sorted(groups):
    pts = groups[key]
    cells = []
    for t in T:
        c = [p for p in pts if p['recall'] >= t]
        if not c: cells.append('%38s' % '-'); continue
        b = max(c, key=lambda p: p['qps'])
        if b['system'] == 'IVF-PQ':
            extra = 'np%d rr%d miss%.0f' % (b['nprobe'], b['rerank_m'], b['misses_per_query'])
        else:
            extra = 'np%d pg%.0f' % (b['nprobe'], b['pages_per_query'])
        cells.append('%7.0f (%5.2f %-20s)' % (b['qps'], b['recall'], extra))
    print('%-8s %-10s %-22s' % key + ' '.join(cells))
