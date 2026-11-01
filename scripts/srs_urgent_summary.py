#!/usr/bin/env python3
"""
Aggregate urgent-task analyses (scripts/srs_urgent_analysis.py -> analysis3/*.json; plans/2026-10-31_urgent_task_plan.md).
Groups: A (resolver, identity into SC at up_latency), Afast (identity at vis_latency), B, C, and {A,B,C}haz (hazard-
logit urgency), by latency gap (g20/g50/g80). Exclusions fixed in advance: training not finished (iter < 6000),
completion < 0.9 (test, sal 1), one-sided choice (|P(right) - 0.5| > 0.35 at the end of training).
Per group: median [min, max] over networks of each measure; Mann-Whitney tests between architectures (main sets pooled
over gaps, and per gap); curve shape at matched measured value latency (residual of midpoint/slope after a linear fit on
measured latency across all main-set networks).
Usage: python scripts/srs_urgent_summary.py <analysis3_dir> -> results/hub_models/urgent/{per_network.csv, summary.md,
tachometric.png}
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'hub_models' / 'urgent'
BINS = np.arange(0, 375, 25)
MEASURES = ['mid', 'slope', 'hi', 'lo', 'cong_hi', 'value_latency_ms', 'value_latency_single_ms', 'value_component_peak',
            'divergence_slope_vs_dt', 'error_rich_ratio', 'S9', 'S8_100', 'S8_150', 'mid_sal0.3', 'mid_sal2',
            'hi_sal0.3', 'hi_sal2', 'gap_hi_minus_lo_asym', 'gap_hi_minus_lo_mid', 'rt_median', 'completed']


def row(d):
    m = re.match(r'^(A|B|C|Afast|Ahaz|Bhaz|Chaz)_g(\d+)_s(\d+)$', d['name'])
    f = d['final']
    c = d['sal1']['conflict']
    r = dict(name=d['name'], group=m.group(1), gap=int(m.group(2)), seed=int(m.group(3)), arch=d['arch'],
             iter=f.get('iter'), train_R=f.get('R'), train_Phigh=f.get('Phigh'),
             mid=c['mid'], slope=c['slope'], hi=c['hi'], lo=c['lo'], cong_hi=d['sal1']['congruent']['hi'],
             rt_median=d['sal1']['rt_median'], completed=d['sal1']['completed'],
             value_latency_ms=d.get('value_latency_ms'), value_latency_single_ms=d.get('value_latency_single_ms'),
             value_component_peak=d.get('value_component_peak'),
             divergence_slope_vs_dt=d['divergence_slope_vs_dt'], error_rich_ratio=d['error_rich_ratio'],
             n_capture_errors=d['n_capture_errors'], S9=d['S9'],
             S8_100=d['S8_frac_by_window'].get('100'), S8_150=d['S8_frac_by_window'].get('150'))
    for sg in ('0.3', '2'):
        r[f'mid_sal{sg}'] = d[f'sal{sg}']['conflict']['mid']
        r[f'hi_sal{sg}'] = d[f'sal{sg}']['conflict']['hi']
    g = d['by_gap']
    r['gap_hi_minus_lo_asym'] = g[2]['hi'] - g[0]['hi']
    r['gap_hi_minus_lo_mid'] = g[2]['mid'] - g[0]['mid']
    r['curve'] = c['curve']
    r['div_by_dt'] = d['divergence_ms_by_dt']
    return r


def fmt(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 'n/a'
    return f'{np.median(v):.2f} [{v.min():.2f}, {v.max():.2f}]'


def mw(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return 'n/a'
    return f'{mannwhitneyu(a, b).pvalue:.3g}'


def main():
    src = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [row(json.load(open(p))) for p in sorted(src.glob('*.json'))]
    df = pd.DataFrame(rows)
    df['excluded'] = ''
    df.loc[df['iter'] != 6000, 'excluded'] = 'unfinished'
    df.loc[(df['excluded'] == '') & (df['completed'] < 0.9), 'excluded'] = 'completion<0.9'
    df.loc[(df['excluded'] == '') & ((df['train_R'] - 0.5).abs() > 0.35), 'excluded'] = 'one-sided'
    df.drop(columns=['curve', 'div_by_dt']).to_csv(OUT / 'per_network.csv', index=False)
    ok = df[df['excluded'] == '']
    main_ = ok[ok['group'].isin(['A', 'B', 'C'])]
    # matched measured latency: residuals after a linear fit on measured value latency (main sets)
    for m in ('mid', 'slope', 'hi'):
        x, y = main_['value_latency_ms'].astype(float), main_[m].astype(float)
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() >= 5:
            b = np.polyfit(x[good], y[good], 1)
            df.loc[main_.index, f'{m}_resid'] = y - np.polyval(b, x)
    ok = df[df['excluded'] == '']
    main_ = ok[ok['group'].isin(['A', 'B', 'C'])]

    L = ['# Urgent-task predictions: resolver (A) vs priority hub (B) vs outcome hub (C)\n',
         'Source: scripts/srs_urgent_analysis.py on CRC results3 (plan: plans/2026-10-31_urgent_task_plan.md). '
         'Conflict trials = rich and salient targets on opposite sides; rPT = RT from go - dt. Values: median '
         '[min, max] over networks. Divergence slope vs dt: 0 = target-locked, -1 = go-locked. error_rich_ratio: '
         'rich-pool rate 100-150 ms after target onset on capture errors / correct conflict trials.\n',
         '## Exclusions']
    ex = df[df['excluded'] != '']
    L.append(', '.join(f'{r.name_} ({r.excluded})' for r in ex.rename(columns={'name': 'name_'}).itertuples())
             if len(ex) else 'none')
    L.append(f'\nIncluded: {len(ok)} of {len(df)} networks.\n')
    cols = ['mid', 'slope', 'hi', 'lo', 'value_latency_ms', 'divergence_slope_vs_dt', 'error_rich_ratio', 'S9',
            'S8_150', 'gap_hi_minus_lo_asym', 'mid_sal0.3', 'mid_sal2', 'rt_median']
    L.append('## Per group and gap')
    L.append('| group | gap | n | ' + ' | '.join(cols) + ' |')
    L.append('|' + '---|' * (len(cols) + 3))
    for (g, gp), s in ok.groupby(['group', 'gap']):
        L.append(f'| {g} | {gp} | {len(s)} | ' + ' | '.join(fmt(s[c]) for c in cols) + ' |')
    L.append('\n## Architecture tests (main sets, pooled over gaps; Mann-Whitney p)')
    tcols = cols + [c for c in ('mid_resid', 'slope_resid', 'hi_resid') if c in df]
    L.append('| measure | A | B | C | A vs B | A vs C | B vs C |')
    L.append('|---|---|---|---|---|---|---|')
    for c in tcols:
        a, b, cc = (main_[main_['group'] == k][c] for k in 'ABC')
        L.append(f'| {c} | {fmt(a)} | {fmt(b)} | {fmt(cc)} | {mw(a, b)} | {mw(a, cc)} | {mw(b, cc)} |')
    L.append('\n## Per-gap tests for the key measures (Mann-Whitney p)')
    L.append('| measure | gap | A vs B | A vs C | B vs C |')
    L.append('|---|---|---|---|---|')
    for c in ('divergence_slope_vs_dt', 'error_rich_ratio', 'S8_150', 'slope', 'hi', 'S9'):
        for gp in sorted(main_['gap'].unique()):
            s = main_[main_['gap'] == gp]
            a, b, cc = (s[s['group'] == k][c] for k in 'ABC')
            L.append(f'| {c} | {gp} | {mw(a, b)} | {mw(a, cc)} | {mw(b, cc)} |')
    L.append('\n## Robustness sets (gap 50): resolver with early identity (Afast), hazard-logit urgency (haz)')
    L.append('| group | n | ' + ' | '.join(cols) + ' |')
    L.append('|' + '---|' * (len(cols) + 2))
    for g in ('A', 'Afast', 'B', 'C', 'Ahaz', 'Bhaz', 'Chaz'):
        s = ok[(ok['group'] == g) & (ok['gap'] == 50)]
        if len(s):
            L.append(f'| {g} | {len(s)} | ' + ' | '.join(fmt(s[c]) for c in cols) + ' |')
    (OUT / 'summary.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))

    # figure: tachometric curves (conflict) per group x gap, median over networks; divergence vs dt
    colors = {'A': '#2a6fdb', 'B': '#d9822b', 'C': '#3a9a5b'}
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    for i, gp in enumerate((20, 50, 80)):
        for g in 'ABC':
            s = ok[(ok['group'] == g) & (ok['gap'] == gp)]
            if not len(s):
                continue
            cur = np.array([[np.nan if v is None else v for v in c] for c in s['curve']], float)
            ax[i].plot(BINS + 12.5, np.nanmedian(cur, 0), color=colors[g], lw=2, label=f'{g} (n={len(s)})')
        ax[i].set(title=f'gap {gp} ms', xlabel='rPT (ms)', ylim=(0, 1))
        ax[i].axhline(0.5, color='0.7', lw=0.8)
    ax[0].set_ylabel('P(choose rich) on conflict trials')
    ax[0].legend(frameon=False, fontsize=8)
    for g in 'ABC':
        s = ok[ok['group'] == g]
        dts = sorted({int(k) for d in s['div_by_dt'] for k in d})
        med = [np.nanmedian([d.get(str(t), np.nan) for d in s['div_by_dt']]) for t in dts]
        ax[3].plot(dts, med, 'o-', color=colors[g], lw=2, label=g)
    ax[3].set(title='rich-salient divergence', xlabel='dt = target - go (ms)', ylabel='ms after target onset')
    ax[3].legend(frameon=False, fontsize=8)
    for a in ax:
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'tachometric.png', dpi=150)


if __name__ == '__main__':
    main()
