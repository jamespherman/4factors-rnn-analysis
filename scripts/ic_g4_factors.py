#!/usr/bin/env python3
"""
Phase 1 / G4 (plans/2026-10-31_intercollicular_plan.md): factor inheritance.

For each significant response (unit x location x window, p < 0.01 in G1; windows visual and presac) the signed
ROC selectivity for reward (high vs low), salience (high vs low bullseye), identity (face vs non-face) and
probability (high vs low) is computed on that location's trials. Aligned selectivity = sign(response) x selectivity:
positive when the factor makes the response larger in magnitude (more activation or deeper suppression).
Compared: classic neurons at their preferred contralateral location (what the opposite SC's classic neurons carry
at the mirrored location, by symmetry) vs ipsi-location responses vs the same units' contralateral responses.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from src.sessions import ALL_SESSIONS  # noqa: E402
from src.intercollicular import session_table, OPPOSITE  # noqa: E402
from compare_selectivity_by_type import roc_with_null  # noqa: E402

OUT = Path(__file__).parent.parent / 'results' / 'intercollicular' / 'g4_factors'
G1 = Path(__file__).parent.parent / 'results' / 'intercollicular' / 'g1_geometry' / 'rows.npy'
FACTORS = ['reward', 'salience', 'identity', 'probability']


def factor_masks(t):
    return {
        'reward': (t['reward'] == 0, t['reward'] == 1),
        'salience': ((t['identity'] == 3) & (t['salience'] == 2), (t['identity'] == 3) & (t['salience'] == 1)),
        'identity': (t['identity'] == 2, t['identity'] == 1),
        'probability': (t['probability'] == 0, t['probability'] == 1),
    }


def main():
    g1 = {(r['session'], r['id']): r for r in np.load(G1, allow_pickle=True)}
    rows = []
    for s in ALL_SESSIONS:
        if s == '08_15':
            continue
        t = session_table(s)
        fm = factor_masks(t)
        contra = [k + 1 for k in range(4) if t['hemi'][k] == 'contra']
        for j, i in enumerate(t['ids']):
            r1 = g1[(s, i)]
            P = r1['P']
            cn = [k for k in contra if k != P][0]
            inb = [k for k in range(1, 5) if t['hemi'][k - 1] == 'ipsi' and t['elev'][k - 1] == t['elev'][P - 1]][0]
            for role, k in [('P', P), ('cn', cn), ('in', inb), ('op', OPPOSITE[P])]:
                for w in ('visual', 'presac'):
                    if not r1[f'{w}_{role}_p'] < 0.01:
                        continue
                    x = t['rates'][w][:, j] - np.nanmean(t['rates']['baseline'][:, j])
                    row = {'session': s, 'id': i, 'type': t['type'][j], 'role': role, 'win': w,
                           'sign': np.sign(r1[f'{w}_{role}'])}
                    for f in FACTORS:
                        m1, m2 = fm[f]
                        loc = t['loc'] == k
                        sel, pv = roc_with_null(x, m1 & loc, m2 & loc, 200)
                        row[f] = sel
                        row[f'{f}_p'] = pv
                    rows.append(row)
        print(s, 'done', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / 'rows.npy', rows, allow_pickle=True)
    report(rows)


def report(rows):
    L = []
    p = L.append
    p('# G4 factor inheritance: aligned selectivity = sign(response) x ROC selectivity (2*(AUC-0.5))\n')
    p('Positive = the factor (high reward, high salience, face, high probability) makes the response larger in '
      'magnitude. Cells: median [fraction p<0.05] (n).\n')
    groups = [
        ('classic, preferred contra (activations)', lambda r: r['type'] == 'E' and r['role'] == 'P' and r['sign'] > 0),
        ('non-classic, contra activations', lambda r: r['type'] == 'I' and r['role'] in ('P', 'cn') and r['sign'] > 0),
        ('non-classic, contra suppressions', lambda r: r['type'] == 'I' and r['role'] in ('P', 'cn') and r['sign'] < 0),
        ('non-classic, ipsi activations', lambda r: r['type'] == 'I' and r['role'] in ('in', 'op') and r['sign'] > 0),
        ('non-classic, ipsi suppressions', lambda r: r['type'] == 'I' and r['role'] in ('in', 'op') and r['sign'] < 0),
        ('classic, ipsi responses', lambda r: r['type'] == 'E' and r['role'] in ('in', 'op')),
    ]
    for w in ('visual', 'presac'):
        p(f'## {w} window')
        p('| group | ' + ' | '.join(FACTORS) + ' |')
        p('|---|' + '---|' * len(FACTORS))
        vals = {}
        for name, sel in groups:
            g = [r for r in rows if r['win'] == w and sel(r)]
            cells = []
            for f in FACTORS:
                v = np.array([r['sign'] * r[f] for r in g if not np.isnan(r[f])])
                pv = np.array([r[f'{f}_p'] for r in g if not np.isnan(r[f])])
                vals[(name, f)] = v
                cells.append(f'{np.median(v):+.3f} [{np.mean(pv < 0.05):.0%}] ({len(v)})' if len(v) else '-')
            p(f'| {name} | ' + ' | '.join(cells) + ' |')
        p('')
        for f in FACTORS:
            a = vals[('non-classic, ipsi activations', f)]
            b = vals[('non-classic, contra activations', f)]
            c = vals[('non-classic, ipsi suppressions', f)]
            d = vals[('non-classic, contra suppressions', f)]
            s1 = f'ipsi vs contra activations p={mannwhitneyu(a, b).pvalue:.2g}' if len(a) > 4 and len(b) > 4 else ''
            s2 = f'ipsi vs contra suppressions p={mannwhitneyu(c, d).pvalue:.2g}' if len(c) > 4 and len(d) > 4 else ''
            s3 = f'ipsi activations vs 0 p={wilcoxon(a).pvalue:.2g}' if len(a) > 4 else ''
            s4 = f'ipsi suppressions vs 0 p={wilcoxon(c).pvalue:.2g}' if len(c) > 4 else ''
            p(f'- {f}: {s1}; {s2}; {s3}; {s4}')
        p('')
    text = '\n'.join(L)
    print(text)
    (OUT / 'summary.md').write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
