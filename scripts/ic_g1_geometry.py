#!/usr/bin/env python3
"""
Phase 1 / G1 (plans/2026-10-31_intercollicular_plan.md): distance-matched geometry.

For each neuron, P = the contralateral location with the largest evoked response (visual or pre-saccadic).
The two 90-deg neighbours of P are one contralateral location (across the horizontal meridian; contra-neighbour)
and one ipsilateral location at the same elevation (across the vertical meridian; ipsi-neighbour); the 180-deg
opposite is ipsilateral at the other elevation. Evoked = window rate minus the same trials' pre-target baseline
(paired t-test per location, p < 0.01). Newton 08_15 (targets on the meridians) is reported separately.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon, binomtest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sessions import ALL_SESSIONS  # noqa: E402
from src.intercollicular import session_table, OPPOSITE  # noqa: E402

OUT = Path(__file__).parent.parent / 'results' / 'intercollicular' / 'g1_geometry'
WINS = ['visual', 'presac', 'postsac']
ALPHA = 0.01


def evoked(t, j, w, k):
    """Mean evoked rate and paired-test p at export location k for unit j, window w."""
    m = (t['loc'] == k) & ~np.isnan(t['rates'][w][:, j]) & ~np.isnan(t['rates']['baseline'][:, j])
    a, b = t['rates'][w][m, j], t['rates']['baseline'][m, j]
    if m.sum() < 8:
        return np.nan, np.nan
    return float(np.mean(a - b)), float(ttest_rel(a, b).pvalue) if np.std(a - b) > 0 else 1.0


def main():
    rows = []
    for s in ALL_SESSIONS:
        t = session_table(s)
        contra = [k + 1 for k in range(4) if t['hemi'][k] == 'contra']
        for j in range(len(t['ids'])):
            ev = {(w, k): evoked(t, j, w, k) for w in WINS for k in range(1, 5)}
            P = max(contra, key=lambda k: max(ev[('visual', k)][0], ev[('presac', k)][0]))
            row = {'session': s, 'monkey': t['monkey'], 'type': t['type'][j], 'id': t['ids'][j], 'P': P,
                   'depth': t['depth'][j], 'meridian': s == '08_15'}
            if s != '08_15':
                cn = [k for k in contra if k != P][0]
                inb = [k for k in range(1, 5) if t['hemi'][k - 1] == 'ipsi' and t['elev'][k - 1] == t['elev'][P - 1]][0]
                op = OPPOSITE[P]
                assert inb != op
                for w in WINS:
                    for name, k in [('P', P), ('cn', cn), ('in', inb), ('op', op)]:
                        row[f'{w}_{name}'], row[f'{w}_{name}_p'] = ev[(w, k)]
            else:
                for w in WINS:
                    for name, k in [('P', P)] + [(f'loc{k}', k) for k in range(1, 5)]:
                        row[f'{w}_{name}'], row[f'{w}_{name}_p'] = ev[(w, k)]
            # pause-type: significant suppression at >= 3 of 4 locations in pre/post-saccadic windows
            row['pause'] = sum(any(ev[(w, k)][0] < 0 and ev[(w, k)][1] < ALPHA for w in ('presac', 'postsac'))
                               for k in range(1, 5)) >= 3
            rows.append(row)
    np.save(OUT / 'rows.npy', rows, allow_pickle=True) if OUT.exists() else (OUT.mkdir(parents=True), np.save(OUT / 'rows.npy', rows, allow_pickle=True))
    report(rows)


def report(rows):
    L = []
    p = L.append
    p('# G1 distance-matched geometry (p < 0.01 per location, paired vs pre-target baseline)\n')
    p(f'pause-type units: E {sum(r["pause"] for r in rows if r["type"] == "E")}, '
      f'I {sum(r["pause"] for r in rows if r["type"] == "I")}\n')
    diag = [r for r in rows if not r['meridian']]
    for w in WINS:
        p(f'## {w} window')
        for t in ['E', 'I']:
            for m in ['Feynman', 'Newton', 'both']:
                g = [r for r in diag if r['type'] == t and (m == 'both' or r['monkey'] == m) and not np.isnan(r[f'{w}_P'])]
                if not g:
                    continue
                sup = {k: np.array([r[f'{w}_{k}'] < 0 and r[f'{w}_{k}_p'] < ALPHA for r in g]) for k in ('cn', 'in', 'op')}
                act = {k: np.array([r[f'{w}_{k}'] > 0 and r[f'{w}_{k}_p'] < ALPHA for r in g]) for k in ('P', 'cn', 'in', 'op')}
                # McNemar-style exact test on discordant pairs: suppression at ipsi- vs contra-neighbour
                b, c = int((sup['in'] & ~sup['cn']).sum()), int((~sup['in'] & sup['cn']).sum())
                pm = binomtest(b, b + c).pvalue if b + c else np.nan
                b2, c2 = int((sup['op'] & ~sup['in']).sum()), int((~sup['op'] & sup['in']).sum())
                pm2 = binomtest(b2, b2 + c2).pvalue if b2 + c2 else np.nan
                p(f'- {t} {m:7s} n={len(g):3d} | activated: P {act["P"].mean():.0%}, contra-nb {act["cn"].mean():.0%}, '
                  f'ipsi-nb {act["in"].mean():.0%}, opposite {act["op"].mean():.0%} | suppressed: contra-nb '
                  f'{sup["cn"].mean():.0%}, ipsi-nb {sup["in"].mean():.0%}, opposite {sup["op"].mean():.0%} | '
                  f'supp ipsi-nb vs contra-nb {b}:{c} p={pm:.2g}; opposite vs ipsi-nb {b2}:{c2} p={pm2:.2g}')
                # normalised activation at the two equidistant neighbours, for units activated at P
                a = [r for r in g if r[f'{w}_P'] > 0 and r[f'{w}_P_p'] < ALPHA]
                if len(a) >= 5:
                    cn = np.array([r[f'{w}_cn'] / r[f'{w}_P'] for r in a])
                    inn = np.array([r[f'{w}_in'] / r[f'{w}_P'] for r in a])
                    op = np.array([r[f'{w}_op'] / r[f'{w}_P'] for r in a])
                    pw = wilcoxon(cn, inn).pvalue
                    p(f'    activated at P (n={len(a)}): response / P median: contra-nb {np.median(cn):+.2f}, '
                      f'ipsi-nb {np.median(inn):+.2f}, opposite {np.median(op):+.2f}; contra-nb vs ipsi-nb p={pw:.2g}')
        p('')
    mer = [r for r in rows if r['meridian']]
    p('## Newton 08_15 (180 contra, 90/270 vertical meridian, 360 ipsi): fraction suppressed / activated per location')
    for w in WINS:
        for t in ['E', 'I']:
            g = [r for r in mer if r['type'] == t]
            s = ', '.join(f'loc{k} sup {np.mean([r[f"{w}_loc{k}"] < 0 and r[f"{w}_loc{k}_p"] < ALPHA for r in g]):.0%} '
                          f'act {np.mean([r[f"{w}_loc{k}"] > 0 and r[f"{w}_loc{k}_p"] < ALPHA for r in g]):.0%}'
                          for k in range(1, 5))
            p(f'- {w} {t} n={len(g)}: {s}')
    text = '\n'.join(L)
    print(text)
    (OUT / 'summary.md').write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
