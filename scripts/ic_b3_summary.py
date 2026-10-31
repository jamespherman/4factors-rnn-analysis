#!/usr/bin/env python3
"""
Summary of the within-SC CCGs (scripts/ic_b3_ccg.py): short-latency peaks and troughs by pre/post class,
normalised by per-pair power, with distance, and the planned contrast of E->I peaks onto ipsi-responsive vs other
non-classic units (ipsi-responsive from G1: significant response at an ipsi location, visual or pre-saccadic).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, binomtest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sessions import ALL_SESSIONS  # noqa: E402

RES = Path(__file__).parent.parent / 'results' / 'intercollicular'
P_SIG = 0.001
REF_STG = 0.01      # a typical detectable E->I gain (English et al. 2017 report gains of ~0.01-0.1)


def main():
    rows = [r for s in ALL_SESSIONS for r in np.load(RES / 'b3_ccg' / f'{s}.npy', allow_pickle=True)]
    rows = [r for r in rows if r['coincident_frac'] <= 0.1]
    g1 = {(r['session'], r['id']): r for r in np.load(RES / 'g1_geometry' / 'rows.npy', allow_pickle=True)}

    def ipsi_resp(s, i):
        r = g1[(s, i)]
        if r['meridian']:
            keys = [f'{w}_loc{k}' for w in ('visual', 'presac') for k in (2, 3, 4)]
        else:
            keys = [f'{w}_{k}' for w in ('visual', 'presac') for k in ('in', 'op')]
        return any(r[f'{k}_p'] < 0.01 for k in keys)

    lags = np.arange(-50, 51)
    cw, aw, sh = (lags >= 1) & (lags <= 4), (lags >= -4) & (lags <= -1), (np.abs(lags) >= 6) & (np.abs(lags) <= 10)
    for r in rows:
        o, e = r['ccg_obs'].astype(float), r['ccg_exp']
        c, a = o[cw].sum(), o[aw].sum()
        # sharp: causal window differs from the anticausal window (binomial) and its jitter-corrected excess
        # exceeds the mean excess per bin of the 6-10 ms shoulders by the same sign
        shoulder = ((o - e)[sh].mean()) * cw.sum()
        excess = (o - e)[cw].sum()
        asym_up = binomtest(int(c), int(c + a), alternative='greater').pvalue < P_SIG if c + a else False
        asym_dn = binomtest(int(c), int(c + a), alternative='less').pvalue < P_SIG if c + a else False
        r['peak'] = r['p_excess'] < P_SIG and asym_up and excess > shoulder
        r['trough'] = r['p_deficit'] < P_SIG and asym_dn and excess < shoulder
        r['broad'] = r['p_excess'] < P_SIG and not r['peak']
        r['powered'] = r['min_detectable_stg'] <= REF_STG
        r['post_ipsi'] = ipsi_resp(r['session'], r['post'])
    L = []
    p = L.append
    p(f'# B3 within-SC CCGs: {len(rows)} directed pairs (different channels, coincidence <= 0.1)\n')
    p(f'Peak/trough: Poisson p < {P_SIG} in 1-4 ms vs jitter, 1-4 ms vs -4..-1 ms asymmetric (binomial p < {P_SIG}) and sharper than the 6-10 ms shoulders. Broad = jitter-significant but not sharp (fast common input). Powered: min detectable STG <= {REF_STG}.\n')
    p('| pre -> post | pairs | powered | peaks (all / powered) | troughs (all / powered) | median STG of peaks |')
    p('|---|---|---|---|---|---|')
    for pre in 'EI':
        for post in 'EI':
            g = [r for r in rows if r['pre_type'] == pre and r['post_type'] == post]
            pw = [r for r in g if r['powered']]
            pk = [r for r in g if r['peak']]
            p(f'| {pre} -> {post} | {len(g)} | {len(pw)} | {len(pk)} / {sum(r["peak"] for r in pw)} '
              f'({sum(r["peak"] for r in pw) / max(len(pw), 1):.1%}) | {sum(r["trough"] for r in g)} / '
              f'{sum(r["trough"] for r in pw)} | {np.median([r["stg"] for r in pk]) if pk else float("nan"):.3f} |')
    p('\n## Distance dependence (powered pairs, all classes)')
    for lo, hi in [(0, 100), (100, 200), (200, 400), (400, 2000)]:
        g = [r for r in rows if r['powered'] and lo <= r['dist'] < hi]
        p(f'- {lo}-{hi} um: {len(g)} pairs, sharp peaks {np.mean([r["peak"] for r in g]) if g else np.nan:.1%}, broad {np.mean([r["broad"] for r in g]) if g else np.nan:.1%}, '
          f'troughs {np.mean([r["trough"] for r in g]) if g else np.nan:.1%}')
    p('\n## Planned contrast: classic -> non-classic peaks, by whether the non-classic target responds at ipsi locations')
    g = [r for r in rows if r['pre_type'] == 'E' and r['post_type'] == 'I' and r['powered']]
    a = [r for r in g if r['post_ipsi']]
    b = [r for r in g if not r['post_ipsi']]
    tab = [[sum(r['peak'] for r in a), len(a) - sum(r['peak'] for r in a)],
           [sum(r['peak'] for r in b), len(b) - sum(r['peak'] for r in b)]]
    p(f'- ipsi-responsive targets: {tab[0][0]}/{len(a)} ({tab[0][0] / max(len(a), 1):.1%}); '
      f'others: {tab[1][0]}/{len(b)} ({tab[1][0] / max(len(b), 1):.1%}); Fisher p={fisher_exact(tab).pvalue:.2g}')
    rp = np.array([r['n_pre'] for r in g]); rq = np.array([r['n_post'] for r in g])
    p(f'- median spikes (pre, post): ipsi-responsive targets {np.median([r["n_pre"] for r in a]):.0f}, '
      f'{np.median([r["n_post"] for r in a]):.0f}; others {np.median([r["n_pre"] for r in b]):.0f}, '
      f'{np.median([r["n_post"] for r in b]):.0f}; median min-detectable STG {np.median([r["min_detectable_stg"] for r in a]):.4f} vs '
      f'{np.median([r["min_detectable_stg"] for r in b]):.4f}')
    ua = {(r['session'], r['post']) for r in a if r['peak']}
    ub = {(r['session'], r['post']) for r in b if r['peak']}
    na = {(r['session'], r['post']) for r in a}
    nb = {(r['session'], r['post']) for r in b}
    p(f'- units receiving >= 1 classic peak: ipsi-responsive {len(ua)}/{len(na)}, others {len(ub)}/{len(nb)}')
    p('\n## Troughs (listed; not interpreted)')
    for r in rows:
        if r['trough']:
            p(f'- {r["session"]} {r["pre"]}({r["pre_type"]}) -> {r["post"]}({r["post_type"]}): STG {r["stg"]:+.4f}, '
              f'p {r["p_deficit"]:.1g}, dist {r["dist"]:.0f} um')
    text = '\n'.join(L)
    print(text)
    (RES / 'b3_ccg' / 'summary.md').write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
