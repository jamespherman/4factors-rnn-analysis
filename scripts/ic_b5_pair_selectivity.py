#!/usr/bin/env python3
"""
Factor selectivity of candidate monosynaptically connected pairs (sharp 1-4 ms CCG peaks / troughs,
scripts/ic_b3_summary.py criteria) compared with powered, unconnected pairs.

Selectivity: visual-epoch ROC selectivity (2*(AUC-0.5)) at each neuron's preferred location from
results/selectivity_by_type/grid_v13_free_free.npz (recorded values; permutation p). Similarity of a pair =
Pearson r across the 5 factors of the two neurons' signed selectivity vectors, and per-factor sign agreement
among factors where both neurons are significant. Null: unconnected powered pairs from the same sessions,
resampled to match the connected pairs' session and distance distribution.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sessions import ALL_SESSIONS  # noqa: E402

RES = Path(__file__).parent.parent / 'results'
FACTORS = ['location', 'identity', 'salience', 'reward', 'probability']
P_SIG = 0.001


def flags(r, lags=np.arange(-50, 51)):
    cw, aw = (lags >= 1) & (lags <= 4), (lags >= -4) & (lags <= -1)
    sh = (np.abs(lags) >= 6) & (np.abs(lags) <= 10)
    o, e = r['ccg_obs'].astype(float), r['ccg_exp']
    c, a = o[cw].sum(), o[aw].sum()
    shoulder = (o - e)[sh].mean() * cw.sum()
    excess = (o - e)[cw].sum()
    up = c + a > 0 and binomtest(int(c), int(c + a), alternative='greater').pvalue < P_SIG
    dn = c + a > 0 and binomtest(int(c), int(c + a), alternative='less').pvalue < P_SIG
    return (r['p_excess'] < P_SIG and up and excess > shoulder), (r['p_deficit'] < P_SIG and dn and excess < shoulder)


def main():
    sel = np.load(RES / 'selectivity_by_type' / 'grid_v13_free_free.npz')
    key = {}
    for n, s in enumerate(sel['session']):
        key.setdefault(s, []).append(n)
    rows = [r for s in ALL_SESSIONS for r in np.load(RES / 'intercollicular' / 'b3_ccg' / f'{s}.npy', allow_pickle=True)
            if r['coincident_frac'] <= 0.1 and r['min_detectable_stg'] <= 0.01]
    from src.spikes import load_spikes
    idx = {}
    for s in ALL_SESSIONS:
        ids = load_spikes(s)['ids']
        for k, i in enumerate(ids):
            idx[(s, int(i))] = key[s][k]

    def vec(s, i):
        n = idx[(s, i)]
        return (np.array([sel[f'recorded_{f}'][n, 0] for f in FACTORS]),
                np.array([sel[f'recorded_{f}'][n, 1] for f in FACTORS]))

    for r in rows:
        r['peak'], r['trough'] = flags(r)
        r['v_pre'], r['p_pre'] = vec(r['session'], r['pre'])
        r['v_post'], r['p_post'] = vec(r['session'], r['post'])

    def sim(g):
        out = []
        for r in g:
            a, b = r['v_pre'], r['v_post']
            ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() >= 4 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                out.append(np.corrcoef(a[ok], b[ok])[0, 1])
        return np.array(out)

    def per_factor(g):
        res = {}
        for k, f in enumerate(FACTORS):
            both = [(r['v_pre'][k], r['v_post'][k]) for r in g
                    if r['p_pre'][k] < 0.05 and r['p_post'][k] < 0.05]
            res[f] = (sum(np.sign(a) == np.sign(b) for a, b in both), len(both))
        return res

    L = []
    p = L.append
    conn = [r for r in rows if r['peak']]
    trough = [r for r in rows if r['trough']]
    unconn = [r for r in rows if not r['peak'] and not r['trough']]
    rng = np.random.default_rng(0)
    # matched null: for each connected pair draw an unconnected pair from the same session within +-100 um distance
    def matched(g, n_rep=2000):
        pools = []
        for r in g:
            pool = [u for u in unconn if u['session'] == r['session'] and abs(u['dist'] - r['dist']) <= 100]
            pools.append(pool)
        sims = []
        for _ in range(n_rep):
            samp = [pl[rng.integers(len(pl))] for pl in pools if pl]
            sims.append(np.nanmedian(sim(samp)))
        return np.array(sims)

    p('# Factor selectivity of candidate monosynaptic pairs (visual-epoch ROC selectivity, recorded)\n')
    for name, g in [('sharp peaks (excitatory candidates)', conn), ('troughs (inhibitory candidates)', trough)]:
        s = sim(g)
        null = matched(g, 1000 if len(g) > 20 else 2000)
        p(f'## {name}: {len(g)} pairs, {len({(r["session"], r["pre"]) for r in g} | {(r["session"], r["post"]) for r in g})} units')
        p(f'- similarity of selectivity profiles (r across 5 factors): median {np.median(s):+.2f} (n={len(s)}); '
          f'session- and distance-matched unconnected pairs median {np.median(null):+.2f} '
          f'[95% {np.percentile(null, 2.5):+.2f}, {np.percentile(null, 97.5):+.2f}]; '
          f'p(matched >= observed) = {np.mean(null >= np.median(s)):.3f}')
        pf, pu = per_factor(g), per_factor(unconn)
        for f in FACTORS:
            a, n = pf[f]
            au, nu = pu[f]
            p(f'- {f}: both significant in {n} pairs; same sign {a}/{n}' +
              (f' ({a / n:.0%}); unconnected {au}/{nu} ({au / max(nu, 1):.0%})' if n else ''))
        by = {}
        for r in g:
            by.setdefault(f'{r["pre_type"]}->{r["post_type"]}', []).append(r)
        p('- by class: ' + ', '.join(f'{k} {len(v)}' for k, v in sorted(by.items())))
        p('')
    p('## Connected pairs, listed (selectivity: loc, identity, salience, reward, probability; * p<0.05)')
    for r in conn + trough:
        fmt = lambda v, pv: ' '.join(f'{x:+.2f}{"*" if q < 0.05 else " "}' if not np.isnan(x) else '  nan ' for x, q in zip(v, pv))
        p(f'- {"peak  " if r["peak"] else "trough"} {r["session"]} {r["pre"]}({r["pre_type"]}) -> {r["post"]}({r["post_type"]}) '
          f'STG {r["stg"]:+.3f} dist {r["dist"]:.0f} | pre [{fmt(r["v_pre"], r["p_pre"])}] post [{fmt(r["v_post"], r["p_post"])}]')
    text = '\n'.join(L)
    out = RES / 'intercollicular' / 'b5_pair_selectivity'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.md').write_text(text + '\n', encoding='utf-8')
    print('\n'.join(L[:40]))


if __name__ == '__main__':
    main()
