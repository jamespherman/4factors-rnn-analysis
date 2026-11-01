#!/usr/bin/env python3
"""
Build the B10 panel data (candidate monosynaptic connections vs factor selectivity) into
results/aim1_panels/B10_ccg_pair_selectivity/.

Re-derives, from the saved CCGs (results/intercollicular/b3_ccg/*.npy) and the recorded visual-epoch ROC
selectivity (results/selectivity_by_type/grid_v13_free_free.npz), exactly the pair sets and similarity measure of
scripts/ic_b5_pair_selectivity.py (powered pairs, coincidence <= 0.1, sharp 1-4 ms peak / trough flags, Pearson r
across factors of the two units' signed selectivity vectors, session- and distance-matched (+-100 um) unconnected
null). Adds: the without-location similarity (identity, salience, reward, probability), which was quoted in
SUMMARY.md but not saved, and unit-level resampling (pairs share units: one presynaptic unit appears in up to 14
pairs).

Cheap: about 1-2 min single-threaded. Run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python scripts/aim1_panels/build_B10.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, poisson

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.sessions import ALL_SESSIONS, session_name  # noqa: E402

RES = ROOT / 'results'
OUT = RES / 'aim1_panels' / 'B10_ccg_pair_selectivity'
CACHE = ROOT / 'data' / 'spike_cache'
FACTORS = ['location', 'identity', 'salience', 'reward', 'probability']
P_SIG = 0.001
POWER_STG = 0.01
N_NULL = 2000
N_BOOT = 5000
LAGS = np.arange(-50, 51)
EXAMPLE = ('08_15', 41, 67)   # narrow 1-ms peak on a flat baseline; chosen by eye among the sharpest (see README)


def flags(r):
    """Sharp-peak / trough flags, identical to scripts/ic_b5_pair_selectivity.py::flags."""
    cw, aw = (LAGS >= 1) & (LAGS <= 4), (LAGS >= -4) & (LAGS <= -1)
    sh = (np.abs(LAGS) >= 6) & (np.abs(LAGS) <= 10)
    o, e = r['ccg_obs'].astype(float), r['ccg_exp']
    c, a = o[cw].sum(), o[aw].sum()
    shoulder = (o - e)[sh].mean() * cw.sum()
    excess = (o - e)[cw].sum()
    up = c + a > 0 and binomtest(int(c), int(c + a), alternative='greater').pvalue < P_SIG
    dn = c + a > 0 and binomtest(int(c), int(c + a), alternative='less').pvalue < P_SIG
    return (r['p_excess'] < P_SIG and up and excess > shoulder), (r['p_deficit'] < P_SIG and dn and excess < shoulder)


def corr(a, b, min_n):
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() >= min_n and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
        return float(np.corrcoef(a[ok], b[ok])[0, 1])
    return np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = np.load(RES / 'selectivity_by_type' / 'grid_v13_free_free.npz')
    key = {}
    for n, s in enumerate(sel['session']):
        key.setdefault(s, []).append(n)
    idx = {}
    for s in ALL_SESSIONS:
        ids = np.load(CACHE / f'{session_name(s)}.npz')['ids']
        assert len(ids) == len(key[s])
        for k, i in enumerate(ids):
            idx[(s, int(i))] = key[s][k]
    rec = np.stack([sel[f'recorded_{f}'][:, 0] for f in FACTORS], 1)
    recp = np.stack([sel[f'recorded_{f}'][:, 1] for f in FACTORS], 1)

    all_rows = [r for s in ALL_SESSIONS for r in np.load(RES / 'intercollicular' / 'b3_ccg' / f'{s}.npy',
                                                            allow_pickle=True) if r['coincident_frac'] <= 0.1]
    pairs = []
    example = None
    for r in all_rows:
        pk, tr = flags(r)
        powered = r['min_detectable_stg'] <= POWER_STG
        a, b = rec[idx[(r['session'], r['pre'])]], rec[idx[(r['session'], r['post'])]]
        pairs.append({'session': r['session'], 'pre_id': r['pre'], 'post_id': r['post'], 'pre_type': r['pre_type'],
                      'post_type': r['post_type'], 'distance_um': float(r['dist']), 'stg': float(r['stg']),
                      'min_detectable_stg': float(r['min_detectable_stg']), 'powered': powered,
                      'peak': pk, 'trough': tr,
                      'r_5factor': corr(a, b, 4), 'r_no_location': corr(a[1:], b[1:], 4)})
        if (r['session'], r['pre'], r['post']) == EXAMPLE:
            example = r
    P = pd.DataFrame(pairs)
    pw = P[P.powered].reset_index(drop=True)
    conn = pw[pw.peak].reset_index(drop=True)
    trough = pw[pw.trough].reset_index(drop=True)
    unconn = pw[~pw.peak & ~pw.trough].reset_index(drop=True)
    troughs_all = P[P.trough]
    print('pairs', len(P), 'powered', len(pw), 'peaks', len(conn), 'troughs powered', len(trough),
          'troughs all', len(troughs_all))

    rng = np.random.default_rng(0)
    # matched pools (same session, |distance difference| <= 100 um), as in ic_b5_pair_selectivity.py
    us, ud = unconn.session.values, unconn.distance_um.values

    def pools(g):
        return [np.flatnonzero((us == s) & (np.abs(ud - d) <= 100)) for s, d in zip(g.session, g.distance_um)]

    stats, null_rows, hist_rows = [], [], []
    for gname, g in [('peak', conn), ('trough', trough)]:
        pl = pools(g)
        assert all(len(x) for x in pl)
        for measure in ('r_no_location', 'r_5factor'):
            obs = g[measure].values
            obs_med = np.nanmedian(obs)
            u_vals = unconn[measure].values
            # pair-level matched null (the original test): median similarity of one matched draw per pair
            draws = np.stack([np.array([x[rng.integers(len(x))] for x in pl]) for _ in range(N_NULL)])
            null_med = np.nanmedian(u_vals[draws], axis=1)
            p_pair = float(np.mean(null_med >= obs_med))
            # unit-level: aggregate to one value per presynaptic unit (median over its pairs), then median over units;
            # the null keeps the same pair-to-unit structure (same draws, same grouping)
            grp = (g.session + '|' + g.pre_id.astype(str)).values
            ug = {u: np.flatnonzero(grp == u) for u in np.unique(grp)}

            def unit_median(v):
                return np.nanmedian([np.nanmedian(v[ii]) if np.any(~np.isnan(v[ii])) else np.nan for ii in ug.values()])
            obs_unit = unit_median(obs)
            null_unit = np.array([unit_median(u_vals[dr]) for dr in draws])
            p_unit = float(np.mean(null_unit >= obs_unit))
            # unit-level cluster bootstrap CI of the observed pair median (resample presynaptic units)
            keys = list(ug)
            boot = []
            for _ in range(N_BOOT):
                pick = rng.integers(len(keys), size=len(keys))
                ii = np.concatenate([ug[keys[k]] for k in pick])
                boot.append(np.nanmedian(obs[ii]))
            # also cluster bootstrap on the difference observed - matched (draw a fresh matched sample per resample)
            diff = []
            for _ in range(N_BOOT):
                pick = rng.integers(len(keys), size=len(keys))
                ii = np.concatenate([ug[keys[k]] for k in pick])
                m = np.array([pl[i][rng.integers(len(pl[i]))] for i in ii])
                diff.append(np.nanmedian(obs[ii]) - np.nanmedian(u_vals[m]))
            ci = np.percentile(boot, [2.5, 97.5])
            dci = np.percentile(diff, [2.5, 97.5])
            n_obs = int(np.sum(~np.isnan(obs)))
            base = dict(window='visual epoch 50-250 ms after target onset (ROC selectivity); CCG over whole stable '
                               'span incl. ITIs', unit_class='all pre/post classes' if gname == 'peak' else 'I->I',
                        session_filter='7 unilateral sessions (Feynman 08_12-08_15, Newton 08_13-08_15), '
                                       'verified trials; powered pairs (min detectable STG <= 0.01)')
            lab = 'without location (identity, salience, reward, probability)' if measure == 'r_no_location' \
                else 'with location (5 factors)'
            stats += [
                dict(statistic=f'{gname} pairs: median profile r, {lab}', value=round(obs_med, 3), n=n_obs,
                     test='median; CI = unit-level cluster bootstrap over presynaptic units',
                     ci_low=round(ci[0], 3), ci_high=round(ci[1], 3), n_units=len(keys), **base),
                dict(statistic=f'matched unconnected for {gname} pairs: median of replicate medians, profile r, {lab}',
                     value=round(float(np.median(null_med)), 3), n=N_NULL,
                     test='session- and distance-matched (+-100 um) unconnected powered pairs, one draw per '
                          'connected pair per replicate; CI = 2.5-97.5 percentile of replicate medians',
                     ci_low=round(float(np.percentile(null_med, 2.5)), 3),
                     ci_high=round(float(np.percentile(null_med, 97.5)), 3), n_units=np.nan, **base),
                dict(statistic=f'{gname} vs matched: p (pair level), {lab}', value=round(p_pair, 4), n=n_obs,
                     test=f'one-sided, fraction of {N_NULL} matched replicates with median >= observed',
                     ci_low=np.nan, ci_high=np.nan, n_units=len(keys), **base),
                dict(statistic=f'{gname}: median over presynaptic units of per-unit median r, {lab}',
                     value=round(obs_unit, 3), n=len(keys), test='per presynaptic unit median, then median',
                     ci_low=np.nan, ci_high=np.nan, n_units=len(keys), **base),
                dict(statistic=f'matched unconnected for {gname} pairs: median of unit-aggregated medians, {lab}',
                     value=round(float(np.median(null_unit)), 3), n=N_NULL,
                     test='same matched draws aggregated per presynaptic unit of the connected pair',
                     ci_low=round(float(np.percentile(null_unit, 2.5)), 3),
                     ci_high=round(float(np.percentile(null_unit, 97.5)), 3), n_units=len(keys), **base),
                dict(statistic=f'{gname} vs matched: p (unit level), {lab}', value=round(p_unit, 4), n=len(keys),
                     test=f'one-sided, fraction of {N_NULL} unit-aggregated matched replicates >= observed',
                     ci_low=np.nan, ci_high=np.nan, n_units=len(keys), **base),
                dict(statistic=f'{gname} minus matched: difference of medians, {lab}',
                     value=round(obs_med - float(np.median(null_med)), 3), n=n_obs,
                     test='CI = unit-level cluster bootstrap (presynaptic units) with a fresh matched draw per '
                          'resample', ci_low=round(dci[0], 3), ci_high=round(dci[1], 3), n_units=len(keys), **base),
            ]
            for v in null_med:
                null_rows.append({'pair_set': gname, 'measure': measure, 'level': 'pair', 'null_median_r': v})
            for v in null_unit:
                null_rows.append({'pair_set': gname, 'measure': measure, 'level': 'unit', 'null_median_r': v})
            if gname == 'peak':
                # matched unconnected distribution for the histogram: exact mixture of the matched pools
                # (each connected pair's pool weighted 1 / pool size, so the weights sum to n connected pairs)
                for i, x in enumerate(pl):
                    v = u_vals[x]
                    ok = ~np.isnan(v)
                    for val in v[ok]:
                        hist_rows.append({'measure': measure, 'r': val, 'weight': 1.0 / ok.sum(),
                                          'matched_to_pair': i})
    # per-pair tables
    keep = ['session', 'pre_id', 'pre_type', 'post_id', 'post_type', 'distance_um', 'stg', 'min_detectable_stg',
            'r_no_location', 'r_5factor']
    conn_out = conn[keep].copy()
    conn_out.insert(0, 'pair_kind', 'sharp_peak')
    tr_out = troughs_all[keep + ['powered']].copy()
    tr_out.insert(0, 'pair_kind', 'trough')
    conn_out['powered'] = True
    pd.concat([conn_out, tr_out]).to_csv(OUT / 'B10_pairs_connected.csv', index=False, float_format='%.5g')
    pd.DataFrame(hist_rows).to_csv(OUT / 'B10_matched_unconnected.csv', index=False, float_format='%.5g')
    pd.DataFrame(null_rows).to_csv(OUT / 'B10_null_medians.csv', index=False, float_format='%.5g')

    # example CCG with pointwise Poisson band around the (analytic) jitter expectation
    assert example is not None
    e = example['ccg_exp'].astype(float)
    pd.DataFrame({'lag_ms': LAGS, 'observed_count': example['ccg_obs'], 'jitter_expected_count': e,
                  'poisson_lo_0.05pct': poisson.ppf(0.0005, e), 'poisson_hi_99.95pct': poisson.ppf(0.9995, e),
                  'session': EXAMPLE[0], 'pre_id': EXAMPLE[1], 'post_id': EXAMPLE[2],
                  'pre_type': example['pre_type'], 'post_type': example['post_type'],
                  'n_pre_spikes': example['n_pre'], 'stg_1to4ms': example['stg']}).to_csv(
        OUT / 'B10_example_ccg.csv', index=False, float_format='%.5g')

    # counts
    cls = conn.pre_type + '->' + conn.post_type
    for c, n in cls.value_counts().sort_index().items():
        stats.append(dict(statistic=f'sharp-peak pairs {c}', value=n, n=n, test='count', ci_low=np.nan,
                          ci_high=np.nan, n_units=np.nan, window='CCG 1-4 ms', unit_class=c,
                          session_filter='powered pairs, 7 sessions'))
    for c in ['E->E', 'E->I', 'I->E', 'I->I']:
        g = pw[(pw.pre_type + '->' + pw.post_type) == c]
        stats.append(dict(statistic=f'sharp-peak rate among powered pairs {c} (%)', value=round(100 * g.peak.mean(), 2),
                          n=len(g), test='fraction', ci_low=np.nan, ci_high=np.nan, n_units=np.nan,
                          window='CCG 1-4 ms', unit_class=c, session_filter='powered pairs, 7 sessions'))
    units = set(zip(conn.session, conn.pre_id)) | set(zip(conn.session, conn.post_id))
    stats += [
        dict(statistic='sharp-peak pairs (powered)', value=len(conn), n=len(conn), test='count', ci_low=np.nan,
             ci_high=np.nan, n_units=len(units), window='CCG 1-4 ms', unit_class='all',
             session_filter='powered pairs, 7 sessions'),
        dict(statistic='max pairs sharing one presynaptic unit', value=int(conn.groupby(['session', 'pre_id']).size().max()),
             n=len(conn), test='count', ci_low=np.nan, ci_high=np.nan,
             n_units=conn.groupby(['session', 'pre_id']).ngroups, window='CCG 1-4 ms', unit_class='all',
             session_filter='powered pairs'),
        dict(statistic='troughs listed (all pairs)', value=len(troughs_all), n=len(troughs_all), test='count',
             ci_low=np.nan, ci_high=np.nan, n_units=np.nan, window='CCG 1-4 ms',
             unit_class=', '.join(f'{k} {v}' for k, v in (troughs_all.pre_type + '->' + troughs_all.post_type).value_counts().items()),
             session_filter='all pairs, coincidence <= 0.1'),
        dict(statistic='troughs in powered pairs', value=len(trough), n=len(trough), test='count', ci_low=np.nan,
             ci_high=np.nan, n_units=np.nan, window='CCG 1-4 ms',
             unit_class=', '.join(f'{k} {v}' for k, v in (trough.pre_type + '->' + trough.post_type).value_counts().items()),
             session_filter='powered pairs'),
        dict(statistic='powered directed pairs', value=len(pw), n=len(pw), test='count', ci_low=np.nan,
             ci_high=np.nan, n_units=np.nan, window='', unit_class='all', session_filter='7 sessions'),
    ]
    S = pd.DataFrame(stats)
    S.to_csv(OUT / 'stats.csv', index=False)
    print(S[['statistic', 'value', 'n', 'ci_low', 'ci_high']].to_string())
    shutil.copy(ROOT / 'scripts' / 'aim1_panels' / 'style.py', OUT / 'style.py')


if __name__ == '__main__':
    main()
