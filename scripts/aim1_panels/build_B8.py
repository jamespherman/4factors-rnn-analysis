#!/usr/bin/env python3
"""
Build the B8 panel data (ipsilateral-response latency vs a commissural relay) into
results/aim1_panels/B8_ipsi_latency_relay/.

Source: results/intercollicular/g23_timing/rows.npy ('lat' rows: one per unit x location role x stimulus class with
a significant visual activation, half-peak onset latency and amplitude; scripts/ic_g23_timing.py) and
results/intercollicular/g1_geometry/rows.npy (response rates). Reproduces ic_g23_timing.report() exactly
(calibration fit, within-unit pairs) and adds the bootstrap CIs, which were not saved:
  - pairs: same unit, same stimulus class (bull_hi / bull_lo / image), ipsi 90-deg neighbour ('in') vs the contra
    90-deg neighbour ('cn') if significant there, else the preferred location ('P'); n = 111 conditions, 41 units
  - excess = (lat_in - lat_contra) - slope * ln(amp_in / amp_contra), slope from the classic-contra calibration
  - CI of the median excess: unit-level cluster bootstrap (resample units with all their conditions), 10,000
    resamples, percentile; variants also saved (condition-level bootstrap; calibration slope re-fitted inside the
    bootstrap by resampling calibration units).
Cheap (< 1 min). Run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python scripts/aim1_panels/build_B8.py
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'results' / 'intercollicular'
OUT = ROOT / 'results' / 'aim1_panels' / 'B8_ipsi_latency_relay'
N_BOOT = 10000
RELAY_BAND = (10.0, 27.0)    # bilateral S1: planted relay read by the rise estimator (see README)
PLANTED_DELAYS = (2.0, 10.0)  # bilateral S1: synaptic delays planted in the simulated relay


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lat = np.load(RES / 'g23_timing' / 'rows.npy', allow_pickle=True).item()['lat']
    L = pd.DataFrame(lat)
    cal = L[(L.type == 'E') & L.role.isin(['P', 'cn'])]
    fit = linregress(np.log(cal.amp), cal.lat)
    slope, icpt = fit.slope, fit.intercept

    # within-unit pairs, exactly as ic_g23_timing.report()
    rows = []
    by = {}
    for r in lat:
        by.setdefault((r['session'], r['id'], r['cls']), {})[r['role']] = r
    for (s, i, c), d in by.items():
        if 'in' in d and ('cn' in d or 'P' in d):
            ref = 'cn' if 'cn' in d else 'P'
            cr = d[ref]
            dl = d['in']['lat'] - cr['lat']
            da = np.log(d['in']['amp'] / cr['amp'])
            rows.append({'session': s, 'unit_id': i, 'unit_type': d['in']['type'], 'stim_class': c,
                         'contra_reference': ref, 'lat_ipsi_ms': d['in']['lat'], 'lat_contra_ms': cr['lat'],
                         'amp_ipsi_sp_s': d['in']['amp'], 'amp_contra_sp_s': cr['amp'],
                         'lat_diff_ms': dl, 'ln_amp_ratio': da, 'expected_from_amp_ms': slope * da,
                         'excess_ms': dl - slope * da})
    P = pd.DataFrame(rows)
    P['unit_key'] = P.session + '|' + P.unit_id.astype(str)
    assert len(P) == 111
    ex = P.excess_ms.to_numpy()
    med = float(np.median(ex))

    rng = np.random.default_rng(0)
    units = P.unit_key.unique()
    uidx = {u: np.flatnonzero(P.unit_key.to_numpy() == u) for u in units}
    boot_unit = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.choice(units, len(units))
        boot_unit[b] = np.median(ex[np.concatenate([uidx[u] for u in pick])])
    boot_cond = np.array([np.median(ex[rng.integers(0, len(ex), len(ex))]) for _ in range(N_BOOT)])
    # propagate calibration uncertainty: resample calibration units and paired units independently
    cal_units = (cal.session + '|' + cal.id.astype(str)).to_numpy()
    cu = np.unique(cal_units)
    cidx = {u: np.flatnonzero(cal_units == u) for u in cu}
    calx, caly = np.log(cal.amp.to_numpy()), cal.lat.to_numpy()
    dl, da = P.lat_diff_ms.to_numpy(), P.ln_amp_ratio.to_numpy()
    boot_full = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ci_ = np.concatenate([cidx[u] for u in rng.choice(cu, len(cu))])
        sl = linregress(calx[ci_], caly[ci_]).slope
        ii = np.concatenate([uidx[u] for u in rng.choice(units, len(units))])
        boot_full[b] = np.median(dl[ii] - sl * da[ii])
    unit_means = P.groupby('unit_key').excess_ms.median()

    P.drop(columns='unit_key').to_csv(OUT / 'B8_pairs.csv', index=False, float_format='%.4f')
    cal_out = pd.DataFrame({'session': cal.session, 'unit_id': cal.id, 'unit_type': cal.type, 'location_role': cal.role,
                            'stim_class': cal.cls, 'amp_sp_s': cal.amp, 'lat_ms': cal.lat})
    cal_out.to_csv(OUT / 'B8_calibration.csv', index=False, float_format='%.4f')
    xs = np.linspace(np.log(cal.amp.min()), np.log(cal.amp.max()), 50)
    pd.DataFrame({'amp_sp_s': np.exp(xs), 'fit_lat_ms': icpt + slope * xs}).to_csv(
        OUT / 'B8_calibration_fit.csv', index=False, float_format='%.4f')
    ci_u = np.percentile(boot_unit, [2.5, 97.5])
    ci_c = np.percentile(boot_cond, [2.5, 97.5])
    ci_f = np.percentile(boot_full, [2.5, 97.5])
    pd.DataFrame([
        {'band': 'commissural_relay_prediction', 'low_ms': RELAY_BAND[0], 'high_ms': RELAY_BAND[1],
         'source': 'bilateral S1 planted-relay simulation, rise estimator, activation (Newton_SC_2026 '
                   'NOTE_2026-10-31_intercollicular_for_4factors.md)'},
        {'band': 'planted_synaptic_delay', 'low_ms': PLANTED_DELAYS[0], 'high_ms': PLANTED_DELAYS[1],
         'source': 'same simulation: true delays planted (2, 5, 10 ms)'},
        {'band': 'excess_median_CI_unit_bootstrap', 'low_ms': ci_u[0], 'high_ms': ci_u[1],
         'source': f'median {med:.2f} ms; {N_BOOT} unit-level cluster bootstrap resamples (41 units), seed 0'},
    ]).to_csv(OUT / 'B8_bands.csv', index=False, float_format='%.3f')

    g1 = [r for r in np.load(RES / 'g1_geometry' / 'rows.npy', allow_pickle=True)
          if r['type'] == 'I' and not r['meridian']]
    act = lambda k: np.mean([r[f'visual_{k}'] > 0 and r[f'visual_{k}_p'] < 0.01 for r in g1])
    sup = lambda k: np.mean([r[f'visual_{k}'] < 0 and r[f'visual_{k}_p'] < 0.01 for r in g1])

    W = 'visual activation onset, target-aligned 1-ms PSTH (Gaussian SD 5 ms), half-peak in 30-250 ms held 10 ms'
    SF = '6 diagonal-target unilateral sessions (Newton 08_15 excluded), verified trials, stable trials, >= 20 trials'
    st = [
        ('calibration slope (latency per ln amplitude)', slope, len(cal), 'OLS latency ~ ln(amp)', W, 'E (classic), contra (P, cn) conditions', SF, np.nan, np.nan),
        ('calibration intercept', icpt, len(cal), 'OLS', W, 'E contra', SF, np.nan, np.nan),
        ('calibration r', fit.rvalue, len(cal), 'Pearson', W, 'E contra', SF, np.nan, np.nan),
        ('calibration units', cal.session.str.cat(cal.id.astype(str)).nunique(), len(cal), 'count', W, 'E contra', SF, np.nan, np.nan),
        ('within-unit ipsi minus contra latency, median (ms)', float(np.median(dl)), len(P), f'Wilcoxon signed-rank p = {wilcoxon(dl).pvalue:.3f}', W, 'E and I', SF, np.nan, np.nan),
        ('expected from amplitude difference, median (ms)', float(np.median(slope * da)), len(P), 'median of slope x ln(amp ratio)', W, 'E and I', SF, np.nan, np.nan),
        ('relay excess, median (ms) [unit bootstrap CI]', med, len(P), f'Wilcoxon p = {wilcoxon(ex).pvalue:.3f}; CI unit-level cluster bootstrap, {N_BOOT}x', W, 'E and I', SF, ci_u[0], ci_u[1]),
        ('relay excess, median (ms) [condition bootstrap CI]', med, len(P), f'CI condition-level bootstrap, {N_BOOT}x (ignores repeated units)', W, 'E and I', SF, ci_c[0], ci_c[1]),
        ('relay excess, median (ms) [unit bootstrap incl. calibration refit]', med, len(P), f'CI: calibration units and paired units resampled, slope refit each time, {N_BOOT}x', W, 'E and I', SF, ci_f[0], ci_f[1]),
        ('relay excess, median of per-unit medians (ms)', float(unit_means.median()), len(unit_means), f'Wilcoxon p = {wilcoxon(unit_means).pvalue:.3f}', W, 'E and I', SF, np.nan, np.nan),
        ('paired conditions: units', len(units), len(P), 'count', W, f'E {P[P.unit_type == "E"].unit_key.nunique()} units / I {P[P.unit_type == "I"].unit_key.nunique()} units', SF, np.nan, np.nan),
        ('paired conditions by class: E', int((P.unit_type == 'E').sum()), len(P), 'count', W, 'E', SF, np.nan, np.nan),
        ('paired conditions by class: I', int((P.unit_type == 'I').sum()), len(P), 'count', W, 'I', SF, np.nan, np.nan),
        ('paired conditions with contra reference = preferred location (not contra neighbour)', int((P.contra_reference == 'P').sum()), len(P), 'count', W, 'E and I', SF, np.nan, np.nan),
        ('fraction of conditions with excess inside relay band 10-27 ms', float(np.mean((ex >= 10) & (ex <= 27))), len(P), 'fraction', W, 'E and I', SF, np.nan, np.nan),
        ('I units activated at ipsi 90-deg neighbour (%)', 100 * act('in'), len(g1), 'paired vs pre-target baseline p < 0.01 (g1_geometry summary.md, "## visual window", row "I both n=154", activated)', 'visual epoch', 'I (non-classic)', SF, np.nan, np.nan),
        ('I units activated at contra 90-deg neighbour (%)', 100 * act('cn'), len(g1), 'same block', 'visual epoch', 'I (non-classic)', SF, np.nan, np.nan),
        ('I units suppressed at ipsi 90-deg neighbour (%)', 100 * sup('in'), len(g1), 'same block, suppressed', 'visual epoch', 'I (non-classic)', SF, np.nan, np.nan),
        ('I units suppressed at contra 90-deg neighbour (%)', 100 * sup('cn'), len(g1), 'same block, suppressed', 'visual epoch', 'I (non-classic)', SF, np.nan, np.nan),
    ]
    S = pd.DataFrame(st, columns=['statistic', 'value', 'n', 'test', 'window', 'unit_class', 'session_filter',
                                  'ci_low', 'ci_high'])
    S.to_csv(OUT / 'stats.csv', index=False, float_format='%.6g')
    print(S[['statistic', 'value', 'n', 'ci_low', 'ci_high']].to_string())
    shutil.copy(ROOT / 'scripts' / 'aim1_panels' / 'style.py', OUT / 'style.py')


if __name__ == '__main__':
    main()
