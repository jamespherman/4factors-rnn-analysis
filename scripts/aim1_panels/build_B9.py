#!/usr/bin/env python3
"""Build the B9 data folder: ipsiversive activity in Y is not a delayed copy of the other colliculus (X), Newton bilateral
srsClassic recordings 08-19 and 08-20 (conflict single-target trials).

Panels
  a-d  per session x target side: X population (all units of the SC contralateral to the target, contraversive burst)
       and Y eligible units (ipsiversive activation), saccade-aligned, each scaled to its perisaccadic maximum, with the
       relay prediction = X shifted by the planted-relay read-out lag of S1 (+10 to +27 ms); rise-onset ticks
  e    pooled cross-SC CCG, 08-20 fixation (primary bound window; 473 pairs), fractional modulation per lag, with the CCG
       a sparse relay (N = 50 converging X neurons) would produce and the strongest relay the bound does not exclude
       (N ~ 170-500), both in measured units (x estimator recovery 0.40-0.55)

Sources (read-only):
  Newton_SC_2026/data/bilateral/intercollicular_latency.mat  (code/run_intercollicular_latency.m: traces, onsets, Delta)
  Newton_SC_2026/data/bilateral/crossmidline_coupling.mat    (code/run_crossmidline_coupling.m: 08-20 fixation CCG)
  Newton_SC_2026/data/bilateral/intercollicular_ccg_split.mat (code/run_intercollicular_ccg_split.m: pooled by Y group)
  _bilateral_source/onset_estimators_S1.csv, intercollicular_bound_B.csv, intercollicular_ccg_split_R.csv,
  intercollicular_latency_combos_*.csv (MATLAB tables exported by export_bilateral_tables.m)
No relay simulation is recomputed: the relay curve is a shift of the recorded X trace by S1's read-out lags.
Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python scripts/aim1_panels/build_B9.py
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2]
PEER = Path(r'D:/Google Drive/HermanLab/data/Newton_SC_2026/data')
SRC = ROOT / 'results' / 'aim1_panels' / '_bilateral_source'
OUT = ROOT / 'results' / 'aim1_panels' / 'B9_intercollicular_relay'
OUT.mkdir(parents=True, exist_ok=True)
shutil.copy(ROOT / 'scripts' / 'aim1_panels' / 'style.py', OUT / 'style.py')
stats = []


def stat(name, value, n, test, window, unit_class, session, ci_lo=np.nan, ci_hi=np.nan, p=np.nan, note=''):
    stats.append(dict(statistic=name, value=value, ci_lo=ci_lo, ci_hi=ci_hi, p=p, n=n, test=test, window=window,
                      unit_class=unit_class, session=session, note=note))


L = sio.loadmat(PEER / 'bilateral' / 'intercollicular_latency.mat', squeeze_me=True, struct_as_record=False)
S1 = pd.read_csv(SRC / 'onset_estimators_S1.csv')
relay = S1[(S1.sign == 1) & (S1.delay_s > 0)]
lag_lo, lag_hi = 1e3 * relay.mean_delta_s_1.min(), 1e3 * relay.mean_delta_s_1.max()
relay_out = relay.assign(true_delay_ms=1e3 * relay.delay_s, readout_delta_rise_ms=1e3 * relay.mean_delta_s_1,
                         readout_sd_rise_ms=1e3 * relay.sd_delta_s_1)[
    ['sign', 'true_delay_ms', 'transfer', 'readout_delta_rise_ms', 'readout_sd_rise_ms']]
relay_out.to_csv(OUT / 'B9_S1_relay_simulation_readout.csv', index=False)
stat('S1 planted relay (activation), rise-estimator read-out Delta range, ms', np.nan, len(relay), 'simulation '
     '(peer tests/test_onset_estimators.m; 100 sims per cell)', 'saccade-aligned', 'Y activation units', '08-19 + 08-20',
     lag_lo, lag_hi, note='true synaptic delays 2-10 ms + 5-ms kernel; linear/saturating/threshold transfer')

trace_rows, onset_rows = [], []
for sgn_name in ('act', 'supp'):
    tr = L['traces'].__dict__[sgn_name]
    combos = pd.read_csv(SRC / f'intercollicular_latency_combos_{sgn_name}.csv')
    for k, c in enumerate(tr):
        t_ms = np.round(1e3 * np.asarray(c.t), 1)
        x, y = np.asarray(c.x, float), np.asarray(c.y, float)
        ses = '08-' + str(c.label)[2:4]
        tgt = 'right' if 'target right' in c.label else 'left'
        cb = combos.iloc[k]
        lab = dict(session=ses, target_side=tgt, x_side=cb.xSide, y_side=cb.ySide, y_response=sgn_name,
                   n_trials=int(cb.nTrials), combo_label=str(c.label))
        # relay prediction: X trace shifted later by the S1 read-out lags (1-ms samples)
        def shift(v, ms):
            out = np.full_like(v, np.nan)
            s = int(round(ms))
            out[s:] = v[:len(v) - s]
            return out
        r_lo, r_hi = shift(x, lag_lo), shift(x, lag_hi)
        for i in range(len(t_ms)):
            trace_rows.append(dict(**lab, time_ms_from_saccade=t_ms[i], x_scaled=x[i], y_scaled=y[i],
                                   relay_pred_shift_lo_scaled=r_lo[i], relay_pred_shift_hi_scaled=r_hi[i]))
        onset_rows.append(dict(**lab, x_rise_onset_ms=1e3 * cb.x_rise_s, y_rise_onset_ms=1e3 * cb.y_rise_s,
                               delta_rise_ms=1e3 * (cb.y_rise_s - cb.x_rise_s), x_departure_onset_ms=1e3 * cb.x_dep_s,
                               y_departure_onset_ms=1e3 * cb.y_dep_s,
                               relay_pred_rise_onset_lo_ms=1e3 * cb.x_rise_s + lag_lo,
                               relay_pred_rise_onset_hi_ms=1e3 * cb.x_rise_s + lag_hi))
pd.DataFrame(trace_rows).to_csv(OUT / 'B9_abcd_traces.csv', index=False)
ON = pd.DataFrame(onset_rows)
ON.to_csv(OUT / 'B9_abcd_onsets.csv', index=False)
for sgn_name, word in (('act', 'activation'), ('supp', 'suppression')):
    R = L['res'].valid.__dict__[sgn_name]
    nY = int(ON[ON.y_response == sgn_name].combo_label.str.extract(r'\((\d+) units\)$')[0].astype(int).sum())
    stat(f'Delta rise onset Y - X ({word}), mean over 4 session x direction combinations, ms', 1e3 * R.delta[0], 4,
         'trial bootstrap (joint X, Y), 95% CI', 'saccade-aligned; baseline 200 ms before target', f'Y {word} units '
         f'({nY}) vs all X units', '08-19 + 08-20', 1e3 * R.ci[0, 0], 1e3 * R.ci[1, 0],
         note=f'Bonferroni (2 signs) upper bound {1e3 * R.upper_bonf[0]:+.1f} ms; pre-stated result "not resolved"')
    z = np.asarray(R.z_vs_relay)[0]
    stat(f'z of real Delta vs S1 relay cells ({word}, rise), range', np.nan, len(z), 'post hoc z = (Delta - mean_relay)'
         ' / sd_relay', 'saccade-aligned', f'Y {word}', '08-19 + 08-20', z.min(), z.max(),
         note='negative = earlier than simulated relay')
    for _, r in ON[ON.y_response == sgn_name].iterrows():
        stat(f'Delta rise onset ({word}), {r.session} target {r.target_side}', r.delta_rise_ms, r.n_trials,
             'rise estimator, point estimate', 'saccade-aligned', f'Y {word}', r.session)

# ---------------- e: pooled cross-SC CCG, 08-20 fixation ----------------
C = sio.loadmat(PEER / 'bilateral' / 'crossmidline_coupling.mat', squeeze_me=True, struct_as_record=False)['out']
F = C.Newton_08202026.fixation
lags = np.asarray(F.lags, int)
O, E = np.asarray(F.ccg, float), np.asarray(F.expected, float)
B = pd.read_csv(SRC / 'intercollicular_bound_B.csv')
bl = B[(B.window == '08-20 fixation (primary)') & (B.direction == 'short_lr')].iloc[0]
REC = (0.40, 0.55)                    # estimator recovery of a planted lag-specific deficit at f <= 0.2 (peer note C3)
RN = (5.0, 10.0)                      # r * N_X required to explain S = 0.63 with n = 0.10-0.20 spikes per 7-ms window
N_SPARSE = 50
ccg = pd.DataFrame(dict(lag_ms=lags, observed_count=O, expected_jitter_count=E, frac_modulation=(O - E) / E,
                        frac_modulation_se=np.sqrt(O) / E))
band = (lags >= 2) & (lags <= 8)
r_sparse = (RN[0] / N_SPARSE, RN[1] / N_SPARSE)
for nm, (r_a, r_b) in (('sparse_relay_N50', r_sparse), ('bound_relay', (bl.qbar_bound_lo, bl.qbar_bound_hi))):
    # measured-unit expectation: deficit r per X spike in lags 2-8 ms, recovered at 0.40-0.55
    lo = -max(REC) * max(r_a, r_b) if nm.startswith('sparse') else -REC[1] * bl.qbar_bound_lo
    hi = -min(REC) * min(r_a, r_b) if nm.startswith('sparse') else -REC[0] * bl.qbar_bound_hi
    ccg[f'{nm}_frac_mod_deep'] = np.where(band, min(lo, hi), 0.0)
    ccg[f'{nm}_frac_mod_shallow'] = np.where(band, max(lo, hi), 0.0)
ccg['session'] = '08-20'
ccg['window'] = 'fixation'
ccg['n_pairs'] = int(F.n_pairs)
ccg.to_csv(OUT / 'B9_e_ccg_0820_fixation.csv', index=False)
pd.DataFrame([dict(model='sparse relay', converging_X_neurons=N_SPARSE, r_times_N=f'{RN[0]:.0f}-{RN[1]:.0f}',
                   per_spike_reduction_r_lo=r_sparse[0], per_spike_reduction_r_hi=r_sparse[1],
                   recovery_lo=REC[0], recovery_hi=REC[1],
                   expected_measured_band_mod_lo=-REC[1] * r_sparse[1], expected_measured_band_mod_hi=-REC[0] * r_sparse[0]),
              dict(model='strongest relay not excluded (bound)', converging_X_neurons=np.nan,
                   converging_X_neurons_lo=RN[0] / bl.qbar_bound_hi, converging_X_neurons_hi=RN[1] / bl.qbar_bound_lo,
                   r_times_N=f'{RN[0]:.0f}-{RN[1]:.0f}', per_spike_reduction_r_lo=bl.qbar_bound_lo,
                   per_spike_reduction_r_hi=bl.qbar_bound_hi, recovery_lo=REC[0], recovery_hi=REC[1],
                   expected_measured_band_mod_lo=bl.modulation_lo95, expected_measured_band_mod_hi=bl.modulation_lo95)]
             ).to_csv(OUT / 'B9_e_relay_models.csv', index=False)
stat('08-20 fixation L->R band (2-8 ms) modulation (band fit)', bl.modulation, int(F.n_pairs), 'interval jitter + '
     'spill-corrected band fit; 95% CI lower bound in ci_lo', 'fixation', 'all cross-midline pairs', '08-20',
     bl.modulation_lo95, np.nan, note='08-20 fixation is the primary bound window (X nearly silent: not perisaccadic)')
stat('q-bar upper bound (L->R, 08-20 fixation)', np.nan, int(F.n_pairs), '-lower95 / recovery (0.40-0.55)', 'fixation',
     'all cross-midline pairs', '08-20', bl.qbar_bound_lo, bl.qbar_bound_hi)
stat('converging X neurons below which a relay is excluded', np.nan, int(F.n_pairs), 'N = (r N) / r_bound, r N = 5-10',
     'fixation bound', 'all pairs', '08-20', RN[0] / bl.qbar_bound_hi, RN[1] / bl.qbar_bound_lo,
     note='write-up "~170-500"; assumes every recorded X neuron projects (f = 1), while the 0.40-0.55 recovery was '
          'measured at f <= 0.2 (~0.15 at f = 1, which would loosen the bound ~3x)')
# all-group CCG split (both sessions, X->Y), for the "flat in every group" statement
G = pd.read_csv(SRC / 'intercollicular_ccg_split_R.csv')
G.to_csv(OUT / 'B9_e_ccg_by_Y_group.csv', index=False)
stat('max |z| over Y groups x windows (X->Y band and zero band)', float(np.nanmax(np.abs(G[['z_XtoY', 'z_zero']].values))),
     int(len(G)), 'band fit z', 'fixation, pre-saccade', '4 Y groups', '08-19 + 08-20 pooled',
     note='peer note: all |z| <= 1.8')
pd.DataFrame(stats).to_csv(OUT / 'stats.csv', index=False)
print(pd.DataFrame(stats)[['statistic', 'value', 'ci_lo', 'ci_hi', 'n']].to_string())
print(ON[['session', 'target_side', 'y_response', 'x_rise_onset_ms', 'y_rise_onset_ms', 'delta_rise_ms']])
