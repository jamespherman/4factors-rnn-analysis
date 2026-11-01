#!/usr/bin/env python3
"""Build the R01 core panel P4 data folder: bilateral SC state (Newton, 08-20 and 08-19).

Panels
  a  P(chose right) by tercile of the pre-target left-minus-right SC bias (300 ms before targets), binomial SE,
     per session; regression z of the bias vs its circular-shift null annotated.
  b  Pooled cross-midline CCG during fixation, (obs - jitter)/jitter, 1-ms bins, -40..40 ms, both sessions; the 2-8 ms
     commissural bands shaded; band-modulation result annotated.

Sources (peer repo, read-only; nothing is recomputed):
  Newton_SC_2026/data/bilateral/pretarget_bias.mat         (code/run_pretarget_bias.m, saved 2026-10-26)
  Newton_SC_2026/data/bilateral/crossmidline_coupling.mat  (code/run_crossmidline_coupling.m, saved 2026-10-26)
The tercile means/SE and the CCG modulation are tabulated exactly as code/plot_bilateral_bias.m and
code/plot_bilateral_coupling.m do (saved bias + choices; saved ccg and jitter expectation). MATLAB prctile
= numpy percentile method 'hazen'.
Run: OMP_NUM_THREADS=1 python scripts/aim1_panels/build_R01_P4.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
PEER = Path(r'D:/Google Drive/HermanLab/data/Newton_SC_2026/data/bilateral')
OUT = ROOT / 'results' / 'aim1_panels' / 'R01_core' / 'P4_bilateral_state'
OUT.mkdir(parents=True, exist_ok=True)
shutil.copy(ROOT / 'scripts' / 'aim1_panels' / 'style.py', OUT / 'style.py')
SES = {'Newton_08202026': '08-20', 'Newton_08192026': '08-19'}
stats = []


def stat(session, statistic, value, test, n, z=np.nan, p=np.nan, p_text='', ci_lo=np.nan, ci_hi=np.nan, note=''):
    stats.append(dict(session=session, statistic=statistic, value=value, z=z, p=p, p_text=p_text, ci_lo=ci_lo,
                      ci_hi=ci_hi, n=n, test=test, note=note))


load = lambda f, v: sio.loadmat(PEER / f, squeeze_me=True, struct_as_record=False, variable_names=[v])[v]
R = load('pretarget_bias.mat', 'results')
C = load('crossmidline_coupling.mat', 'out')

# ---- a: choice by bias tercile (as plot_bilateral_bias.m panel A)
rows = []
for key, lab in SES.items():
    r = getattr(R, key)
    b = np.asarray(r.bias, float)[np.asarray(r.idx2a, int) - 1]
    y = np.asarray(r.y2a, float)
    assert b.size == y.size == int(r.n2a)
    e = np.percentile(b, [100 / 3, 200 / 3], method='hazen')
    g = np.searchsorted(e, b, side='right')
    for t, name in enumerate(['R > L', 'middle', 'L > R']):
        yy = y[g == t]
        p = yy.mean()
        rows.append(dict(session=lab, tercile=t + 1, tercile_label=name, n_trials=yy.size, p_chose_right=p,
                         se_binomial=np.sqrt(p * (1 - p) / yy.size)))
    nnull = np.size(r.tNull)
    stat(lab, 'pretarget_bias_z', float(r.coef), 'logistic regression z of L-R bias (beyond prev side, running side '
         'mean, reward side, gaze x, eye movements) vs circular-shift null', int(r.n2a), z=float(r.z),
         p=max(float(r.p_shift), 1 / nnull), p_text=f'{round(float(r.p_shift) * nnull)}/{nnull} shifts as large (p < 1/{nnull}); p column = upper bound 1/{nnull}',
         note='value = bias coefficient, log-odds per SD; window 300 ms before targets')
    stat(lab, 'pretarget_bias_z_no_covariates', np.nan, 'logistic regression z, no covariates', int(r.n2a),
         z=float(r.z_no_cov))
    rn = np.asarray(r.rNull, float)
    stat(lab, 'pretarget_LR_shared_fluctuation_partial_r', float(r.r_partial),
         'partial correlation of left and right SC pre-target population activity vs null', int(r.n2c),
         p=float(r.p2c), ci_lo=np.percentile(rn, 2.5), ci_hi=np.percentile(rn, 97.5),
         note=f'positive = shared (not push-pull); raw r = {float(r.r_raw):.3f}; ci = 2.5/97.5 pct of saved null '
              f'(n = {rn.size})')
pd.DataFrame(rows).to_csv(OUT / 'P4_a_choice_by_bias_tercile.csv', index=False)

# ---- b: fixation CCGs (as plot_bilateral_coupling.m panel A)
ccg = []
bands = dict(zero='-1..1 ms', short_lr='+2..+8 ms (L->R)', short_rl='-8..-2 ms (R->L)',
             flank_lr='+9..+19 ms', flank_rl='-19..-9 ms')
for key, lab in SES.items():
    f = getattr(C, key).fixation
    m = (np.asarray(f.ccg, float) - np.asarray(f.expected, float)) / np.asarray(f.expected, float)
    for lag, obs, ex, mm in zip(np.asarray(f.lags), np.asarray(f.ccg), np.asarray(f.expected), m):
        ccg.append(dict(session=lab, lag_ms=int(lag), observed=obs, jitter_expected=ex, modulation=mm))
    for bk, bdesc in bands.items():
        mod, se, z = getattr(f.mod, bk), getattr(f.mod_se, bk), getattr(f.z, bk)
        stat(lab, f'ccg_fixation_band_{bk}', 100 * mod, 'band_fit modulation (% of expected coincidences) vs jitter; '
             'z from band_fit; p two-sided normal from z', int(f.n_pairs), z=float(z), p=2 * norm.sf(abs(z)),
             ci_lo=100 * (mod - 1.96 * se), ci_hi=100 * (mod + 1.96 * se),
             note=f'band {bdesc}; n = pairs; {int(f.n_segments)} fixation segments, {float(f.seconds):.0f} s')
pd.DataFrame(ccg).to_csv(OUT / 'P4_b_ccg_fixation.csv', index=False)
pd.DataFrame([dict(band='short_lr', lag_lo_ms=2, lag_hi_ms=8), dict(band='short_rl', lag_lo_ms=-8, lag_hi_ms=-2)]
             ).to_csv(OUT / 'P4_b_bands_shaded.csv', index=False)
pd.DataFrame(stats).to_csv(OUT / 'stats.csv', index=False)

if (OUT / 'plot_P4.py').exists():  # plot script lives in the output folder (standalone, CSV-only)
    subprocess.run([sys.executable, str(OUT / 'plot_P4.py')], check=True, cwd=OUT)
print('wrote', OUT)
