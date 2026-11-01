#!/usr/bin/env python3
"""Plot R01 core panel P4 (bilateral SC state, Newton 08-20 and 08-19) from the CSVs in this folder only.
a  P(chose right) by pre-target L-R SC bias tercile (mean +/- binomial SE), z vs circular-shift null annotated.
b  Fixation cross-midline CCG, (obs - jitter)/jitter, with the 2-8 ms commissural bands shaded.
Run from anywhere: python plot_P4.py  (needs numpy, pandas, matplotlib; style.py alongside)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
style.apply()
A = pd.read_csv(HERE / 'P4_a_choice_by_bias_tercile.csv')
B = pd.read_csv(HERE / 'P4_b_ccg_fixation.csv')
BANDS = pd.read_csv(HERE / 'P4_b_bands_shaded.csv')
S = pd.read_csv(HERE / 'stats.csv')
COL = {'08-20': style.DATA, '08-19': style.MUTED}
W, H = 2.4, 2.0

fig, (axa, axb) = plt.subplots(1, 2, figsize=(2 * W + 0.5, H), gridspec_kw=dict(wspace=0.55))

# ---- a
for k, (ses, d) in enumerate(A.groupby('session', sort=False)):
    x = d.tercile + 0.06 * (k - 0.5)
    z = S.query('session == @ses and statistic == "pretarget_bias_z"').iloc[0]
    axa.errorbar(x.to_numpy(), d.p_chose_right.to_numpy(), yerr=d.se_binomial.to_numpy(), color=COL[ses], marker='o', ms=3.5, lw=1.4,
                 elinewidth=0.9, capsize=0, label=f'{ses}  n = {int(z.n)}  z = {z.z:.2f}')
axa.axhline(0.5, color=style.NULL, lw=0.6, zorder=0)
axa.set(xlim=(0.6, 3.4), ylim=(0, 1), xticks=[1, 2, 3], yticks=[0, 0.5, 1])
axa.set_xticklabels(['R > L', 'middle', 'L > R'])
axa.set_xlabel('pre-target SC bias tercile\n(300 ms before targets)')
axa.set_ylabel('P(chose right)')
axa.legend(loc='upper left', handlelength=1.2, borderaxespad=0.2)
axa.text(0.98, 0.03, 'z vs circular-shift null\n(beyond history, reward, gaze)', transform=axa.transAxes,
         ha='right', va='bottom', fontsize=5.5, color=style.MUTED)
axa.set_title('Pre-target collicular state predicts choice', loc='left')

# ---- b
for _, b in BANDS.iterrows():
    axb.axvspan(b.lag_lo_ms - 0.5, b.lag_hi_ms + 0.5, color=style.NULL, alpha=0.7, lw=0, zorder=0)
axb.axhline(0, color=style.MUTED, lw=0.5, zorder=1)
for ses, d in B.groupby('session', sort=False):
    axb.step(d.lag_ms.to_numpy(), 100 * d.modulation.to_numpy(), where='mid', color=COL[ses], lw=0.9, label=ses, zorder=2)
axb.set(xlim=(-40, 40), ylim=(-6, 6), xticks=[-40, -20, 0, 20, 40], yticks=[-5, 0, 5])
axb.set_xlabel('lag (ms), fixation\n(+ = right SC after left)')
axb.set_ylabel('CCG excess over jitter (%)')
axb.text(5, 5.3, 'L→R', ha='center', fontsize=5.5, color=style.MUTED)
axb.text(-5, 5.3, 'R→L', ha='center', fontsize=5.5, color=style.MUTED)
axb.legend(loc='upper left', ncol=1, handlelength=1.2, borderaxespad=0.2, columnspacing=0.8)
sb = S[S.statistic.isin(['ccg_fixation_band_short_lr', 'ccg_fixation_band_short_rl'])]
axb.text(0.5, 0.03, f'2–8 ms bands: |excess| ≤ {sb.value.abs().max():.1f}%, |z| ≤ {sb.z.abs().max():.1f} (n.s.)',
         transform=axb.transAxes, ha='center', va='bottom', fontsize=5.5, color=style.DATA,
         bbox=dict(fc='white', ec='none', pad=0.5))
axb.set_title('No fast coupling across the midline', loc='left')

for ax, lab in ((axa, 'a'), (axb, 'b')):
    ax.text(-0.3, 1.1, lab, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')
style.save(fig, str(HERE / 'P4_bilateral_state'))
