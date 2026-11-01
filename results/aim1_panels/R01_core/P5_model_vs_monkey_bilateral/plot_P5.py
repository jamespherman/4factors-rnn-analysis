#!/usr/bin/env python3
"""Plot R01 core panel P5 (pre-target bilateral SC state: monkey vs trained conflict-task E-I RNNs) from the CSVs in
this folder only. Run from anywhere: python plot_P5.py  (needs numpy, pandas, matplotlib; style.py alongside).
i    P(chose right) by tercile of the pre-target left-minus-right population bias (E units in the model).
ii   Partial correlation across trials of left and right pre-target population activity; one dot per session/network.
     Grey open dots: same networks with the input noise on the fixation channel (the only input shared by both
     hemispheres) removed.
iii  Bias coefficient in the choice logistic regression (log-odds per SD of bias). Monkey: beyond history, reward
     side and gaze (P4). Model: within big-side x salient-side x dT cells (cell fixed effects; cells whose choice
     never varies dropped)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
style.apply()
A = pd.read_csv(HERE / 'P5_i_choice_by_bias_tercile.csv')
B = pd.read_csv(HERE / 'P5_ii_LR_partial_correlation.csv')
C = pd.read_csv(HERE / 'P5_iii_bias_choice_coefficient.csv')
W, H = style.PANEL
H = 2.1
MON = {'08-20': style.DATA, '08-19': style.MUTED}
GRP = {'bilateral': ('Bilateral RNN', 'o', style.RESOLVER, '-'),
       'no_commissure': ('No-commissure RNN', 'o', 'white', '--')}
rng = np.random.default_rng(0)

fig, axs = plt.subplots(1, 3, figsize=(3 * W + 0.8, H), gridspec_kw=dict(wspace=0.6))

# ---- i
ax = axs[0]
for k, (ses, d) in enumerate(A[A.source == 'monkey'].groupby('group', sort=False)):
    ax.errorbar((d.tercile + 0.05 * (k - 0.5)).to_numpy(), d.p_chose_right.to_numpy(), yerr=d.se.to_numpy(), color=MON[ses], marker='o', ms=3, lw=1.1,
                elinewidth=0.8, capsize=0, label=f'monkey {ses}')
for k, (grp, (lab, mk, fc, ls)) in enumerate(GRP.items()):
    d = A[(A.source == 'model') & (A.group == grp)]
    ax.errorbar((d.tercile + 0.05 * (k + 1.5)).to_numpy(), d.p_chose_right.to_numpy(), yerr=d.se.to_numpy(), color=style.RESOLVER, mfc=fc, marker=mk,
                ms=3, lw=1.1, ls=ls, elinewidth=0.8, capsize=0, label=lab + f' (n = {int(d.n_networks.iloc[0])})')
ax.axhline(0.5, color=style.NULL, lw=0.6, zorder=0)
ax.set(xlim=(0.6, 3.6), ylim=(0, 1.22), xticks=[1, 2, 3], yticks=[0, 0.5, 1])
ax.set_xticklabels(['R > L', 'middle', 'L > R'])
ax.set_xlabel('pre-target SC bias tercile')
ax.set_ylabel('P(chose right)')
ax.legend(loc='upper left', handlelength=1.6, borderaxespad=0.0, fontsize=5.2)
ax.set_title('Pre-target bias and choice', loc='left')


def strip(ax, df, val, ylab, title, ref0=True, ctrl=None, model_val=None):
    mv = model_val or val
    xt, xl = [], []
    m = df[df.source == 'monkey']
    for k, r in enumerate(m.itertuples()):
        ax.plot(0 + 0.12 * (k - 0.5), getattr(r, val), 'o', color=MON[r.group], ms=4, zorder=3)
    xt.append(0); xl.append('monkey')
    for j, (grp, (lab, mk, fc, ls)) in enumerate(GRP.items()):
        x0 = j + 1
        d = df[(df.source == 'model') & (df.group == grp) & (df.condition == 'intact') & (df.unit_class == 'E')]
        jit = rng.uniform(-0.1, 0.1, len(d))
        ax.plot(x0 - 0.12 + jit, d[mv].to_numpy(), mk, color=style.RESOLVER, mfc=fc, ms=3, mew=0.8, zorder=3)
        ax.plot([x0 - 0.3, x0 + 0.06], [d[mv].mean()] * 2, color=style.RESOLVER, lw=1.2, zorder=4)
        if ctrl is not None:
            dc = df[(df.source == 'model') & (df.group == grp) & (df.condition == ctrl) & (df.unit_class == 'E')]
            ax.plot(x0 + 0.2 + rng.uniform(-0.06, 0.06, len(dc)), dc[val].to_numpy(), 'o', color=style.MUTED, mfc='white',
                    ms=2.5, mew=0.6, zorder=2)
        xt.append(x0); xl.append({'bilateral': 'bilateral', 'no_commissure': 'no comm.'}[grp])
    if ref0:
        ax.axhline(0, color=style.NULL, lw=0.6, zorder=0)
    ax.set(xticks=xt, xlim=(-0.5, 2.5))
    ax.set_xticklabels(xl)
    ax.set_ylabel(ylab)
    ax.set_title(title, loc='left')


strip(axs[1], B, 'partial_r', 'partial r (left SC, right SC)\npre-target activity', 'Shared L–R fluctuation',
      ctrl='no_shared_input_noise')
axs[1].set_ylim(-0.05, 0.62)
axs[1].text(0.98, 0.99, 'grey: shared (fixation-input)\nnoise removed', transform=axs[1].transAxes, ha='right',
            va='top', fontsize=5.2, color=style.MUTED)
strip(axs[2], C, 'bias_coef', 'bias coefficient\n(log-odds per SD)', 'Bias predicts choice?', model_val='bias_coef_celladj')
axs[2].text(0.98, 0.98, 'model: E units, within\nreward x salience x dT cells', transform=axs[2].transAxes, ha='right',
            va='top', fontsize=5.2, color=style.MUTED)

for ax, lab in zip(axs, 'abc'):
    ax.text(-0.32, 1.1, lab, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')
style.save(fig, str(HERE / 'P5_model_vs_monkey_bilateral'))
