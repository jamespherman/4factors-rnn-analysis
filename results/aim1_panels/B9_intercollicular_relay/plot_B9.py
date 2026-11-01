#!/usr/bin/env python3
"""B9: ipsiversive activity vs a relay from the other colliculus (Newton bilateral SC, srsClassic).
Reads only the CSVs in this folder (+ style.py)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
style.apply()
a_ = np.asarray
TR = pd.read_csv(HERE / 'B9_abcd_traces.csv')
ON = pd.read_csv(HERE / 'B9_abcd_onsets.csv')
CC = pd.read_csv(HERE / 'B9_e_ccg_0820_fixation.csv')
RM = pd.read_csv(HERE / 'B9_e_relay_models.csv')
S = pd.read_csv(HERE / 'stats.csv')
RESP = 'act'
T0, T1 = -250, 80

fig = plt.figure(figsize=(7.0, 4.3))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.62, wspace=0.45, left=0.07, right=0.98, bottom=0.1,
                      top=0.92)
order = [('08-19', 'right'), ('08-19', 'left'), ('08-20', 'right'), ('08-20', 'left')]
for k, (ses, tgt) in enumerate(order):
    ax = fig.add_subplot(gs[0, k])
    d = TR[(TR.session == ses) & (TR.target_side == tgt) & (TR.y_response == RESP)]
    d = d[(d.time_ms_from_saccade >= T0) & (d.time_ms_from_saccade <= T1)]
    o = ON[(ON.session == ses) & (ON.target_side == tgt) & (ON.y_response == RESP)].iloc[0]
    t = a_(d.time_ms_from_saccade)
    ax.fill_between(t, a_(d.relay_pred_shift_lo_scaled), a_(d.relay_pred_shift_hi_scaled), color=style.RESOLVER,
                    alpha=0.15, lw=0)
    ax.plot(t, a_(d.relay_pred_shift_lo_scaled), ls='--', color=style.RESOLVER, lw=0.9)
    ax.plot(t, a_(d.relay_pred_shift_hi_scaled), ls='--', color=style.RESOLVER, lw=0.9)
    ax.plot(t, a_(d.x_scaled), color=style.MUTED, lw=1.3)
    ax.plot(t, a_(d.y_scaled), color=style.DATA, lw=1.3)
    for v, c in ((o.x_rise_onset_ms, style.MUTED), (o.y_rise_onset_ms, style.DATA)):
        ax.plot([v, v], [1.18, 1.36], color=c, lw=1.6)
    ax.axvline(0, color=style.MUTED, lw=0.5, ls=':')
    ax.axhline(0, color=style.MUTED, lw=0.4)
    ax.set_xlim(T0, T1)
    ax.set_ylim(-0.5, 1.4)
    ax.set_yticks([0, 1])
    nX = o.combo_label.split('(')[1].split(' ')[0]
    nY = o.combo_label.split('(')[2].split(' ')[0]
    ax.set_title(f'{"abcd"[k]}  {ses}, target {tgt}\nX {o.x_side} SC ({nX}), Y {o.y_side} ({nY})', loc='left',
                 fontsize=6.5)
    ax.text(0.03, 0.03, f'Δ {o.delta_rise_ms:+.0f} ms', transform=ax.transAxes, fontsize=6)
    ax.set_xlabel('ms from saccade onset')
    if k == 0:
        ax.set_ylabel('scaled rate')
        ax.text(-240, 0.88, 'X burst', color=style.MUTED, fontsize=5.5)
        ax.text(-240, 0.70, 'Y ipsiversive', color=style.DATA, fontsize=5.5)
        ax.text(-240, 0.52, 'relay of X\n(+10-27 ms)', color=style.RESOLVER, fontsize=5.5, va='top')

# e: onset differences per combination vs relay prediction
ax = fig.add_subplot(gs[1, 0:2])
lo = S[S.statistic.str.startswith('S1 planted relay')].iloc[0]
m = S[S.statistic.str.startswith('Delta rise onset Y - X (activation)')].iloc[0]
ax.axvspan(lo.ci_lo, lo.ci_hi, color=style.RESOLVER, alpha=0.18, lw=0)
ax.text((lo.ci_lo + lo.ci_hi) / 2, 5.25, 'relay of recorded X\n(S1 read-out)', color=style.RESOLVER, fontsize=5.5,
        ha='center', va='bottom')
for i, (ses, tgt) in enumerate(order):
    o = ON[(ON.session == ses) & (ON.target_side == tgt) & (ON.y_response == RESP)].iloc[0]
    ax.plot(o.delta_rise_ms, 4 - i, 'o', color=style.DATA, ms=3.5)
ax.errorbar([m.value], [0], xerr=[[m.value - m.ci_lo], [m.ci_hi - m.value]], fmt='D', color=style.DATA, ms=3.5, lw=1,
            capsize=0)
ax.axvline(0, color=style.MUTED, lw=0.5, ls=':')
ax.set_yticks([4, 3, 2, 1, 0])
ax.set_yticklabels([f'{s}, target {t}' for s, t in order] + ['mean (95% CI)'], fontsize=6)
ax.set_ylim(-0.7, 5.2)
ax.set_xlim(-80, 45)
ax.set_xlabel('rise onset Y - X (ms)')
ax.set_title('e  ipsiversive activation onset vs relay', loc='left')
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)

# f: CCG
ax = fig.add_subplot(gs[1, 2:4])
c = CC[(CC.lag_ms >= -25) & (CC.lag_ms <= 25)]
lag = a_(c.lag_ms)
ax.fill_between(lag, a_(c.frac_modulation - c.frac_modulation_se), a_(c.frac_modulation + c.frac_modulation_se),
                color=style.NULL, lw=0, step='mid')
ax.step(lag, a_(c.frac_modulation), where='mid', color=style.DATA, lw=1)
sp = RM[RM.model == 'sparse relay'].iloc[0]
bd = RM[RM.model.str.startswith('strongest')].iloc[0]
ax.fill_between(lag, a_(c.sparse_relay_N50_frac_mod_deep), a_(c.sparse_relay_N50_frac_mod_shallow), step='mid',
                color=style.RESOLVER, alpha=0.18, lw=0)
ax.step(lag, a_((c.sparse_relay_N50_frac_mod_deep + c.sparse_relay_N50_frac_mod_shallow) / 2), where='mid',
        ls='--', color=style.RESOLVER, lw=1.1)
ax.step(lag, a_(c.bound_relay_frac_mod_deep), where='mid', ls=':', color=style.RESOLVER, lw=1.1)
ax.text(9, -0.075, f'sparse relay\n({int(sp.converging_X_neurons)} X neurons)', color=style.RESOLVER, fontsize=5.5,
        va='center')
ax.text(-24, -0.035, f'strongest relay not excluded\n({bd.converging_X_neurons_lo:.0f}-'
        f'{bd.converging_X_neurons_hi:.0f} X neurons, dotted)', color=style.RESOLVER, fontsize=5.5, va='center')
ax.axhline(0, color=style.MUTED, lw=0.5)
ax.axvline(0, color=style.MUTED, lw=0.5, ls=':')
ax.set_xlim(-25, 25)
ax.set_ylim(-0.11, 0.04)
ax.set_xlabel('lag (ms; + = right SC after left SC)')
ax.set_ylabel('CCG modulation (fraction)')
ax.set_title(f'f  cross-SC CCG, 08-20 fixation ({int(c.n_pairs.iloc[0])} pairs)', loc='left')
style.save(fig, str(HERE / 'B9_intercollicular_relay'))
