#!/usr/bin/env python3
"""B7: no reward-difference coding and no competitor normalisation (Newton bilateral SC, srsClassic, 08-20).
Reads only the CSVs in this folder (+ style.py)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import style

HERE = Path(__file__).resolve().parent
style.apply()
a_ = np.asarray
T = pd.read_csv(HERE / 'B7_i_timecourse.csv')
NB = pd.read_csv(HERE / 'B7_i_null_bands.csv')
NET = pd.read_csv(HERE / 'B7_model_S1_S2_per_network.csv')
U = pd.read_csv(HERE / 'B7_ii_unit_coefficients.csv')
SM = pd.read_csv(HERE / 'B7_ii_population_mean_summary.csv')
SES, EPOCH, MODEL = '08-20', 'delay', 'a'

t = T[T.session == SES]
pre = t[t.aligned_to == 'target'].sort_values('t_center_ms')
sac = t[t.aligned_to == 'saccade'].iloc[0]
bon = NB[(NB.session == SES) & NB.level.str.startswith('Bonferroni')].iloc[0]
unc = NB[(NB.session == SES) & NB.level.str.startswith('uncorrected')].iloc[0]
res = NET[NET.role == 'resolver']

fig = plt.figure(figsize=(7.0, 2.6))
outer = fig.add_gridspec(1, 2, width_ratios=[5.0, 3.0], wspace=0.32, left=0.07, right=0.98, bottom=0.17, top=0.88)
gl = outer[0].subgridspec(1, 2, width_ratios=[4.2, 0.75], wspace=0.06)
ax = fig.add_subplot(gl[0])
axs = fig.add_subplot(gl[1], sharey=ax)
YL = (-3.6, 4.6)
for a in (ax, axs):
    a.axhspan(-bon.choice_z_crit, bon.choice_z_crit, color=style.NULL, lw=0, zorder=0)
    a.axhline(0, color=style.MUTED, lw=0.5)
ax.plot([-100, 0], [YL[0] + 0.15] * 2, color=style.MUTED, lw=0.8)
ax.text(-50, YL[0] + 0.25, 'placebo', ha='center', va='bottom', fontsize=5.5, color=style.MUTED)
# trained resolver S1 (reward-difference t), delay window 100-450 ms: range over networks + mean
ax.fill_between([100, 450], res.S1_delay_t_mean.min(), res.S1_delay_t_mean.max(), color=style.RESOLVER, alpha=0.15,
                lw=0, zorder=1)
ax.plot([100, 450], [res.S1_delay_t_mean.mean()] * 2, ls='--', color=style.RESOLVER, lw=1.2, zorder=2)
x = a_(pre.t_center_ms)
ax.plot(x, a_(pre.reward_t), 's-', color=style.MUTED, mfc='white', ms=3, lw=1, zorder=3, label='reward difference (t)')
ax.plot(x, a_(pre.choice_z_signed), 'o-', color=style.DATA, ms=3.2, lw=1.2, zorder=4, label='added choice (z)')
sig = pre[pre.choice_sig_bonferroni == True]  # noqa: E712
for _, r in sig.iterrows():
    ax.text(r.t_center_ms, r.choice_z_signed + 0.3, f'CP {r.added_CP:.2f}', ha='center', fontsize=5.5)
ax.set_xlim(-110, 460)
ax.set_ylim(*YL)
ax.set_xticks([-100, 0, 100, 200, 300, 400])
ax.set_xlabel('window centre (ms from target onset)')
ax.set_ylabel('coefficient (z or t) in L - R signal')
ax.set_title('i  L - R SC signal, 08-20 (n 227 choice trials)', loc='left')
ax.text(-105, bon.choice_z_crit - 0.1, 'Bonferroni null\n(11 windows)',
        ha='left', va='top', fontsize=5.5, color=style.MUTED)
handles = [Line2D([], [], marker='o', color=style.DATA, ms=3, lw=1.2, label='added choice (z)'),
           Line2D([], [], marker='s', color=style.MUTED, mfc='white', ms=3, lw=1, label='reward difference (t)'),
           Line2D([], [], ls='--', color=style.RESOLVER, lw=1.2,
                  label=f'trained resolvers, reward t (S1; {len(res)} nets)')]
ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.2, 0.06), fontsize=5.5, handlelength=1.8)
# pre-saccade reference
axs.plot([0], [sac.reward_t], 's', color=style.MUTED, mfc='white', ms=3)
axs.plot([0], [min(sac.choice_z_signed, YL[1] - 0.25)], marker='^', color=style.DATA, ms=4)
axs.text(0.08, YL[1] - 0.25, f'z {sac.choice_z_signed:.1f}\nCP {sac.added_CP:.2f}', fontsize=5.5, va='top')
jit = np.linspace(-0.15, 0.15, len(res))
axs.plot(jit, a_(res.S1_presac_t_mean), 'o', color=style.RESOLVER, mfc='none', ms=2.5, mew=0.7)
axs.set_xlim(-0.4, 0.6)
axs.set_xticks([0])
axs.set_xticklabels(['-100..0\nre sacc.'])
plt.setp(axs.get_yticklabels(), visible=False)
axs.spines['left'].set_visible(False)
axs.tick_params(axis='y', length=0)

# (ii) per-unit competitor-reward coefficients
gs2 = outer[1].subgridspec(2, 1, height_ratios=[1.5, 1], hspace=0.08)
ah = fig.add_subplot(gs2[0])
am = fig.add_subplot(gs2[1], sharex=ah)
u = U[(U.session == SES) & (U.epoch == EPOCH)][f'comp_coef_model_{MODEL}']
ah.hist(a_(u), bins=np.arange(-0.275, 0.276, 0.025), color=style.DATA, alpha=0.85, lw=0)
ah.axvline(0, color=style.MUTED, lw=0.5)
ah.set_ylabel('units')
ah.set_title(f'ii  competitor-reward coefficient, delay ({len(u)} units)', loc='left')
plt.setp(ah.get_xticklabels(), visible=False)
s = SM[(SM.session == SES) & (SM.epoch == EPOCH) & (SM.model == MODEL)].iloc[0]
PL = 'planted -10%/100 ms\n(power ≥ 0.98)'
rows = {'recorded mean': 4, 'null (choice only)': 3, PL: 2, 'trained resolvers': 1, 'trained hubs': 0}
am.plot([s.null_q025, s.null_q975], [rows['null (choice only)']] * 2, color=style.NULL, lw=5, solid_capstyle='butt')
am.plot([s.null_mean], [rows['null (choice only)']], '|', color=style.MUTED, ms=6)
am.plot([s['planted_g0.1_q025'], s['planted_g0.1_q975']], [rows[PL]] * 2, ls='--',
        color=style.RESOLVER, lw=1.2)
am.plot([s['planted_g0.1_mean']], [rows[PL]], '|', color=style.RESOLVER, ms=6)
am.plot([s.real_mean], [rows['recorded mean']], 'D', color=style.DATA, ms=3.5)
am.text(s.real_mean + 0.02, rows['recorded mean'], f'{s.real_mean:+.3f}', fontsize=5.5, va='center')
jy = np.linspace(-0.25, 0.25, len(res))
am.plot(a_(res.S2_mean_comp_coef_mean), rows['trained resolvers'] + jy, 'o', color=style.RESOLVER, ms=2.5)
hubs = NET[NET.role != 'resolver']
cols = [style.ARCH[a] for a in hubs.architecture]
am.scatter(a_(hubs.S2_mean_comp_coef_mean), rows['trained hubs'] + np.linspace(-0.25, 0.25, len(hubs)), c=cols, s=5,
           lw=0)
am.axvline(0, color=style.MUTED, lw=0.5)
am.set_yticks(list(rows.values()))
am.set_yticklabels(list(rows.keys()), fontsize=5.5)
am.set_ylim(-0.6, 4.6)
am.set_xlim(-0.28, 0.28)
am.set_xlabel('log rate change per 100 ms competitor reward')
am.tick_params(axis='y', length=0)
am.spines['left'].set_visible(False)
style.save(fig, str(HERE / 'B7_lr_signal_normalisation'))
