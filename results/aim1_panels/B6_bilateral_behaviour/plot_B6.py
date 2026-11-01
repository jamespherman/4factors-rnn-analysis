#!/usr/bin/env python3
"""B6: bilateral conflict task behaviour (Newton, srsClassic). Reads only the CSVs in this folder (+ style.py)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
style.apply()
np_ = np.asarray
P = pd.read_csv(HERE / 'B6_ab_psychometric_reward_points.csv')
F = pd.read_csv(HERE / 'B6_ab_psychometric_reward_fit.csv')
PB = pd.read_csv(HERE / 'B6_c_psychometric_belief_points.csv')
FB = pd.read_csv(HERE / 'B6_c_psychometric_belief_fit.csv')
DB = pd.read_csv(HERE / 'B6_d_rt_vs_belief_bins.csv')
DM = pd.read_csv(HERE / 'B6_d_rt_vs_blockmean_points.csv')
E = pd.read_csv(HERE / 'B6_e_within_block_rich_minus_lean.csv')
S = pd.read_csv(HERE / 'stats.csv')


def sv(name, ses='08-20'):
    return S[(S.statistic == name) & (S.session == ses)].iloc[0]


COND = {'congruent': dict(color=style.DATA, mfc=style.DATA, label='congruent'),
        'conflict': dict(color=style.MUTED, mfc='white', label='conflict')}


def psy(ax, pts, fit, x, xlab, title):
    for cond, kw in COND.items():
        f = fit[fit.condition == cond]
        ax.fill_between(np_(f[x]), np_(f.band95_lo), np_(f.band95_hi), color=kw['color'], alpha=0.12, lw=0)
        ax.plot(np_(f[x]), np_(f.p_fit), color=kw['color'], lw=1.2)
        p = pts[pts.condition == cond]
        ax.errorbar(np_(p[f'{x}_bin_mean']), np_(p.p_chose_right), yerr=np_([p.p_chose_right - p.ci95_lo, p.ci95_hi - p.p_chose_right]),
                    fmt='o', color=kw['color'], mfc=kw['mfc'], mec=kw['color'], ms=3.5, lw=0.8, capsize=0,
                    label=kw['label'])
    ax.axhline(0.5, color=style.MUTED, lw=0.5, ls=':')
    ax.axvline(0, color=style.MUTED, lw=0.5, ls=':')
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(xlab)
    ax.set_ylabel('P(choose right)')
    ax.set_title(title, loc='left')


fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.6))
ax = axs[0, 0]
psy(ax, P[P.session == '08-20'], F[F.session == '08-20'], 'reward_diff_ms', 'reward right - left (ms)',
    'a  08-20: actual reward')
b = sv('choice beta: reward difference (per 100 ms)')
bs = sv('choice beta: salience difference (per +-1 level)')
ax.text(0.03, 0.97, f'reward β {b.value:+.2f}/100 ms\n[{b.ci_lo:+.2f}, {b.ci_hi:+.2f}], n {int(b.n)}\n'
        f'salience β {bs.value:+.2f}, p {bs.p:.3f}', transform=ax.transAxes, va='top', fontsize=6)
ax.legend(loc='lower right', handletextpad=0.2, borderaxespad=0.1)

ax = axs[0, 1]
psy(ax, P[P.session == '08-19'], F[F.session == '08-19'], 'reward_diff_ms', 'reward right - left (ms)',
    'b  08-19: actual reward')
b = sv('choice beta: reward difference (per 100 ms)', '08-19')
ax.text(0.03, 0.97, f'reward β {b.value:+.2f}/100 ms\n[{b.ci_lo:+.2f}, {b.ci_hi:+.2f}], n {int(b.n)}',
        transform=ax.transAxes, va='top', fontsize=6)

ax = axs[0, 2]
psy(ax, PB, FB, 'belief_diff_ms', 'belief value right - left (ms)', 'c  08-20: fitted belief (circular)')
ax.text(0.03, 0.97, 'belief fitted to these choices;\nα at its 0.01 bound', transform=ax.transAxes, va='top',
        fontsize=6, color=style.MUTED)

ax = axs[1, 0]
d = DB[DB.session == '08-20']
m = DM[DM.session == '08-20']
ax.errorbar(np_(m.value_ms), np_(m.rt_ms_mean), yerr=np_(m.rt_ms_sem), fmt='s', color=style.MUTED, mfc='white', ms=3, lw=0.6,
            capsize=0, label='true block-mean value\n(block x identity)')
ax.errorbar(np_(d.value_ms_mean), np_(d.rt_ms_mean), yerr=np_(d.rt_ms_sem), fmt='o-', color=style.DATA, ms=3.5, lw=1.2, capsize=0,
            label='belief value (quintiles)')
r1 = sv('Spearman rho, single-target RT vs belief value of shown identity')
r2 = sv('Spearman rho, single-target RT vs TRUE block-mean value of shown identity (drift control)')
ax.text(0.03, 0.03, f'belief: ρ {r1.value:+.2f} (p {r1.p:.0e}), n {int(r1.n)}\n'
        f'block mean: ρ {r2.value:+.2f} (p {r2.p:.2f})', transform=ax.transAxes, va='bottom', fontsize=6)
ax.set_xlabel('value of shown identity (ms reward)')
ax.set_ylabel('single-target RT (ms)')
ax.set_title('d  08-20: RT vs value', loc='left')
ax.legend(loc='upper right', fontsize=5.5, handletextpad=0.2, borderaxespad=0.1)
ylo, yhi = ax.get_ylim()
ax.set_ylim(ylo - 0.25 * (yhi - ylo), yhi + 0.25 * (yhi - ylo))

ax = axs[1, 1]
for i, ses in enumerate(['08-20', '08-19']):
    e = E[E.session == ses]
    x = i + np.linspace(-0.12, 0.12, len(e))
    ax.plot(x, np_(e.rt_diff_rich_minus_lean_ms), 'o', color=style.DATA if ses == '08-20' else style.MUTED,
            mfc='white' if ses == '08-19' else style.DATA, ms=3.2)
    w = sv('within-block RT difference, shown rich - shown lean identity (mean over blocks), ms', ses)
    ax.plot([i - 0.25, i + 0.25], [w.value] * 2, color=style.DATA, lw=1.5)
    ax.plot([i, i], [w.ci_lo, w.ci_hi], color=style.DATA, lw=0.8)
    ax.text(i, 54, f'{w.value:+.1f} ms\np {w.p:.2f}', ha='center', va='top', fontsize=6)
ax.axhline(0, color=style.MUTED, lw=0.5, ls=':')
ax.set_xticks([0, 1])
ax.set_xticklabels([f'08-20\n({len(E[E.session == "08-20"])} blocks)', f'08-19\n({len(E[E.session == "08-19"])} blocks)'])
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(-35, 56)
ax.set_ylabel('RT rich - lean identity (ms)')
ax.set_title('e  within-block value effect', loc='left')
axs[1, 2].axis('off')
axs[1, 2].text(0, 1, 'Newton bilateral SC, srsClassic task.\n'
               'a-c: choice trials, one target per\nhemifield; points = quintile bins\n(Wilson 95% CI); lines = '
               'logistic fit\nper condition (bootstrap 95% band).\n'
               'Conflict: salience and reward favour\nopposite targets.\n'
               'd-e: single-target trials. Median RT\nper block x identity in e (drift cancels).',
               va='top', fontsize=6, color=style.MUTED, transform=axs[1, 2].transAxes)
fig.tight_layout(w_pad=1.2, h_pad=1.4)
style.save(fig, str(HERE / 'B6_bilateral_behaviour'))
