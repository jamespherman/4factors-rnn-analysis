#!/usr/bin/env python3
"""Regenerate P2 (and the alternative b' panel) from the CSVs in this folder. Edit freely to restyle."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
style.apply()
u = pd.read_csv(HERE / 'P2_units.csv')
st = pd.read_csv(HERE / 'stats.csv')
LAB = {'image': 'image effect (SD)', 'salience': 'salience effect (SD)', 'expected': 'location-prior effect (SD)',
       'reward': 'reward effect (SD)'}
TITLE = {'a': 'stimulus × stimulus', 'b': 'goal × stimulus', 'b_alt': 'goal × stimulus'}


def fmt_p(p):
    return f'p = {p:.0e}'.replace('e-0', 'e-') if p < 0.001 else f'p = {p:.2f}'


def scatter(ax, panel, letter, tpos=(0.03, 0.97, 'top')):
    s = st[(st.panel == panel) & (st.unit_class == 'all')].iloc[0]
    x, y = s.x_effect, s.y_effect
    ax.axhline(0, color=style.NULL, lw=0.7, zorder=0)
    ax.axvline(0, color=style.NULL, lw=0.7, zorder=0)
    for ty, kw in (('I', dict(facecolors='white', edgecolors=style.MUTED)),
                   ('E', dict(facecolors=style.DATA, edgecolors=style.DATA))):
        g = u[u.unit_type == ty][[f'{x}_sd', f'{y}_sd']].dropna()
        ax.scatter(g.iloc[:, 0], g.iloc[:, 1], s=7, lw=0.5, zorder=3 if ty == 'E' else 2,
                   label='classic (E)' if ty == 'E' else 'interneuron (I)', **kw)
    ax.text(tpos[0], tpos[1], f'r = {s.spearman_r:+.2f}\n{fmt_p(s.p)}\nn = {int(s.n)}', transform=ax.transAxes, ha='left',
            va=tpos[2], fontsize=6)
    ax.set_xlabel(LAB[x])
    ax.set_ylabel(LAB[y])
    ax.set_title(f'{letter}  {TITLE[panel]}', loc='left')


fig, axs = plt.subplots(1, 2, figsize=(2 * style.PANEL[0], style.PANEL[1]), gridspec_kw=dict(wspace=0.45))
scatter(axs[0], 'a', 'a')
scatter(axs[1], 'b', 'b', tpos=(0.03, 0.03, 'bottom'))
axs[0].legend(loc='lower right', fontsize=5.5, handletextpad=0.1, borderaxespad=0.1, markerscale=1.2)
style.save(fig, str(HERE / 'P2_factor_alignment_scatter'))

fig, ax = plt.subplots(figsize=style.PANEL)
scatter(ax, 'b_alt', "b'")
ax.legend(loc='lower right', fontsize=5.5, handletextpad=0.1, borderaxespad=0.1, markerscale=1.2)
style.save(fig, str(HERE / 'P2alt_reward_vs_salience'))
