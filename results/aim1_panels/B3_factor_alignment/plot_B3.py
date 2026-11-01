#!/usr/bin/env python3
"""Regenerate B3 from the CSVs in this folder (numpy/pandas/matplotlib only). Edit freely to restyle."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
style.apply()
units = pd.read_csv(HERE / 'B3_units.csv')
pairs = pd.read_csv(HERE / 'B3_pairs.csv')
rel = pd.read_csv(HERE / 'B3_reliability.csv')
FACT = ['reward', 'expected', 'image', 'salience', 'face']
FLAB = {'reward': 'reward', 'expected': 'prior', 'image': 'image', 'salience': 'salience', 'face': 'face'}
CLS = {'E': dict(color=style.DATA, mfc=style.DATA, label='classic (E)'),
       'I': dict(color=style.MUTED, mfc='none', label='interneuron (I)')}


def pair(cls, a, b):
    q = pairs[(pairs.unit_class == cls) & (((pairs.factor_a == a) & (pairs.factor_b == b)) |
                                           ((pairs.factor_a == b) & (pairs.factor_b == a)))]
    return q.iloc[0]


def pfmt(p):
    return 'p<0.001' if p < 0.001 else f'p={p:.2g}'


fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2], wspace=0.75, hspace=0.6)

# ---- (a), (b) scatters ------------------------------------------------------------------------------------------
for k, (xa, yb, tag) in enumerate([('salience', 'reward', 'a'), ('salience', 'image', 'b')]):
    ax = fig.add_subplot(gs[0, k])
    for cls in ('I', 'E'):
        g = units[units.unit_type == cls]
        ax.scatter(g[f'{xa}_sd'], g[f'{yb}_sd'], s=7, lw=0.6, edgecolors=CLS[cls]['color'],
                   facecolors=CLS[cls]['mfc'], label=CLS[cls]['label'], zorder=3 if cls == 'E' else 2)
    ax.axhline(0, color=style.NULL, lw=0.6, zorder=0)
    ax.axvline(0, color=style.NULL, lw=0.6, zorder=0)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(*((-1.2, 1.6) if yb == 'reward' else (-0.85, 2.0)))
    txt = []
    for cls in ('E', 'I'):
        q = pair(cls, xa, yb)
        txt.append(f'{cls}: r = {q.spearman_r_full:+.2f} ({pfmt(q.p_trial_shuffle_2sided)}, n={q.n_units})')
    ax.text(0.02, 0.99, '\n'.join(txt), transform=ax.transAxes, va='top', ha='left', fontsize=5.5)
    ax.set_xlabel('salience effect (SD)')
    ax.set_ylabel(f'{FLAB[yb]} effect (SD)')
    ax.set_title(f'{tag}  {FLAB[yb]} vs salience', loc='left')
    if k == 0:
        ax.legend(loc='lower right', fontsize=5.5, handletextpad=0.1, borderaxespad=0.1, markerscale=1.3)

# ---- (c) matrices -----------------------------------------------------------------------------------------------
for k, cls in enumerate(('E', 'I')):
    ax = fig.add_subplot(gs[1, k])
    M = np.full((5, 5), np.nan)
    for i, a in enumerate(FACT):
        for j, b in enumerate(FACT):
            if i > j:
                M[i, j] = pair(cls, a, b).spearman_r_full
    im = ax.imshow(M, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    for i, a in enumerate(FACT):
        for j, b in enumerate(FACT):
            if i > j:
                q = pair(cls, a, b)
                star = '*' if q.p_trial_shuffle_2sided < 0.05 else ''
                ax.text(j, i, f'{q.spearman_r_full:+.2f}{star}'.replace('+0.', '.').replace('-0.', '-.'),
                        ha='center', va='center', fontsize=5.3, color='white' if abs(q.spearman_r_full) > 0.33 else style.DATA)
            elif i == j:
                r = rel[(rel.unit_class == cls) & (rel.factor == a)].split_half_r.iloc[0]
                ax.text(j, i, f'{r:.2f}'.replace('0.', '.'), ha='center', va='center', fontsize=5.3,
                        color=style.MUTED, style='italic')
    ax.set_xticks(range(5))
    ax.set_xticklabels([FLAB[f] for f in FACT], rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(5))
    ax.set_yticklabels([FLAB[f] for f in FACT], fontsize=6)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.tick_params(length=0)
    n = int(units[units.unit_type == cls].shape[0])
    ax.set_title(f'c  {CLS[cls]["label"]}, n={n}', loc='left')
    if k == 1:
        cb = fig.colorbar(im, cax=ax.inset_axes([0.62, 0.93, 0.38, 0.05]), orientation='horizontal', ticks=[-0.5, 0, 0.5])
        cb.ax.set_title('Spearman r', fontsize=5.5, pad=2)
        cb.ax.tick_params(labelsize=5.5, length=2)
        cb.outline.set_linewidth(0.4)
fig.text(0.05, 0.015, 'c: lower triangle = r across units (* trial-shuffle p<0.05); diagonal (grey italic) = split-half '
         'reliability of each effect', fontsize=5.5, color=style.MUTED)

# ---- (d) forest: r vs null band vs reliability ceiling ------------------------------------------------------------
ax = fig.add_subplot(gs[:, 2])
order = [('reward', 'expected'), ('reward', 'image'), ('reward', 'salience'), ('reward', 'face'),
         ('expected', 'image'), ('expected', 'salience'), ('expected', 'face'),
         ('image', 'salience'), ('image', 'face'), ('salience', 'face')]
yl = []
for i, (a, b) in enumerate(order):
    for k, cls in enumerate(('E', 'I')):
        y = i + (k - 0.5) * 0.36
        q = pair(cls, a, b)
        ax.plot([q.null_trial_shuffle_q025, q.null_trial_shuffle_q975], [y, y], color=style.NULL, lw=3.2,
                solid_capstyle='butt', zorder=1)
        if not np.isnan(q.ceiling_full_data):
            ax.plot([q.ceiling_full_data], [y], marker='|', ms=6, mew=1.2, color=style.DATA, zorder=2)
        ax.plot([q.spearman_r_full], [y], marker='o', ms=3.4, mec=CLS[cls]['color'], mfc=CLS[cls]['mfc'],
                mew=0.8, ls='none', zorder=3)
    yl.append(f'{FLAB[a]}-{FLAB[b]}')
ax.axvline(0, color=style.MUTED, lw=0.5, zorder=0)
ax.axhspan(-0.5, 6.5, color='#f3f3f1', zorder=0, lw=0)
ax.text(0.98, 3.0, 'goal-stimulus\n& goal-goal', fontsize=5.5, color=style.MUTED, ha='right', va='center', rotation=0)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(yl, fontsize=6)
ax.set_ylim(len(order) - 0.5, -0.5)
ax.set_xlim(-0.4, 1.0)
ax.set_xlabel('r across units (factor effects)')
ax.set_title('d  alignment vs null and ceiling', loc='left')
h = [plt.Line2D([], [], color=style.NULL, lw=3.2, label='trial-shuffle null 95%'),
     plt.Line2D([], [], marker='|', ms=6, mew=1.2, color=style.DATA, ls='none',
                label='reliability ceiling = r if one\naxis (SC priority map prediction)'),
     plt.Line2D([], [], marker='o', ms=3.4, color=style.DATA, ls='none', label='classic (E)'),
     plt.Line2D([], [], marker='o', ms=3.4, mec=style.MUTED, mfc='none', ls='none', label='interneuron (I)')]
ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.45, -0.1), fontsize=5.5, ncol=1, handlelength=1.6)
style.save(fig, str(HERE / 'B3_factor_alignment'))
