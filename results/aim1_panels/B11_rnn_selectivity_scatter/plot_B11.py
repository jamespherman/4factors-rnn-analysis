#!/usr/bin/env python3
"""B11: recorded vs data-fitted RNN visual-epoch ROC selectivity, per factor (columns) and cell class (rows).
Reproduces scatter_recorded_vs_rnn_roc_grid_v13_free_free.{png,pdf} (the original figure, moved here) from
B11_units.csv, in the shared grant style. Reads only the CSVs in this folder (plus style.py). Run: python plot_B11.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style

HERE = Path(__file__).resolve().parent
style.apply()

U = pd.read_csv(HERE / 'B11_units.csv')
FACTORS = ['location', 'identity', 'salience', 'reward', 'probability']
LABELS = {'location': 'Location\n(pref. vs opposite)', 'identity': 'Identity\n(face vs non-face)',
          'salience': 'Salience\n(high vs low)', 'reward': 'Reward\n(high vs low)',
          'probability': 'Probability\n(high vs low)'}
ROWS = [('E', 'Classic SC neuron', 'RNN E unit'), ('I', 'Putative interneuron', 'RNN I unit')]
MONKEYS = [('Feynman', 'o'), ('Newton', '^')]

fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.3))
for row, (t, xlab, ylab) in enumerate(ROWS):
    for col, f in enumerate(FACTORS):
        ax = axes[row, col]
        g = U[(U.factor == f) & (U.unit_type == t)].dropna(subset=['recorded_roc_auc', 'rnn_roc_auc_mean'])
        lines = []
        for m, mk in MONKEYS:
            h = g[g.monkey == m]
            sig = h.recorded_significant.to_numpy(bool)
            x, y = h.recorded_roc_auc.to_numpy(), h.rnn_roc_auc_mean.to_numpy()
            ax.scatter(x[sig], y[sig], marker=mk, s=6, color=style.DATA, lw=0, alpha=0.75)
            ax.scatter(x[~sig], y[~sig], marker=mk, s=6, facecolors='none', edgecolors=style.MUTED, lw=0.4)
            lines.append(f'{m[0]} r = {np.corrcoef(x, y)[0, 1]:+.2f}')
        ax.plot([0, 1], [0, 1], color=style.MUTED, ls='--', lw=0.6)
        ax.axhline(0.5, color=style.NULL, lw=0.5, zorder=0)
        ax.axvline(0.5, color=style.NULL, lw=0.5, zorder=0)
        ax.set_xlim(0.15, 1)
        ax.set_ylim(0.15, 1)
        ax.set_aspect('equal')
        ax.set_xticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.text(0.04, 0.97, '\n'.join(lines), transform=ax.transAxes, va='top', fontsize=5.5)
        if row == 0:
            ax.set_title(LABELS[f], fontsize=6.5)
        if col == 0:
            ax.set_ylabel(f'{ylab}: ROC area')
        else:
            ax.set_yticklabels([])
        if row == 1:
            ax.set_xlabel('Recorded ROC area')
        else:
            ax.set_xticklabels([])
    axes[row, 0].annotate(xlab, xy=(-0.62, 0.5), xycoords='axes fraction', rotation=90, va='center',
                          ha='center', fontsize=7, fontweight='bold')
handles = [plt.Line2D([], [], marker=mk, ls='', color=style.DATA, ms=3, label=m) for m, mk in MONKEYS]
handles += [plt.Line2D([], [], marker='o', ls='', color=style.DATA, ms=3, label='recorded p < 0.05'),
            plt.Line2D([], [], marker='o', ls='', mfc='none', color=style.MUTED, ms=3, label='recorded n.s.')]
fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.3, h_pad=0.6)
style.save(fig, str(HERE / 'B11_rnn_selectivity_scatter'))
print('saved', HERE / 'B11_rnn_selectivity_scatter.png')
