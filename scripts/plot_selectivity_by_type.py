#!/usr/bin/env python3
"""
Scatter of recorded vs RNN ROC area per factor: classic SC neurons vs their RNN E units (top row) and
putative interneurons vs their RNN I units (bottom row). Circles Feynman, triangles Newton; filled markers
are neurons whose recorded selectivity is significant (permutation p < 0.05).

RNN ROC area is the mean over noise draws and restarts, so it is less noisy than a single recorded value
and sits closer to 0.5 for units without consistent selectivity.

Usage:
    python scripts/plot_selectivity_by_type.py   (after compare_selectivity_by_type.py)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

OUT = Path(__file__).parent.parent / 'results' / 'selectivity_by_type'
STEM = 'grid_v13_free_free'
FACTORS = ['location', 'identity', 'salience', 'reward', 'probability']
LABELS = {'location': 'Location\n(preferred vs opposite)', 'identity': 'Identity\n(face vs non-face)',
          'salience': 'Salience\n(high vs low)', 'reward': 'Reward\n(high vs low)',
          'probability': 'Probability\n(high vs low)'}
ROWS = [('E', 'Classic SC neurons', 'RNN E units', '#2166AC'),
        ('I', 'Putative interneurons', 'RNN I units', '#E66101')]
MONKEYS = [('Feynman', 'o'), ('Newton', '^')]


def main():
    d = np.load(OUT / f'{STEM}.npz')
    monkey = np.array(['Feynman' if s.startswith('Feynman') else 'Newton' for s in d['session']])
    fig, axes = plt.subplots(2, len(FACTORS), figsize=(16, 7.2))
    for row, (t, xlab, ylab, color) in enumerate(ROWS):
        for col, f in enumerate(FACTORS):
            ax = axes[row, col]
            rec = d[f'recorded_{f}']
            x_all = rec[:, 0] / 2 + 0.5
            y_all = np.nanmean(d[f'rnn_{f}'][..., 0], axis=(1, 2)) / 2 + 0.5
            lines = []
            for m, marker in MONKEYS:
                sub = (d['neuron_type'] == t) & (monkey == m) & ~np.isnan(x_all) & ~np.isnan(y_all)
                sig = sub & (rec[:, 1] < 0.05)
                ax.scatter(x_all[sig], y_all[sig], marker=marker, s=26, c=color, edgecolors=color,
                           linewidths=0.8, alpha=0.8)
                ax.scatter(x_all[sub & ~sig], y_all[sub & ~sig], marker=marker, s=26, facecolors='none',
                           edgecolors=color, linewidths=0.8, alpha=0.6)
                r = pearsonr(x_all[sub], y_all[sub])[0]
                lines.append(f'{m[0]}: r={r:+.2f} n={sub.sum()}')
            ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
            ax.axhline(0.5, color='0.7', lw=0.5)
            ax.axvline(0.5, color='0.7', lw=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.text(0.03, 0.97, '\n'.join(lines), transform=ax.transAxes, va='top', fontsize=7.5,
                    fontfamily='monospace', bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1))
            if row == 0:
                ax.set_title(LABELS[f], fontsize=10)
            ax.set_xlabel(f'{xlab}: ROC area', fontsize=8.5)
            ax.set_ylabel(f'{ylab}: ROC area', fontsize=8.5)
            ax.tick_params(labelsize=7.5)
    handles = [plt.Line2D([], [], marker=mk, ls='', color='0.3', label=m) for m, mk in MONKEYS]
    handles += [plt.Line2D([], [], marker='o', ls='', color='0.3', label='recorded p < 0.05'),
                plt.Line2D([], [], marker='o', ls='', mfc='none', color='0.3', label='recorded n.s.')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Visual epoch (50-250 ms after target onset), preferred location. RNN: per-unit context models '
                 '(grid_v13), mean of 6 restarts x 5 Poisson draws', fontsize=10)
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'scatter_recorded_vs_rnn_roc_{STEM}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"saved {OUT / f'scatter_recorded_vs_rnn_roc_{STEM}'}.png/.pdf")


if __name__ == '__main__':
    main()
