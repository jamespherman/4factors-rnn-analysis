#!/usr/bin/env python3
"""
R01 main figure: recorded vs RNN visual-epoch selectivity, all seeds of one model set.

Overlays the per-seed panel data written by compare_recorded_vs_rnn_selectivity.py
(one point per neuron per seed; each seed evaluated on its own held-out trials) and
reports the recorded-vs-RNN correlation for each seed.

Usage:
    python scripts/compare_recorded_vs_rnn_selectivity.py --models cond_target --seed 42  # etc.
    python scripts/make_r01_figure_all_seeds.py --models cond_target --seeds 42 43 44
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from compare_recorded_vs_rnn_selectivity import OUTPUT_DIR, PANELS, PANEL_LABELS

SEED_COLORS = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#A6761D']


def main():
    parser = argparse.ArgumentParser(description='Recorded vs RNN selectivity, all seeds')
    parser.add_argument('--models', default='cond_target',
                        help="cond, cond_target, or <grid>_<variant> for context models, e.g. grid_v12_free_free")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--trials', default='val', choices=['val', 'all'])
    parser.add_argument('--output', default='figure_recorded_vs_rnn')
    args = parser.parse_args()

    runs = {}
    for seed in args.seeds:
        path = OUTPUT_DIR / 'panel_data' / f'figure_recorded_vs_rnn_{args.models}_{args.trials}_seed{seed}.npz'
        assert path.exists(), f"Missing {path}; run compare_recorded_vs_rnn_selectivity.py --seed {seed}"
        runs[seed] = dict(np.load(path))

    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 3.8))
    stats = {}
    for ax, panel in zip(axes, PANELS):
        lines, stats[panel] = [], {}
        for k, (seed, d) in enumerate(runs.items()):
            rec, rnn = d[f'recorded_{panel}'], d[f'rnn_{panel}']
            is_e = d['neuron_type'] == 'E'
            valid = ~np.isnan(rec) & ~np.isnan(rnn)
            color = SEED_COLORS[k % len(SEED_COLORS)]
            ax.scatter(rec[valid & is_e], rnn[valid & is_e], s=14, alpha=0.6, color=color,
                       edgecolors='none', label=f'seed {seed}')
            ax.scatter(rec[valid & ~is_e], rnn[valid & ~is_e], s=14, alpha=0.8, facecolors='none',
                       edgecolors=color, linewidths=1.0)
            r, p = pearsonr(rec[valid], rnn[valid])
            stats[panel][seed] = {'r': float(r), 'p': float(p), 'n': int(valid.sum())}
            lines.append(r)

        lim = 1.05
        ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, linewidth=1)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        rtxt = (' / '.join(f'{r:+.2f}' for r in lines) if len(lines) <= 3 else
                f'{np.mean(lines):+.2f} [{min(lines):+.2f}, {max(lines):+.2f}]')
        ax.text(0.04, 0.96, 'r = ' + rtxt, transform=ax.transAxes, fontsize=8.5,
                va='top', fontfamily='monospace',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))
        ax.set_xlabel('Recorded selectivity', fontsize=10)
        ax.set_ylabel('RNN selectivity', fontsize=10)
        ax.set_title(PANEL_LABELS[panel], fontsize=11)
        if panel == PANELS[0]:
            ax.legend(loc='lower right', fontsize=7.5, framealpha=0.9, markerscale=1.3)

    first = next(iter(runs.values()))
    n_neurons, n_sessions = len(first['neuron_type']), len(np.unique(first['session']))
    context = args.models.startswith('grid')
    fig.suptitle(f"Recorded vs RNN selectivity: {args.models} models, {len(runs)} {'restarts' if context else 'seeds'}, "
                 f"{n_neurons} neurons, {n_sessions} sessions, held-out trials", fontsize=10, y=1.02)
    fig.text(0.5, -0.05, 'Selectivity = 2 x (ROC area - 0.5), 50-250 ms after target onset, at each '
             "neuron's preferred location. Filled: E, open: I. r per seed in legend order (mean [min, max] for more than 3). "
             'RNN: training noise, Poisson spikes, mean of 10 draws.\n'
             + ('* Trial reward and probability are not model inputs; context models receive the block reward map '
                'and probability context.' if context else '* Not available in model inputs before reward.'),
             ha='center', fontsize=8, color='0.3')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{args.output}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / f'{args.output}.pdf', bbox_inches='tight', facecolor='white')
    with open(OUTPUT_DIR / 'panel_data' / f'{args.output}_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({p: {s: round(v['r'], 2) for s, v in d.items()} for p, d in stats.items()}))
    print(f"Saved {OUTPUT_DIR / args.output}.png/.pdf")


if __name__ == '__main__':
    main()
