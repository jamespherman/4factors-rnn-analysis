#!/usr/bin/env python3
"""
Condition-specific PSTHs, recorded vs model: the best- and worst-matched units.

For each session, the free_free model (per-unit context input) restart with the best
selection-set likelihood in results/verified_grid is run on all verified trials (mean
over noise draws). Recorded and model rates are aligned to target onset and averaged
within location x reward condition (8 conditions). A unit's match is the Pearson r
between its recorded and model condition PSTHs, concatenated over conditions. Its
ceiling is the split-half reliability of the recorded condition PSTHs (odd vs even
trials, Spearman-Brown corrected).

Most trials are training trials, so this shows fit quality, not generalization.

Usage:
    python scripts/make_psth_match_figure.py [--variant free_free] [--n-show 6]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import RNNDataset
from src.sessions import load_export, stable_mask, DEFAULT_SESSIONS
from src.analysis.pipeline_selectivity import get_target_onset_bins
from evaluate_context_models import load_context_model, build_inputs, run_with_currents, N_DRAWS

GRID = Path('results/verified_grid')
OUT_DIR = Path('results/psth_match')
BIN_MS = 25.0
WINDOW_BINS = (-12, 24)  # -300 to +600 ms around target onset
COLORS = {1: '#c2570c', 0: '#1f6fb2'}  # high / low reward on this trial
LABELS = {1: 'high reward', 0: 'low reward'}


def best_restart(variant: str, session: str) -> Path:
    """Restart with the best selection-set log-likelihood."""
    dirs = sorted(GRID.glob(f'{variant}/{session}_seed*'))
    return max(dirs, key=lambda d: json.loads((d / 'summary.json').read_text())['select_ll'])


def aligned(rates: np.ndarray, valid: np.ndarray, onset: np.ndarray) -> np.ndarray:
    """[trials, bins, units] -> [trials, window bins, units] around target onset, NaN outside the trial."""
    lo, hi = WINDOW_BINS
    out = np.full((rates.shape[0], hi - lo, rates.shape[2]), np.nan)
    for t, o in enumerate(onset):
        if o < 0:
            continue
        for k, b in enumerate(range(o + lo, o + hi)):
            if 0 <= b < rates.shape[1] and valid[t, b]:
                out[t, k] = rates[t, b]
    return out


def condition_psths(x: np.ndarray, conds: list) -> tuple:
    """Mean and SE per condition: [conditions, window bins, units]."""
    mean = np.stack([np.nanmean(x[c], axis=0) for c in conds])
    n = np.stack([np.sum(~np.isnan(x[c]), axis=0) for c in conds])
    se = np.stack([np.nanstd(x[c], axis=0) for c in conds]) / np.sqrt(np.maximum(n, 1))
    return mean, se


def unit_r(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-unit Pearson r between [conditions, bins, units] arrays (flattened over conditions and bins)."""
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1])
    r = np.full(a.shape[1], np.nan)
    for i in range(a.shape[1]):
        ok = ~np.isnan(a[:, i]) & ~np.isnan(b[:, i])
        if ok.sum() > 3 and np.std(a[ok, i]) > 0 and np.std(b[ok, i]) > 0:
            r[i] = np.corrcoef(a[ok, i], b[ok, i])[0, 1]
    return r


def session_units(variant: str, session: str) -> list:
    """Condition PSTHs and match statistics for every recorded unit of a session."""
    model_dir = best_restart(variant, session)
    model, config = load_context_model(model_dir)
    data = load_export(session, config['trials'])
    inputs = build_inputs(data, config)
    rec_units = np.array(config['recorded_unit_indices'])
    rates = sum(run_with_currents(model, inputs, seed)[0][:, :, rec_units] for seed in range(N_DRAWS)) / N_DRAWS

    recorded = np.transpose(data['firing_rates'], (2, 1, 0))
    rates = np.where(stable_mask(data).T[:, None, :], rates, np.nan)  # compare only on each neuron's stable trials
    mask = RNNDataset(data).get_all_trials()['mask'].numpy()
    valid = (mask[:, :, 0] if mask.ndim == 3 else mask) > 0
    onset = get_target_onset_bins(data)
    rec_x, mod_x = aligned(recorded, valid, onset), aligned(rates, valid, onset)

    location = np.asarray(data['trial_location']).ravel().astype(int)
    reward = np.asarray(data['trial_reward']).ravel().astype(int)
    keys = [(loc, rew) for loc in (1, 2, 3, 4) for rew in (1, 0)]
    conds = [(location == loc) & (reward == rew) & (onset >= 0) for loc, rew in keys]
    rec_m, rec_se = condition_psths(rec_x, conds)
    mod_m, _ = condition_psths(mod_x, conds)
    match = unit_r(rec_m, mod_m)

    # Split-half reliability of the recorded condition PSTHs
    idx = np.arange(len(onset))
    half = [condition_psths(rec_x, [c & (idx % 2 == p) for c in conds])[0] for p in (0, 1)]
    r_half = unit_r(*half)
    ceiling = 2 * r_half / (1 + r_half)

    is_e = np.asarray(data['neuron_type']).ravel() == 1
    ids = np.asarray(data['neuron_ids']).ravel().astype(int)
    vis = slice(-WINDOW_BINS[0] + 2, -WINDOW_BINS[0] + 10)  # 50-250 ms
    units = []
    for i in range(len(rec_units)):
        pref = int(np.nanargmax([np.nanmean(rec_m[[keys.index((loc, 1)), keys.index((loc, 0))], vis, i])
                                 for loc in (1, 2, 3, 4)])) + 1
        units.append({'session': session, 'neuron_id': int(ids[i]), 'type': 'E' if is_e[i] else 'I',
                      'r': float(match[i]), 'ceiling': float(ceiling[i]), 'pref': pref,
                      'rec': rec_m[:, :, i], 'rec_se': rec_se[:, :, i], 'model': mod_m[:, :, i],
                      'mean_rate': float(np.nanmean(rec_m[:, :, i])), 'model_dir': model_dir.as_posix()})
    return units, keys, [int(c.sum()) for c in conds]


def plot_unit(fig, spec, u: dict, keys: list, t: np.ndarray, first_col_label: bool):
    """One unit: four location panels, recorded (solid + SE band) and model (dashed) per reward condition."""
    sub = GridSpecFromSubplotSpec(1, 4, subplot_spec=spec, wspace=0.08)
    ymax = np.nanmax(np.concatenate([u['rec'] + u['rec_se'], u['model']])) * 1.08
    axes = []
    for j, loc in enumerate((1, 2, 3, 4)):
        ax = fig.add_subplot(sub[j])
        for rew in (1, 0):
            c = keys.index((loc, rew))
            ax.fill_between(t, u['rec'][c] - u['rec_se'][c], u['rec'][c] + u['rec_se'][c],
                            color=COLORS[rew], alpha=0.18, lw=0)
            ax.plot(t, u['rec'][c], color=COLORS[rew], lw=1.6)
            ax.plot(t, u['model'][c], color=COLORS[rew], lw=1.4, ls=(0, (3, 1.5)))
        ax.axvline(0, color='0.6', lw=0.6)
        ax.set_ylim(0, ymax)
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(labelsize=6, length=2)
        ax.set_xticks([0, 300])
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        if j > 0:
            ax.set_yticklabels([])
        ax.set_title(f'loc {loc}' + (' (pref)' if loc == u['pref'] else ''), fontsize=6.5, pad=2,
                     fontweight='bold' if loc == u['pref'] else 'normal')
        axes.append(ax)
    axes[0].set_ylabel(f"{u['session']} #{u['neuron_id']} ({u['type']})\n"
                       f"r = {u['r']:.2f}, ceiling {u['ceiling']:.2f}", fontsize=6.5)
    return axes


def main():
    parser = argparse.ArgumentParser(description='Best/worst unit PSTH matches, recorded vs model')
    parser.add_argument('--variant', default='free_free')
    parser.add_argument('--n-show', type=int, default=6)
    parser.add_argument('--min-ceiling', type=float, default=0.6)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    units, n_trials = [], {}
    for session in DEFAULT_SESSIONS:
        u, keys, counts = session_units(args.variant, session)
        units += u
        n_trials[session] = counts
    n_all = len(units)
    units = [u for u in units if not np.isnan(u['r']) and not np.isnan(u['ceiling'])]
    order = sorted(units, key=lambda u: u['r'], reverse=True)
    best, worst = order[:args.n_show], order[-args.n_show:][::-1]
    # Raw worst matches are mostly unreliable units; also show the largest shortfalls
    # below the ceiling among reliable ones
    reliable = [u for u in units if u['ceiling'] >= args.min_ceiling]
    shortfall = sorted(reliable, key=lambda u: u['r'] - u['ceiling'])[:args.n_show]

    t = (np.arange(*WINDOW_BINS) + 0.5) * BIN_MS
    n = args.n_show
    fig = plt.figure(figsize=(19, 2.2 + 1.25 * n))
    outer = GridSpec(n + 2, 3, figure=fig, height_ratios=[2.3, 0.35] + [1] * n, hspace=0.62, wspace=0.16,
                     left=0.05, right=0.99, top=0.95, bottom=0.04)

    # Top: match vs ceiling for all units
    ax = fig.add_subplot(outer[0, 0])
    r = np.array([u['r'] for u in units])
    ceil = np.array([u['ceiling'] for u in units])
    is_e = np.array([u['type'] == 'E' for u in units])
    shown = {id(u) for u in best + worst + shortfall}
    hl = np.array([id(u) in shown for u in units])
    ax.plot([0, 1], [0, 1], color='0.75', lw=0.8, zorder=0)
    ax.scatter(ceil[is_e], r[is_e], s=14, color='#4a4a4a', label='E', zorder=2)
    ax.scatter(ceil[~is_e], r[~is_e], s=14, facecolor='white', edgecolor='#4a4a4a', label='I', zorder=2)
    ax.scatter(ceil[hl], r[hl], s=46, facecolor='none', edgecolor='#c2570c', lw=1.2, label='shown below', zorder=3)
    ax.set_xlabel('recorded split-half reliability (ceiling)', fontsize=8)
    ax.set_ylabel('model vs recorded r', fontsize=8)
    ax.set_xlim(min(0, np.nanmin(ceil)) - 0.02, 1.01)
    ax.set_ylim(min(0, np.nanmin(r)) - 0.02, 1.01)
    ax.tick_params(labelsize=7)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    ax.axvline(args.min_ceiling, color='0.75', lw=0.8, ls=':', zorder=0)
    ax.set_title(f'{len(units)} recorded units, 3 sessions: median r = {np.nanmedian(r):.2f}, '
                 f'median ceiling = {np.nanmedian(ceil):.2f}', fontsize=8.5, loc='left')

    ax = fig.add_subplot(outer[0, 1:])
    ax.axis('off')
    ax.text(0, 1.0,
            'Condition-specific PSTHs, recorded vs model\n'
            f'Model: {args.variant} (per-unit context input), best restart per session by\n'
            'selection-set likelihood, mean over 5 noise draws. Verified successful trials.\n'
            'Conditions: target location (columns) x reward on that trial (colour).\n'
            'r: Pearson r over the 8 condition PSTHs (-300 to +600 ms from target onset).\n'
            'Ceiling: split-half reliability of the recorded PSTHs (Spearman-Brown).\n'
            'About 70% of trials were used for training: this shows fit, not generalization.',
            fontsize=7.5, va='top', family='sans-serif')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS[1], lw=1.6, label='recorded, high reward'),
               Line2D([], [], color=COLORS[1], lw=1.4, ls=(0, (3, 1.5)), label='model, high reward'),
               Line2D([], [], color=COLORS[0], lw=1.6, label='recorded, low reward'),
               Line2D([], [], color=COLORS[0], lw=1.4, ls=(0, (3, 1.5)), label='model, low reward')]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc='lower left', ncol=2)

    for col, group, title in [(0, best, f'Best {n} matches'), (1, worst, f'Worst {n} matches (all units)'),
                              (2, shortfall, f'Largest shortfall below ceiling (ceiling >= {args.min_ceiling})')]:
        for row, u in enumerate(group):
            axes = plot_unit(fig, outer[row + 2, col], u, keys, t, col == 0)
            if row == 0:
                axes[0].annotate(title, xy=(0, 1.55), xycoords='axes fraction', fontsize=9.5, fontweight='bold')
            if row == n - 1:
                for a in axes:
                    a.set_xlabel('ms from target', fontsize=6.5, labelpad=1)
    fig.text(0.005, 0.5, 'firing rate (sp/s)', rotation=90, va='center', fontsize=8)

    stem = OUT_DIR / f'psth_best_worst_{args.variant}'
    fig.savefig(f'{stem}.png', dpi=200)
    fig.savefig(f'{stem}.pdf')
    summary = {'variant': args.variant, 'models': sorted({u['model_dir'] for u in units}),
               'trials_per_condition': {s: dict(zip([f'loc{l}_rew{w}' for l, w in keys], c)) for s, c in n_trials.items()},
               'n_units': n_all, 'r_vs_ceiling_corr': float(np.corrcoef(r, ceil)[0, 1]),
               'median_r': float(np.nanmedian(r)), 'median_ceiling': float(np.nanmedian(ceil)),
               'units': [{k: u[k] for k in ('session', 'neuron_id', 'type', 'r', 'ceiling', 'pref', 'mean_rate')}
                         for u in order]}
    (OUT_DIR / f'psth_match_{args.variant}.json').write_text(json.dumps(summary, indent=1))
    print(f"saved {stem}.png; median r {np.nanmedian(r):.2f}, median ceiling {np.nanmedian(ceil):.2f}")
    for u in best + worst + shortfall:
        print(f"  {u['session']} #{u['neuron_id']} {u['type']} r={u['r']:.2f} ceil={u['ceiling']:.2f} rate={u['mean_rate']:.1f}")


if __name__ == '__main__':
    main()
