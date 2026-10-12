#!/usr/bin/env python3
"""
Compare recorded neuronal selectivity vs RNN unit selectivity.

Selectivity follows the neuronal pipeline's window ROC analysis
(src/analysis/pipeline_selectivity.py): 50-250 ms after target onset (located per
trial), trials at each neuron's preferred location, pipeline factor contrasts.
Each recorded neuron is compared with the RNN unit that was fit to it, using the
same trials and the same preferred location.

Model sets:
    legacy:       results/replication (08_13, 08_14) and results/final_model (08_15).
                  These were trained with recorded neuron j fit by model unit j,
                  regardless of cell type, so that is the unit compared here.
    cond:         results/retrained/cond/*_seed{seed} (scripts/train_session.py)
    cond_target:  results/retrained/cond_target/*_seed{seed} (scripts/train_session.py)

Trial sets:
    val: held-out validation trials only (default)
    all: all trials

The location panel compares preferred vs opposite location, with the preferred
location estimated on trials disjoint from the evaluated ones (training trials
for 'val', even trials for 'all') to avoid circularity.

Usage:
    python scripts/compare_recorded_vs_rnn_selectivity.py --models cond_target --trials val
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_mat_file, RNNDataset, train_val_split
from src.model import EIRNN
from src.analysis.pipeline_selectivity import (
    get_target_onset_bins, compute_window_rates, compute_preferred_locations,
    compute_factor_selectivity, compute_location_selectivity
)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'results' / 'r01_figure'

SESSIONS = ['08_13', '08_14', '08_15']
MODEL_SETS = {
    'legacy': {
        '08_13': BASE_DIR / 'results' / 'replication' / 'Newton_08_13_2025_SC',
        '08_14': BASE_DIR / 'results' / 'replication' / 'Newton_08_14_2025_SC',
        '08_15': BASE_DIR / 'results' / 'final_model',
    },
}
RETRAINED_DIR = BASE_DIR / 'results' / 'retrained'


def get_model_dirs(models: str, seed: int) -> dict:
    """Model directory per session for a model set (seed applies to retrained sets)."""
    if models in MODEL_SETS:
        return MODEL_SETS[models]
    return {s: RETRAINED_DIR / models / f'Newton_{s}_2025_SC_seed{seed}' for s in SESSIONS}

# Panels: factors carried by the model inputs first; reward and probability are
# not available to the model before reward delivery
PANELS = ['identity', 'salience', 'location', 'reward', 'probability']
PANEL_LABELS = {
    'identity': 'Identity\n(face vs non-face)',
    'salience': 'Salience\n(high vs low)',
    'location': 'Location\n(pref vs opposite)',
    'reward': 'Reward\n(high vs low)*',
    'probability': 'Probability\n(high vs low)*',
}

COLORS = {
    'e_neurons': '#2166AC',  # Blue
    'i_neurons': '#E66101',  # Orange
}


def load_model(model_dir: Path, n_inputs: int, device: str = 'cpu'):
    """
    Load a trained model and the recorded-neuron-to-unit mapping it was fit with.

    Returns:
        model: EIRNN in eval mode with noise disabled
        config: Model config dict
        recorded_units: [n_recorded] model unit fit to each recorded neuron, or None
            if the config doesn't record it (legacy models: unit j for neuron j)
    """
    with open(model_dir / 'config.json') as f:
        config = json.load(f)
    state_dict = torch.load(model_dir / 'model_best.pt', map_location=device, weights_only=False)

    n_total = state_dict['W_rec_raw'].shape[0]
    n_exc = state_dict['W_out'].shape[1]

    model = EIRNN(
        n_exc=n_exc,
        n_inh=n_total - n_exc,
        n_inputs=n_inputs,
        dt=config.get('dt', 25.0),
        tau=config.get('tau', 50.0),
        noise_scale=0.0,  # Disable noise for inference
        input_embed_dim=config.get('input_embed_dim', 56),
        input_embed_type=config.get('input_embed_type', 'attention'),
        attention_heads=config.get('attention_heads', 4),
        learnable_h0=config.get('learnable_h0', True),
        h0_init=config.get('h0_init', 0.1),
        arch_version=config.get('arch_version', 'jan2026'),
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    recorded_units = config.get('recorded_unit_indices')
    return model, config, (np.array(recorded_units) if recorded_units is not None else None)


def get_rnn_rates(model: EIRNN, dataset: RNNDataset, device: str = 'cpu') -> np.ndarray:
    """Run forward pass and get RNN firing rates [n_trials, n_bins, n_units]."""
    inputs = dataset.get_all_trials()['inputs'].to(device)
    with torch.no_grad():
        rates, _ = model(inputs)
    return rates.cpu().numpy()


def process_session(session_id: str, model_dir: Path, trial_set: str, device: str = 'cpu') -> dict:
    """Compute recorded and RNN selectivity for one session."""
    print(f"\nProcessing session {session_id} ({model_dir.relative_to(BASE_DIR)})...")

    data = load_mat_file(str(DATA_DIR / f'rnn_export_Newton_{session_id}_2025_SC.mat'))
    assert float(data['bin_size_ms']) == 25.0, "Expected 25 ms bins"
    neuron_type = np.asarray(data['neuron_type']).ravel()
    n_trials = data['firing_rates'].shape[2]

    dataset = RNNDataset(data)
    model, config, recorded_units = load_model(model_dir, dataset.get_input_dim(), device)
    if recorded_units is None:
        recorded_units = np.arange(len(neuron_type))
    assert len(recorded_units) == len(neuron_type), "Mapping doesn't cover all recorded neurons"
    unit_is_e = recorded_units < model.n_exc
    print(f"  Neurons: {(neuron_type == 1).sum()} E, {(neuron_type == 2).sum()} I; "
          f"fit by {unit_is_e.sum()} E units, {(~unit_is_e).sum()} I units "
          f"({np.sum(unit_is_e != (neuron_type == 1))} type mismatches)")

    # Evaluation trials
    if 'val_idx' in config:
        val_idx = np.array(config['val_idx'], dtype=int)
    else:
        _, val_idx = train_val_split(dataset, val_fraction=0.2, seed=config.get('seed', 42))
        val_idx = np.array(val_idx, dtype=int)
    is_val = np.zeros(n_trials, dtype=bool)
    is_val[val_idx] = True

    if trial_set == 'val':
        eval_mask, loc_pref_mask = is_val, ~is_val
    else:
        eval_mask = np.ones(n_trials, dtype=bool)
        eval_mask_odd = (np.arange(n_trials) % 2) == 1
        loc_pref_mask = ~eval_mask_odd

    # Visual-epoch rates
    onset_bins = get_target_onset_bins(data)
    recorded_rates = compute_window_rates(np.transpose(data['firing_rates'], (2, 1, 0)), onset_bins)
    rnn_rates = compute_window_rates(get_rnn_rates(model, dataset, device)[:, :, recorded_units], onset_bins)
    print(f"  Trials with valid visual epoch: {(onset_bins >= 0).sum()}/{n_trials}; "
          f"evaluated on {trial_set} ({(eval_mask & (onset_bins >= 0)).sum()} usable)")

    # Preferred locations come from the recorded neuron and are shared with its unit
    trial_location = np.asarray(data['trial_location']).ravel()
    preferred = compute_preferred_locations(recorded_rates, trial_location)
    loc_preferred = compute_preferred_locations(recorded_rates, trial_location, loc_pref_mask)

    result = {'session': session_id, 'neuron_type': neuron_type, 'unit_is_e': unit_is_e}
    for name, rates in [('recorded', recorded_rates), ('rnn', rnn_rates)]:
        factors = compute_factor_selectivity(rates, data, preferred, eval_mask)
        sel = {f: factors[f]['selectivity'] for f in factors}
        loc_mask = eval_mask_odd if trial_set == 'all' else eval_mask
        sel['location'] = compute_location_selectivity(
            rates, trial_location, loc_preferred['for_factors'], loc_mask)
        result[name] = sel
    return result


def summarize(all_data: list) -> dict:
    """Pearson correlation between recorded and RNN selectivity per panel."""
    stats = {}
    for panel in PANELS:
        rec = np.concatenate([d['recorded'][panel] for d in all_data])
        rnn = np.concatenate([d['rnn'][panel] for d in all_data])
        is_e = np.concatenate([d['neuron_type'] == 1 for d in all_data])
        valid = ~np.isnan(rec) & ~np.isnan(rnn)
        entry = {'n': int(valid.sum()), 'n_e': int((valid & is_e).sum()), 'n_i': int((valid & ~is_e).sum()),
                 'mean_abs_recorded': float(np.abs(rec[valid]).mean()) if valid.any() else np.nan,
                 'mean_abs_rnn': float(np.abs(rnn[valid]).mean()) if valid.any() else np.nan}
        for label, sub in [('all', valid), ('e', valid & is_e), ('i', valid & ~is_e)]:
            if sub.sum() > 2:
                r, p = pearsonr(rec[sub], rnn[sub])
                entry[f'r_{label}'], entry[f'p_{label}'] = float(r), float(p)
        stats[panel] = entry
    return stats


def create_figure(all_data: list, stats: dict, title: str):
    """1 x 5 scatter of recorded vs RNN selectivity, E filled and I open."""
    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 3.6))

    for ax, panel in zip(axes, PANELS):
        rec = np.concatenate([d['recorded'][panel] for d in all_data])
        rnn = np.concatenate([d['rnn'][panel] for d in all_data])
        is_e = np.concatenate([d['neuron_type'] == 1 for d in all_data])
        valid = ~np.isnan(rec) & ~np.isnan(rnn)

        ax.scatter(rec[valid & is_e], rnn[valid & is_e], c=COLORS['e_neurons'], s=30, alpha=0.7,
                   label=f'E (n={(valid & is_e).sum()})', edgecolors='none')
        ax.scatter(rec[valid & ~is_e], rnn[valid & ~is_e], s=30, alpha=0.7, marker='o',
                   facecolors='none', edgecolors=COLORS['i_neurons'], linewidths=1.5,
                   label=f'I (n={(valid & ~is_e).sum()})')

        lim = 1.05
        ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, linewidth=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.set_aspect('equal')

        if 'r_all' in stats[panel]:
            ax.text(0.04, 0.96, f"r = {stats[panel]['r_all']:.2f}\np = {stats[panel]['p_all']:.3g}",
                    transform=ax.transAxes, fontsize=9, va='top', fontfamily='monospace',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

        ax.set_xlabel('Recorded selectivity', fontsize=10)
        ax.set_ylabel('RNN selectivity', fontsize=10)
        ax.set_title(PANEL_LABELS[panel], fontsize=11)
        if panel == PANELS[0]:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    fig.suptitle(title, fontsize=10, y=1.02)
    fig.text(0.5, -0.04, 'Selectivity = 2 x (ROC area - 0.5), 50-250 ms after target onset, '
             "at each neuron's preferred location.  * Not available in model inputs before reward.",
             ha='center', fontsize=8, color='0.3')
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Recorded vs RNN visual-epoch selectivity')
    parser.add_argument('--models', default='cond_target', choices=['legacy', 'cond', 'cond_target'])
    parser.add_argument('--seed', type=int, default=42, help='Seed of retrained models')
    parser.add_argument('--trials', default='val', choices=['val', 'all'])
    args = parser.parse_args()

    device = 'cpu'
    model_dirs = get_model_dirs(args.models, args.seed)
    all_data = [process_session(s, model_dirs[s], args.trials, device) for s in SESSIONS]
    stats = summarize(all_data)

    print("\nRecorded vs RNN selectivity (Pearson r, all neurons):")
    for panel, s in stats.items():
        print(f"  {panel:12s} n={s['n']:3d}  r={s.get('r_all', np.nan):+.2f}  p={s.get('p_all', np.nan):.3g}  "
              f"mean|rec|={s['mean_abs_recorded']:.3f}  mean|rnn|={s['mean_abs_rnn']:.3f}")

    label = args.models if args.models == 'legacy' else f'{args.models} (seed {args.seed})'
    title = f"Models: {label}  |  trials: {'held-out validation' if args.trials == 'val' else 'all'}"
    fig = create_figure(all_data, stats, title)

    stem = f'figure_recorded_vs_rnn_{args.models}_{args.trials}'
    if args.models != 'legacy':
        stem += f'_seed{args.seed}'
    (OUTPUT_DIR / 'panel_data').mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f'{stem}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    np.savez(OUTPUT_DIR / 'panel_data' / f'{stem}.npz',
             **{f'recorded_{p}': np.concatenate([d['recorded'][p] for d in all_data]) for p in PANELS},
             **{f'rnn_{p}': np.concatenate([d['rnn'][p] for d in all_data]) for p in PANELS},
             neuron_type=np.concatenate([np.where(d['neuron_type'] == 1, 'E', 'I') for d in all_data]),
             unit_is_e=np.concatenate([d['unit_is_e'] for d in all_data]),
             session=np.concatenate([[d['session']] * len(d['neuron_type']) for d in all_data]))
    with open(OUTPUT_DIR / 'panel_data' / f'{stem}_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved {OUTPUT_DIR / stem}.png/.pdf and panel data")


if __name__ == '__main__':
    main()
