#!/usr/bin/env python3
"""
Compare recorded neuronal selectivity vs RNN neuron selectivity.

Creates a 1x4 panel figure showing recorded vs RNN selectivity for each factor.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_mat_file, RNNDataset
from model import EIRNN, create_model_from_data

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'results' / 'r01_figure'

# Sessions and their model directories
SESSIONS = {
    '08_13': {
        'data': DATA_DIR / 'rnn_export_Newton_08_13_2025_SC.mat',
        'model': BASE_DIR / 'results' / 'replication' / 'Newton_08_13_2025_SC',
    },
    '08_14': {
        'data': DATA_DIR / 'rnn_export_Newton_08_14_2025_SC.mat',
        'model': BASE_DIR / 'results' / 'replication' / 'Newton_08_14_2025_SC',
    },
    '08_15': {
        'data': DATA_DIR / 'rnn_export_Newton_08_15_2025_SC.mat',
        'model': BASE_DIR / 'results' / 'final_model',
    },
}

# Colors
COLORS = {
    'e_neurons': '#2166AC',  # Blue
    'i_neurons': '#E66101',  # Orange
}


def compute_roc_selectivity(firing_rates, trial_labels):
    """
    Compute ROC-based selectivity for each neuron.

    Args:
        firing_rates: [n_trials, n_neurons] - Trial-averaged firing rates
        trial_labels: [n_trials] - Binary labels (0/1)

    Returns:
        selectivity: [n_neurons] - Selectivity index: 2*(AUC - 0.5)
    """
    n_trials, n_neurons = firing_rates.shape
    selectivity = np.zeros(n_neurons)

    labels = np.array(trial_labels)
    unique_labels = np.unique(labels[~np.isnan(labels)])

    if len(unique_labels) < 2:
        return selectivity

    for i in range(n_neurons):
        rates = firing_rates[:, i]
        valid = ~np.isnan(rates) & ~np.isnan(labels)

        if valid.sum() < 10:
            selectivity[i] = np.nan
            continue

        try:
            auc = roc_auc_score(labels[valid], rates[valid])
            selectivity[i] = 2 * (auc - 0.5)
        except Exception:
            selectivity[i] = np.nan

    return selectivity


def get_neuron_counts(data):
    """Get E and I neuron counts from data."""
    neuron_type = data['neuron_type']
    if neuron_type.dtype.kind == 'U':  # Unicode string
        n_e = np.sum(neuron_type == 'E')
        n_i = np.sum(neuron_type == 'I')
    else:
        # Numeric encoding: 1 = classic (E), 2 = interneuron (I)
        neuron_type_flat = neuron_type.flatten()
        n_e = np.sum(neuron_type_flat == 1)
        n_i = np.sum(neuron_type_flat == 2)
    return int(n_e), int(n_i)


def load_model(model_dir, n_e_recorded, n_i_recorded, n_inputs, device='cpu'):
    """Load trained model from checkpoint."""
    import json

    model_path = model_dir / 'model_best.pt'
    config_path = model_dir / 'config.json'

    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        return None

    # Load config
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    # Load state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    # Determine model architecture from state dict
    W_rec_shape = state_dict['W_rec_raw'].shape
    n_total = W_rec_shape[0]
    n_exc = state_dict['W_out'].shape[1]  # Output neurons = E neurons
    n_inh = n_total - n_exc

    print(f"  Model: {n_exc} E, {n_inh} I (total {n_total})")

    # Create model
    model = EIRNN(
        n_inputs=config.get('input_embed_dim', 56),
        n_exc=n_exc,
        n_inh=n_inh,
        n_outputs=2,
        dt=config.get('dt', 25.0),
        tau=config.get('tau', 50.0),
        noise_scale=0.0,  # Disable noise for inference
        learnable_h0=config.get('learnable_h0', True),
        h0_init=config.get('h0_init', 0.1),
    )

    # Add input embedding
    from model import InputEmbedding
    model.input_embed = InputEmbedding(
        n_inputs=n_inputs,
        embed_dim=config.get('input_embed_dim', 56),
        embed_type=config.get('input_embed_type', 'attention'),
        attention_heads=config.get('attention_heads', 4),
    )

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Store architecture info in config for later use
    config['n_exc'] = n_exc
    config['n_inh'] = n_inh
    config['n_total'] = n_total

    return model, config


def get_rnn_rates(model, dataset, device='cpu'):
    """Run forward pass and get RNN firing rates."""
    model.eval()

    all_trials = dataset.get_all_trials()
    inputs = all_trials['inputs'].to(device)  # [n_trials, time, n_inputs]

    with torch.no_grad():
        rates, _ = model(inputs)  # rates: [n_trials, time, n_neurons]

    # Convert to numpy and average over time
    rates = rates.cpu().numpy()
    mean_rates = rates.mean(axis=1)  # [n_trials, n_neurons]

    return mean_rates


def process_session(session_id, session_info, device='cpu'):
    """Process a single session and return selectivity data."""
    print(f"\nProcessing session {session_id}...")

    # Load data
    data = load_mat_file(str(session_info['data']))
    n_e, n_i = get_neuron_counts(data)
    n_neurons = n_e + n_i
    n_trials = int(data['n_trials'].item() if data['n_trials'].size == 1 else data['n_trials'][0])

    print(f"  Neurons: {n_e} E, {n_i} I")
    print(f"  Trials: {n_trials}")

    # Get recorded firing rates
    firing_rates = data['firing_rates']  # [n_neurons, time, n_trials]
    recorded_mean_rates = firing_rates.mean(axis=1).T  # [n_trials, n_neurons]

    # Get trial labels
    trial_reward = data['trial_reward'].flatten()
    trial_salience_raw = data.get('trial_salience', np.zeros(n_trials)).flatten()
    trial_location = data.get('trial_location', np.zeros(n_trials)).flatten()
    trial_identity = data.get('trial_identity', np.zeros(n_trials)).flatten()

    # Binarize salience: High (2) vs Low (0)
    salience_mask = (trial_salience_raw == 0) | (trial_salience_raw == 2)
    trial_salience_binary = (trial_salience_raw == 2).astype(float)
    trial_salience_binary[~salience_mask] = np.nan

    # Binarize identity: Face (1) vs Non-face (2)
    identity_mask = (trial_identity == 1) | (trial_identity == 2)
    trial_identity_binary = (trial_identity == 1).astype(float)
    trial_identity_binary[~identity_mask] = np.nan

    # Binarize location: Use most common vs others, or location 0 vs others
    unique_locs = np.unique(trial_location[~np.isnan(trial_location)])
    if len(unique_locs) >= 2:
        # Use first location vs all others
        trial_location_binary = (trial_location == unique_locs[0]).astype(float)
    else:
        trial_location_binary = np.full(n_trials, np.nan)

    # Compute recorded selectivity
    recorded_sel = {
        'reward': compute_roc_selectivity(recorded_mean_rates, trial_reward),
        'salience': compute_roc_selectivity(recorded_mean_rates, trial_salience_binary),
        'location': compute_roc_selectivity(recorded_mean_rates, trial_location_binary),
        'identity': compute_roc_selectivity(recorded_mean_rates, trial_identity_binary),
    }

    # Load model and get RNN rates
    dataset = RNNDataset(data)
    n_inputs = dataset[0]['inputs'].shape[1]

    model_result = load_model(session_info['model'], n_e, n_i, n_inputs, device)
    if model_result is None:
        return None

    model, config = model_result
    rnn_mean_rates = get_rnn_rates(model, dataset, device)

    # Get only the fitted neurons (first n_e E neurons and first n_i I neurons)
    n_exc_model = config.get('n_exc', 160)
    n_inh_model = config.get('n_inh', 40)

    # Fitted E neurons are first n_e of the E population
    # Fitted I neurons are first n_i of the I population
    fitted_e_idx = list(range(n_e))
    fitted_i_idx = list(range(n_exc_model, n_exc_model + n_i))
    fitted_idx = fitted_e_idx + fitted_i_idx

    rnn_fitted_rates = rnn_mean_rates[:, fitted_idx]  # [n_trials, n_e + n_i]

    # Compute RNN selectivity
    rnn_sel = {
        'reward': compute_roc_selectivity(rnn_fitted_rates, trial_reward),
        'salience': compute_roc_selectivity(rnn_fitted_rates, trial_salience_binary),
        'location': compute_roc_selectivity(rnn_fitted_rates, trial_location_binary),
        'identity': compute_roc_selectivity(rnn_fitted_rates, trial_identity_binary),
    }

    # Create neuron type labels
    neuron_types = ['E'] * n_e + ['I'] * n_i

    return {
        'recorded_sel': recorded_sel,
        'rnn_sel': rnn_sel,
        'neuron_types': neuron_types,
        'n_e': n_e,
        'n_i': n_i,
        'session': session_id,
    }


def create_figure(all_data):
    """Create the 1x4 panel figure."""
    factors = ['reward', 'salience', 'location', 'identity']
    factor_labels = {
        'reward': 'Reward\n(goal-directed)',
        'salience': 'Salience\n(high vs low)',
        'location': 'Location\n(pref vs non-pref)',
        'identity': 'Identity\n(face vs non-face)',
    }

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

    for ax, factor in zip(axes, factors):
        # Collect data across sessions
        recorded_e, rnn_e = [], []
        recorded_i, rnn_i = [], []

        for session_data in all_data:
            n_e = session_data['n_e']
            n_i = session_data['n_i']

            rec = session_data['recorded_sel'][factor]
            rnn = session_data['rnn_sel'][factor]

            # E neurons
            recorded_e.extend(rec[:n_e])
            rnn_e.extend(rnn[:n_e])

            # I neurons
            recorded_i.extend(rec[n_e:n_e+n_i])
            rnn_i.extend(rnn[n_e:n_e+n_i])

        recorded_e = np.array(recorded_e)
        rnn_e = np.array(rnn_e)
        recorded_i = np.array(recorded_i)
        rnn_i = np.array(rnn_i)

        # Remove NaN
        valid_e = ~np.isnan(recorded_e) & ~np.isnan(rnn_e)
        valid_i = ~np.isnan(recorded_i) & ~np.isnan(rnn_i)

        # Plot E neurons
        ax.scatter(recorded_e[valid_e], rnn_e[valid_e],
                   c=COLORS['e_neurons'], s=30, alpha=0.7,
                   label=f'E (n={valid_e.sum()})', edgecolors='none')

        # Plot I neurons (open circles)
        ax.scatter(recorded_i[valid_i], rnn_i[valid_i],
                   s=30, alpha=0.7,
                   marker='o', facecolors='none', edgecolors=COLORS['i_neurons'],
                   linewidths=1.5, label=f'I (n={valid_i.sum()})')

        # Unity line
        lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
                  abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
        lim = max(lim, 0.5)
        ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, linewidth=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        # Compute correlation (all neurons)
        all_rec = np.concatenate([recorded_e[valid_e], recorded_i[valid_i]])
        all_rnn = np.concatenate([rnn_e[valid_e], rnn_i[valid_i]])

        if len(all_rec) > 2:
            r, p = pearsonr(all_rec, all_rnn)
            ax.text(0.05, 0.95, f'r = {r:.2f}\np = {p:.3f}',
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace')

        ax.set_xlabel('Recorded selectivity', fontsize=10)
        ax.set_ylabel('RNN selectivity', fontsize=10)
        ax.set_title(factor_labels[factor], fontsize=11)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.set_aspect('equal')

        if factor == 'reward':
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    plt.tight_layout()

    return fig


def main():
    print("=" * 60)
    print("Comparing Recorded vs RNN Neuron Selectivity")
    print("=" * 60)

    device = 'cpu'

    # Process each session
    all_data = []
    for session_id, session_info in SESSIONS.items():
        result = process_session(session_id, session_info, device)
        if result is not None:
            all_data.append(result)

    if not all_data:
        print("No data processed!")
        return

    # Create figure
    print("\nCreating figure...")
    fig = create_figure(all_data)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = OUTPUT_DIR / 'figure_recorded_vs_rnn.png'
    pdf_path = OUTPUT_DIR / 'figure_recorded_vs_rnn.pdf'

    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')

    print(f"\nFigure saved to:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")

    # Save data
    data_path = OUTPUT_DIR / 'panel_data' / 'recorded_vs_rnn_data.npz'

    # Combine data for saving
    all_recorded = {f: [] for f in ['reward', 'salience', 'location', 'identity']}
    all_rnn = {f: [] for f in ['reward', 'salience', 'location', 'identity']}
    all_types = []
    all_sessions = []

    for session_data in all_data:
        for factor in all_recorded.keys():
            all_recorded[factor].extend(session_data['recorded_sel'][factor])
            all_rnn[factor].extend(session_data['rnn_sel'][factor])
        all_types.extend(session_data['neuron_types'])
        all_sessions.extend([session_data['session']] * (session_data['n_e'] + session_data['n_i']))

    np.savez(data_path,
             recorded_reward=np.array(all_recorded['reward']),
             recorded_salience=np.array(all_recorded['salience']),
             recorded_location=np.array(all_recorded['location']),
             recorded_identity=np.array(all_recorded['identity']),
             rnn_reward=np.array(all_rnn['reward']),
             rnn_salience=np.array(all_rnn['salience']),
             rnn_location=np.array(all_rnn['location']),
             rnn_identity=np.array(all_rnn['identity']),
             neuron_type=np.array(all_types),
             session=np.array(all_sessions))

    print(f"  Data: {data_path}")

    plt.close()


if __name__ == '__main__':
    main()
