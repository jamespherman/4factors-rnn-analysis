#!/usr/bin/env python3
"""
Generate 1×4 panel figure comparing recorded vs RNN neuron selectivity.

This creates the figure specified in specs/recorded_vs_rnn_selectivity_plan.md
using the conditioned-loss model from results/conditioned_loss_08_15/.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model_from_data
from src.data_loader import load_session, train_val_split

# Paths
BASE_DIR = Path("/Users/jph/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Code/4factors-rnn-analysis")
MODEL_DIR = BASE_DIR / "results/conditioned_loss_08_15"
DATA_FILE = BASE_DIR / "data/rnn_export_Newton_08_15_2025_SC.mat"
OUTPUT_DIR = MODEL_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors (from plan)
E_COLOR = '#2166AC'  # Blue
I_COLOR = '#E66101'  # Orange


def compute_roc_selectivity(rates: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute ROC-based selectivity for each neuron.

    Args:
        rates: [n_trials, n_neurons] mean firing rates per trial
        labels: [n_trials] binary labels (0 or 1)

    Returns:
        selectivity: [n_neurons] selectivity index in [-1, 1]
            +1 = perfect preference for label=1
            -1 = perfect preference for label=0
            0 = no selectivity
    """
    n_neurons = rates.shape[1]
    selectivity = np.zeros(n_neurons)

    # Need at least some trials in each class
    n_low = (labels == 0).sum()
    n_high = (labels == 1).sum()

    if n_low < 3 or n_high < 3:
        return selectivity

    for i in range(n_neurons):
        neuron_rates = rates[:, i]
        try:
            auc = roc_auc_score(labels, neuron_rates)
            # Convert AUC to selectivity: 2 * (AUC - 0.5)
            selectivity[i] = 2 * (auc - 0.5)
        except:
            selectivity[i] = 0

    return selectivity


def load_data_and_model():
    """Load the dataset and trained model."""
    print("Loading data...")
    dataset = load_session(str(DATA_FILE))
    train_idx, val_idx = train_val_split(dataset, val_fraction=0.2, seed=42)

    # Get all data with conditions
    all_data = dataset.get_all_trials(include_conditions=True)

    # Use validation set for evaluation
    val_data = {
        'inputs': all_data['inputs'][val_idx],
        'targets': all_data['targets'][val_idx],
        'trial_conditions': all_data['trial_conditions'][val_idx],
        'trial_reward': all_data['trial_reward'][val_idx],
        'trial_location': all_data['trial_location'][val_idx],
        'trial_salience': all_data['trial_salience'][val_idx],
    }

    # Get trial info from raw data for identity and probability
    trial_identity = dataset.trial_identity[val_idx]
    val_data['trial_identity'] = torch.tensor(trial_identity, dtype=torch.long)

    trial_probability = dataset.trial_probability[val_idx]
    val_data['trial_probability'] = torch.tensor(trial_probability, dtype=torch.long)

    neuron_info = dataset.get_neuron_info()
    n_inputs = dataset.get_input_dim()

    print("Loading model...")
    device = 'cpu'
    model = create_model_from_data(
        n_classic=neuron_info['n_exc'],
        n_interneuron=neuron_info['n_inh'],
        n_inputs=n_inputs,
        enforce_ratio=True,
        dt=25.0,
        tau=50.0,
        noise_scale=0.1,
        spectral_radius=0.9,
        input_embed_dim=56,
        input_embed_type='attention',
        attention_heads=4,
        learnable_h0=True,
        h0_init=0.1,
        device=device
    )
    model.load_state_dict(torch.load(MODEL_DIR / 'model_best.pt', weights_only=True))
    model.eval()

    return model, val_data, neuron_info


def get_model_rates(model, val_data, n_recorded):
    """Get trial-averaged model firing rates."""
    with torch.no_grad():
        inputs = val_data['inputs']
        model_rates, _ = model(inputs)
        # Limit to recorded neurons
        model_rates = model_rates[:, :, :n_recorded]
        # Time-average
        return model_rates.mean(dim=1).numpy()


def get_model_rates_window(model, val_data, n_recorded, start_bin, end_bin):
    """Get model firing rates averaged within a specific time window.

    Args:
        model: The RNN model
        val_data: Validation data dictionary with 'inputs' key
        n_recorded: Number of recorded neurons to include
        start_bin: Starting time bin (inclusive)
        end_bin: Ending time bin (exclusive)

    Returns:
        rates: [n_trials, n_neurons] mean firing rates in the window
    """
    with torch.no_grad():
        inputs = val_data['inputs']
        model_rates, _ = model(inputs)
        # Limit to recorded neurons
        model_rates = model_rates[:, :, :n_recorded]
        # Average within the specified time window
        return model_rates[:, start_bin:end_bin, :].mean(dim=1).numpy()


def compute_all_selectivity(model, val_data, neuron_info, bin_size_ms=25.0):
    """Compute selectivity for all factors for both recorded and RNN.

    Uses firing rates in the 50-250ms window after target onset.
    """
    n_recorded = neuron_info['n_total']
    n_exc = neuron_info['n_exc']

    # Time window: 50-250ms after target onset
    # With bin_size_ms=25, this is bins 2-10 (indices 2:10)
    start_bin = int(50 / bin_size_ms)   # bin 2
    end_bin = int(250 / bin_size_ms)    # bin 10
    print(f"Using time window: {start_bin * bin_size_ms:.0f}-{end_bin * bin_size_ms:.0f}ms (bins {start_bin}:{end_bin})")

    # Get recorded rates (time-averaged in window)
    target_rates = val_data['targets'].numpy()
    recorded_rates = target_rates[:, start_bin:end_bin, :].mean(axis=1)  # [n_trials, n_neurons]

    # Get RNN rates (time-averaged in window)
    rnn_rates = get_model_rates_window(model, val_data, n_recorded, start_bin, end_bin)

    results = {}

    # === REWARD ===
    reward_labels = val_data['trial_reward'].numpy()
    # Convert to binary if needed (may be 1/2 encoded)
    if reward_labels.min() >= 1:
        reward_labels = reward_labels - 1

    results['Reward'] = {
        'recorded_sel': compute_roc_selectivity(recorded_rates, reward_labels),
        'rnn_sel': compute_roc_selectivity(rnn_rates, reward_labels),
    }
    print(f"Reward: {(reward_labels==0).sum()} low, {(reward_labels==1).sum()} high")

    # === SALIENCE ===
    salience = val_data['trial_salience'].numpy()
    # Binarize: exclude middle level if present
    unique_sal = np.unique(salience)
    print(f"Salience unique values: {unique_sal}")

    if len(unique_sal) >= 2:
        # Use lowest vs highest
        low_val = unique_sal.min()
        high_val = unique_sal.max()
        sal_mask = (salience == low_val) | (salience == high_val)
        sal_labels = (salience[sal_mask] == high_val).astype(int)
        sal_recorded = recorded_rates[sal_mask]
        sal_rnn = rnn_rates[sal_mask]

        results['Salience'] = {
            'recorded_sel': compute_roc_selectivity(sal_recorded, sal_labels),
            'rnn_sel': compute_roc_selectivity(sal_rnn, sal_labels),
        }
        print(f"Salience: {(sal_labels==0).sum()} low, {(sal_labels==1).sum()} high")
    else:
        # No salience variation
        results['Salience'] = {
            'recorded_sel': np.zeros(n_recorded),
            'rnn_sel': np.zeros(n_recorded),
        }

    # === PROBABILITY ===
    # High probability (expected) vs Low probability (unexpected) target location
    probability = val_data['trial_probability'].numpy()
    # Convert to binary: 0 = low/unexpected, 1 = high/expected
    prob_labels = (probability == 1).astype(int)

    results['Probability'] = {
        'recorded_sel': compute_roc_selectivity(recorded_rates, prob_labels),
        'rnn_sel': compute_roc_selectivity(rnn_rates, prob_labels),
    }
    print(f"Probability: {(prob_labels==0).sum()} low (unexpected), {(prob_labels==1).sum()} high (expected)")

    # === IDENTITY ===
    identity = val_data['trial_identity'].numpy()
    unique_id = np.unique(identity)
    print(f"Identity unique values: {unique_id}")

    # Face (1) vs Non-face (2), exclude Bullseye (0) if present
    if 1 in unique_id and 2 in unique_id:
        id_mask = (identity == 1) | (identity == 2)
        id_labels = (identity[id_mask] == 1).astype(int)  # 1 for face
        id_recorded = recorded_rates[id_mask]
        id_rnn = rnn_rates[id_mask]

        results['Identity'] = {
            'recorded_sel': compute_roc_selectivity(id_recorded, id_labels),
            'rnn_sel': compute_roc_selectivity(id_rnn, id_labels),
        }
        print(f"Identity: {(id_labels==0).sum()} non-face, {(id_labels==1).sum()} face")
    elif len(unique_id) >= 2:
        # Use any two categories
        val1, val2 = unique_id[0], unique_id[1]
        id_mask = (identity == val1) | (identity == val2)
        id_labels = (identity[id_mask] == val2).astype(int)
        id_recorded = recorded_rates[id_mask]
        id_rnn = rnn_rates[id_mask]

        results['Identity'] = {
            'recorded_sel': compute_roc_selectivity(id_recorded, id_labels),
            'rnn_sel': compute_roc_selectivity(id_rnn, id_labels),
        }
    else:
        results['Identity'] = {
            'recorded_sel': np.zeros(n_recorded),
            'rnn_sel': np.zeros(n_recorded),
        }

    return results, n_exc


def create_figure(selectivity_results, n_exc, save_path):
    """Create the 1×4 panel figure."""
    factors = ['Reward', 'Salience', 'Probability', 'Identity']

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

    for i, factor in enumerate(factors):
        ax = axes[i]
        data = selectivity_results[factor]

        recorded_sel = data['recorded_sel']
        rnn_sel = data['rnn_sel']

        n_neurons = len(recorded_sel)

        # E neurons (indices 0 to n_exc-1)
        e_idx = np.arange(min(n_exc, n_neurons))
        # I neurons (indices n_exc to end)
        i_idx = np.arange(n_exc, n_neurons)

        # Plot E neurons
        ax.scatter(recorded_sel[e_idx], rnn_sel[e_idx],
                   c=E_COLOR, s=40, alpha=0.7, label=f'E (n={len(e_idx)})',
                   edgecolors='white', linewidths=0.5)

        # Plot I neurons
        ax.scatter(recorded_sel[i_idx], rnn_sel[i_idx],
                   c=I_COLOR, s=40, alpha=0.7, label=f'I (n={len(i_idx)})',
                   marker='o', facecolors='none', linewidths=1.5)

        # Unity line
        lim = max(np.abs(recorded_sel).max(), np.abs(rnn_sel).max(), 0.5) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.4, linewidth=1)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)

        # Set limits
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        # Compute correlation
        valid = ~(np.isnan(recorded_sel) | np.isnan(rnn_sel))
        if valid.sum() > 5:
            r, p = stats.pearsonr(recorded_sel[valid], rnn_sel[valid])
        else:
            r, p = 0, 1

        # Labels
        ax.set_xlabel('Recorded selectivity')
        ax.set_ylabel('RNN selectivity')
        ax.set_title(f'{factor}\nr = {r:.2f}, p = {p:.3f}')

        if i == 0:
            ax.legend(loc='upper left', fontsize=8)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Conditioned-Loss Model: Recorded vs RNN Selectivity (ROC)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figure saved to: {save_path}")

    # Print summary statistics
    print("\n=== Selectivity Comparison Summary ===")
    print(f"{'Factor':<12} {'r':>8} {'p-value':>10} {'Interpretation':<30}")
    print("-" * 65)

    for factor in factors:
        data = selectivity_results[factor]
        recorded_sel = data['recorded_sel']
        rnn_sel = data['rnn_sel']

        valid = ~(np.isnan(recorded_sel) | np.isnan(rnn_sel))
        if valid.sum() > 5:
            r, p = stats.pearsonr(recorded_sel[valid], rnn_sel[valid])
        else:
            r, p = 0, 1

        if p < 0.01:
            interp = "*** Highly significant"
        elif p < 0.05:
            interp = "* Significant"
        else:
            interp = "Not significant"

        print(f"{factor:<12} {r:>8.3f} {p:>10.4f} {interp:<30}")


def main():
    print("=" * 60)
    print("SELECTIVITY FIGURE GENERATION")
    print("Conditioned-Loss Model (Newton_08_15)")
    print("=" * 60)

    # Load data and model
    model, val_data, neuron_info = load_data_and_model()

    # Compute selectivity
    print("\nComputing selectivity indices...")
    selectivity_results, n_exc = compute_all_selectivity(model, val_data, neuron_info)

    # Create figure
    print("\nGenerating figure...")
    create_figure(selectivity_results, n_exc, OUTPUT_DIR / 'recorded_vs_rnn_selectivity.png')

    # Also save as PDF
    create_figure(selectivity_results, n_exc, OUTPUT_DIR / 'recorded_vs_rnn_selectivity.pdf')

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
