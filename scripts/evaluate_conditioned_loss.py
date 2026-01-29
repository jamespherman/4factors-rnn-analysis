#!/usr/bin/env python3
"""
Evaluation script for condition-specific loss model.

Compares the conditioned-loss model with the original grand-average model,
generating comparison figures and metrics.

Usage:
    python scripts/evaluate_conditioned_loss.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model_from_data
from src.data_loader import load_session, train_val_split

# Paths
BASE_DIR = Path("/Users/jph/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Code/4factors-rnn-analysis")
ORIGINAL_MODEL_DIR = BASE_DIR / "results/final_model"
CONDITIONED_MODEL_DIR = BASE_DIR / "results/conditioned_loss_08_15"
DATA_FILE = BASE_DIR / "data/rnn_export_Newton_08_15_2025_SC.mat"
OUTPUT_DIR = CONDITIONED_MODEL_DIR / "comparison_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_models_and_data():
    """Load both models and the validation data."""
    print("Loading data...")
    dataset = load_session(str(DATA_FILE))
    train_idx, val_idx = train_val_split(dataset, val_fraction=0.2, seed=42)

    all_data = dataset.get_all_trials(include_conditions=True)
    val_data = {
        'inputs': all_data['inputs'][val_idx],
        'targets': all_data['targets'][val_idx],
        'mask': all_data['mask'][val_idx],
        'trial_conditions': all_data['trial_conditions'][val_idx],
        'trial_reward': all_data['trial_reward'][val_idx],
        'trial_location': all_data['trial_location'][val_idx],
        'trial_salience': all_data['trial_salience'][val_idx],
    }

    neuron_info = dataset.get_neuron_info()
    n_inputs = dataset.get_input_dim()

    device = 'cpu'

    # Create model architecture
    def create_model():
        return create_model_from_data(
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

    print("Loading original model...")
    original_model = create_model()
    original_model.load_state_dict(torch.load(ORIGINAL_MODEL_DIR / 'model_best.pt', weights_only=True))
    original_model.eval()

    print("Loading conditioned-loss model...")
    conditioned_model = create_model()
    conditioned_model.load_state_dict(torch.load(CONDITIONED_MODEL_DIR / 'model_best.pt', weights_only=True))
    conditioned_model.eval()

    return original_model, conditioned_model, val_data, neuron_info, dataset.bin_size_ms


def compute_factor_selectivity(rates: np.ndarray, factor_values: np.ndarray) -> np.ndarray:
    """Compute d-prime selectivity for each neuron."""
    # Get mean rates per trial
    if rates.ndim == 3:
        mean_rates = rates.mean(axis=1)
    else:
        mean_rates = rates

    low_mask = factor_values == 0
    high_mask = factor_values == 1

    n_neurons = mean_rates.shape[1]
    selectivity = np.zeros(n_neurons)

    for i in range(n_neurons):
        low_rates = mean_rates[low_mask, i]
        high_rates = mean_rates[high_mask, i]

        if len(low_rates) < 3 or len(high_rates) < 3:
            continue

        mean_diff = high_rates.mean() - low_rates.mean()
        pooled_var = (low_rates.var() + high_rates.var()) / 2
        pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 1e-6
        selectivity[i] = mean_diff / pooled_std

    return selectivity


def get_model_rates(model, val_data, device='cpu'):
    """Get model firing rates."""
    with torch.no_grad():
        inputs = val_data['inputs'].to(device)
        model_rates, _ = model(inputs)
        n_recorded = val_data['targets'].shape[2]
        return model_rates[:, :, :n_recorded].cpu().numpy()


def compare_selectivity(original_model, conditioned_model, val_data):
    """Compare selectivity matching between models."""
    device = 'cpu'

    target_rates = val_data['targets'].numpy()
    original_rates = get_model_rates(original_model, val_data, device)
    conditioned_rates = get_model_rates(conditioned_model, val_data, device)

    factors = {
        'reward': val_data['trial_reward'].numpy(),
        'salience': val_data['trial_salience'].numpy(),
    }

    # Add binarized location (for each location vs others)
    location = val_data['trial_location'].numpy()

    results = {}

    for factor_name, factor_values in factors.items():
        if len(np.unique(factor_values)) < 2:
            continue

        target_sel = compute_factor_selectivity(target_rates, factor_values)
        original_sel = compute_factor_selectivity(original_rates, factor_values)
        conditioned_sel = compute_factor_selectivity(conditioned_rates, factor_values)

        r_original, p_original = stats.pearsonr(target_sel, original_sel)
        r_conditioned, p_conditioned = stats.pearsonr(target_sel, conditioned_sel)

        results[factor_name] = {
            'target_sel': target_sel,
            'original_sel': original_sel,
            'conditioned_sel': conditioned_sel,
            'r_original': r_original,
            'p_original': p_original,
            'r_conditioned': r_conditioned,
            'p_conditioned': p_conditioned,
        }

        print(f"\n{factor_name.capitalize()} selectivity:")
        print(f"  Original model: r = {r_original:.4f}, p = {p_original:.4f}")
        print(f"  Conditioned model: r = {r_conditioned:.4f}, p = {p_conditioned:.4f}")
        print(f"  Improvement: {r_conditioned - r_original:+.4f}")

    # Location selectivity (average across locations)
    location_sels_target = []
    location_sels_original = []
    location_sels_conditioned = []

    for loc in np.unique(location):
        loc_binary = (location == loc).astype(int)
        location_sels_target.append(compute_factor_selectivity(target_rates, loc_binary))
        location_sels_original.append(compute_factor_selectivity(original_rates, loc_binary))
        location_sels_conditioned.append(compute_factor_selectivity(conditioned_rates, loc_binary))

    target_loc_sel = np.abs(np.stack(location_sels_target)).mean(axis=0)
    original_loc_sel = np.abs(np.stack(location_sels_original)).mean(axis=0)
    conditioned_loc_sel = np.abs(np.stack(location_sels_conditioned)).mean(axis=0)

    r_original, p_original = stats.pearsonr(target_loc_sel, original_loc_sel)
    r_conditioned, p_conditioned = stats.pearsonr(target_loc_sel, conditioned_loc_sel)

    results['location'] = {
        'target_sel': target_loc_sel,
        'original_sel': original_loc_sel,
        'conditioned_sel': conditioned_loc_sel,
        'r_original': r_original,
        'p_original': p_original,
        'r_conditioned': r_conditioned,
        'p_conditioned': p_conditioned,
    }

    print(f"\nLocation selectivity:")
    print(f"  Original model: r = {r_original:.4f}, p = {p_original:.4f}")
    print(f"  Conditioned model: r = {r_conditioned:.4f}, p = {p_conditioned:.4f}")
    print(f"  Improvement: {r_conditioned - r_original:+.4f}")

    return results


def plot_selectivity_comparison(selectivity_results, save_path):
    """Plot selectivity comparison between models."""
    factors = list(selectivity_results.keys())
    n_factors = len(factors)

    fig, axes = plt.subplots(2, n_factors, figsize=(5 * n_factors, 10))

    for i, factor in enumerate(factors):
        data = selectivity_results[factor]

        # Top row: Original model
        ax = axes[0, i]
        ax.scatter(data['target_sel'], data['original_sel'], alpha=0.6, s=30)
        ax.plot([-3, 3], [-3, 3], 'k--', alpha=0.5)
        ax.set_xlabel(f'Target {factor} selectivity')
        ax.set_ylabel(f'Original model selectivity')
        ax.set_title(f'{factor.capitalize()}: Original\nr = {data["r_original"]:.3f}')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)

        # Bottom row: Conditioned model
        ax = axes[1, i]
        ax.scatter(data['target_sel'], data['conditioned_sel'], alpha=0.6, s=30, c='orange')
        ax.plot([-3, 3], [-3, 3], 'k--', alpha=0.5)
        ax.set_xlabel(f'Target {factor} selectivity')
        ax.set_ylabel(f'Conditioned model selectivity')
        ax.set_title(f'{factor.capitalize()}: Conditioned\nr = {data["r_conditioned"]:.3f}')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Selectivity Matching: Recorded vs RNN', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_selectivity_bar_comparison(selectivity_results, save_path):
    """Bar plot comparing selectivity correlations."""
    factors = list(selectivity_results.keys())

    original_corrs = [selectivity_results[f]['r_original'] for f in factors]
    conditioned_corrs = [selectivity_results[f]['r_conditioned'] for f in factors]

    x = np.arange(len(factors))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, original_corrs, width, label='Original (Grand Average)')
    bars2 = ax.bar(x + width/2, conditioned_corrs, width, label='Conditioned Loss')

    ax.set_xlabel('Factor')
    ax.set_ylabel('Selectivity Correlation (r)')
    ax.set_title('RNN vs Recorded Selectivity Matching')
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in factors])
    ax.legend()
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, original_corrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, conditioned_corrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def compare_ltrial(save_path):
    """Compare L_trial trajectories between models."""
    # Load training histories
    original_history = None
    conditioned_history = None

    orig_path = ORIGINAL_MODEL_DIR / 'training_log.json'
    cond_path = CONDITIONED_MODEL_DIR / 'training_log.json'

    if orig_path.exists():
        with open(orig_path) as f:
            original_history = json.load(f)

    if cond_path.exists():
        with open(cond_path) as f:
            conditioned_history = json.load(f)

    if original_history is None and conditioned_history is None:
        print("No training histories found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # L_trial comparison
    ax = axes[0]
    if original_history and 'L_trial' in original_history:
        epochs_orig = np.arange(1, len(original_history['L_trial']) + 1)
        ax.plot(epochs_orig, original_history['L_trial'], 'b-', linewidth=2,
                label='Original (Grand Average)')

    if conditioned_history and 'L_trial' in conditioned_history:
        epochs_cond = np.arange(1, len(conditioned_history['L_trial']) + 1)
        ax.plot(epochs_cond, conditioned_history['L_trial'], 'r-', linewidth=2,
                label='Conditioned Loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('L_trial')
    ax.set_title('Trial-Matching Loss Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation correlation comparison
    ax = axes[1]
    if original_history and 'val_correlation' in original_history:
        epochs_orig = np.arange(1, len(original_history['val_correlation']) + 1)
        ax.plot(epochs_orig, original_history['val_correlation'], 'b-', linewidth=2,
                label='Original')

    if conditioned_history and 'val_correlation' in conditioned_history:
        epochs_cond = np.arange(1, len(conditioned_history['val_correlation']) + 1)
        ax.plot(epochs_cond, conditioned_history['val_correlation'], 'r-', linewidth=2,
                label='Conditioned')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation PSTH Correlation')
    ax.set_title('PSTH Correlation Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    # Print final values
    print("\nFinal L_trial values:")
    if original_history and 'L_trial' in original_history:
        print(f"  Original: {original_history['L_trial'][-1]:.4f}")
    if conditioned_history and 'L_trial' in conditioned_history:
        print(f"  Conditioned: {conditioned_history['L_trial'][-1]:.4f}")

    print("\nFinal validation correlation:")
    if original_history and 'val_correlation' in original_history:
        print(f"  Original: {max(original_history['val_correlation']):.4f}")
    if conditioned_history and 'val_correlation' in conditioned_history:
        print(f"  Conditioned: {max(conditioned_history['val_correlation']):.4f}")


def plot_per_condition_psth(conditioned_model, val_data, neuron_idx, bin_size_ms, save_path):
    """Plot condition-specific PSTHs for example neuron."""
    device = 'cpu'

    with torch.no_grad():
        inputs = val_data['inputs'].to(device)
        targets = val_data['targets'].to(device)
        conditions = val_data['trial_conditions']

        model_rates, _ = conditioned_model(inputs)
        n_recorded = targets.shape[2]
        model_rates = model_rates[:, :, :n_recorded]

        time_axis = np.arange(targets.shape[1]) * bin_size_ms / 1000

        unique_conds = torch.unique(conditions).numpy()
        n_show = min(8, len(unique_conds))
        show_conds = unique_conds[:n_show]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i, cond in enumerate(show_conds):
            ax = axes[i]
            mask = conditions.numpy() == cond

            model_psth = model_rates[mask].mean(dim=0)[:, neuron_idx].cpu().numpy()
            target_psth = targets[mask].mean(dim=0)[:, neuron_idx].cpu().numpy()

            ax.plot(time_axis, target_psth, 'b-', linewidth=2, label='Target')
            ax.plot(time_axis, model_psth, 'r--', linewidth=2, label='Model')

            # Decode condition
            loc = cond // 4
            rew = (cond % 4) // 2
            sal = cond % 2
            ax.set_title(f'Loc{loc+1} R{"H" if rew else "L"} S{"H" if sal else "L"} (n={mask.sum()})')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Firing Rate')
            if i == 0:
                ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Neuron {neuron_idx}: Condition-Specific PSTHs (Conditioned Model)', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()


def generate_summary_report(selectivity_results, save_path):
    """Generate summary report."""
    report = """# Condition-Specific Loss Evaluation Report

## Summary

"""

    # Selectivity comparison table
    report += "### Selectivity Matching (r: Recorded vs RNN)\n\n"
    report += "| Factor | Original Model | Conditioned Model | Improvement |\n"
    report += "|--------|----------------|-------------------|-------------|\n"

    for factor, data in selectivity_results.items():
        improvement = data['r_conditioned'] - data['r_original']
        report += f"| {factor.capitalize()} | {data['r_original']:.4f} | {data['r_conditioned']:.4f} | {improvement:+.4f} |\n"

    # Mean improvement
    mean_improvement = np.mean([data['r_conditioned'] - data['r_original']
                                for data in selectivity_results.values()])
    report += f"\n**Mean improvement**: {mean_improvement:+.4f}\n"

    # Interpretation
    report += "\n### Interpretation\n\n"
    if mean_improvement > 0.1:
        report += "The conditioned-loss model shows substantial improvement in selectivity matching.\n"
    elif mean_improvement > 0:
        report += "The conditioned-loss model shows modest improvement in selectivity matching.\n"
    else:
        report += "The conditioned-loss model does not improve selectivity matching.\n"

    with open(save_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to {save_path}")


def main():
    print("=" * 60)
    print("CONDITION-SPECIFIC LOSS EVALUATION")
    print("=" * 60)

    # Load models and data
    original_model, conditioned_model, val_data, neuron_info, bin_size_ms = load_models_and_data()

    # Compare selectivity
    print("\n--- Selectivity Comparison ---")
    selectivity_results = compare_selectivity(original_model, conditioned_model, val_data)

    # Generate figures
    print("\n--- Generating Figures ---")

    print("  Selectivity comparison scatter...")
    plot_selectivity_comparison(selectivity_results, OUTPUT_DIR / 'selectivity_scatter_comparison.png')

    print("  Selectivity bar comparison...")
    plot_selectivity_bar_comparison(selectivity_results, OUTPUT_DIR / 'selectivity_bar_comparison.png')

    print("  L_trial comparison...")
    compare_ltrial(OUTPUT_DIR / 'ltrial_comparison.png')

    # Find best neuron for example PSTH
    conditioned_rates = get_model_rates(conditioned_model, val_data)
    target_rates = val_data['targets'].numpy()

    n_neurons = target_rates.shape[2]
    correlations = []
    for i in range(n_neurons):
        model_psth = conditioned_rates.mean(axis=0)[:, i]
        target_psth = target_rates.mean(axis=0)[:, i]
        r = np.corrcoef(model_psth, target_psth)[0, 1]
        correlations.append(r if not np.isnan(r) else 0)

    best_neuron = np.argmax(correlations)
    print(f"  Per-condition PSTH examples (neuron {best_neuron})...")
    plot_per_condition_psth(conditioned_model, val_data, best_neuron, bin_size_ms,
                           OUTPUT_DIR / 'per_condition_psth_examples.png')

    # Generate report
    print("\n--- Generating Report ---")
    generate_summary_report(selectivity_results, OUTPUT_DIR / 'evaluation_summary.md')

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
