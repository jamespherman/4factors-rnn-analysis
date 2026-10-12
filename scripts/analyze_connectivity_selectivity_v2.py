#!/usr/bin/env python3
"""
Connectivity-Selectivity Analysis Script (v2 - Fixed)

Analyzes the relationship between recurrent connectivity and factor selectivity
using ROC-AUC based selectivity matching the neuronal analysis pipeline.

Fixes from v1:
- Properly handles recorded vs hidden neuron dimensions
- Extracts recorded-neuron-only weight blocks for analysis
- Handles identity factor edge cases
- Includes graded ablation analysis

Usage:
    python scripts/analyze_connectivity_selectivity_v2.py

Output: results/biological_input_experiments/connectivity_analysis/
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EIRNN, create_model_from_data
from src.data_loader import load_mat_file, validate_data, find_target_onset_bin
from src.input_assignment import create_biological_assignment
from src.analysis.roc_selectivity import (
    compute_all_roc_selectivities, compute_emergent_roc_selectivity,
    find_visual_epoch_bins, ROCSelectivityResult, compute_roc_auc,
    permutation_test_roc
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    'data_path': 'data/rnn_export_Newton_08_15_2025_SC.mat',
    'results_dir': 'results/biological_input_experiments',
    'output_dir': 'results/biological_input_experiments/connectivity_analysis',

    # ROC parameters
    'visual_window_start_ms': 50.0,
    'visual_window_end_ms': 250.0,
    'min_trials': 5,
    'n_perm': 200,
    'n_bootstrap': 200,

    # Model parameters
    'input_embed_dim': 56,
    'input_embed_type': 'attention',
    'attention_heads': 4,
    'learnable_h0': True,
    'tau': 50.0,
    'dt': 25.0,
}


# ==============================================================================
# WEIGHT ANALYSIS FUNCTIONS (for recorded neurons only)
# ==============================================================================

def extract_recorded_weight_blocks(
    W_rec: np.ndarray,
    n_recorded_e: int,
    n_recorded_i: int,
    n_model_exc: int
) -> Dict[str, np.ndarray]:
    """
    Extract weight blocks involving only recorded neurons.

    In the model:
    - Recorded E: indices 0:n_recorded_e
    - Hidden E: indices n_recorded_e:n_model_exc
    - Recorded I: indices n_model_exc:n_model_exc+n_recorded_i

    Args:
        W_rec: Full model weight matrix [n_total, n_total]
        n_recorded_e: Number of recorded E neurons (14)
        n_recorded_i: Number of recorded I neurons (28)
        n_model_exc: Total E neurons in model (112)

    Returns:
        Dict with 'EE', 'IE', 'EI', 'II' weight blocks for recorded neurons
    """
    return {
        'EE': W_rec[:n_recorded_e, :n_recorded_e],                          # Recorded E -> Recorded E
        'IE': W_rec[:n_recorded_e, n_model_exc:n_model_exc+n_recorded_i],   # Recorded I -> Recorded E
        'EI': W_rec[n_model_exc:n_model_exc+n_recorded_i, :n_recorded_e],   # Recorded E -> Recorded I
        'II': W_rec[n_model_exc:n_model_exc+n_recorded_i, n_model_exc:n_model_exc+n_recorded_i]  # Recorded I -> Recorded I
    }


def analyze_weight_selectivity_correlation_recorded(
    weight_block: np.ndarray,
    source_selectivity: np.ndarray,
    target_selectivity: np.ndarray,
    connection_type: str
) -> Dict[str, Any]:
    """
    Analyze correlation between weights and selectivity similarity for recorded neurons.

    Args:
        weight_block: [n_target, n_source] weight matrix
        source_selectivity: [n_source] selectivity values
        target_selectivity: [n_target] selectivity values
        connection_type: 'EE', 'IE', 'EI', or 'II'

    Returns:
        Dict with correlation results
    """
    n_target, n_source = weight_block.shape

    weights = []
    sel_similarities = []

    for i in range(n_target):
        for j in range(n_source):
            # Skip if same neuron (for EE, II blocks)
            if connection_type in ['EE', 'II'] and i == j:
                continue

            w = np.abs(weight_block[i, j])
            if w < 1e-10 or np.isnan(w):
                continue

            sel_i = target_selectivity[i]
            sel_j = source_selectivity[j]

            if np.isnan(sel_i) or np.isnan(sel_j):
                continue

            # Selectivity similarity (negative distance)
            sel_sim = -np.abs(sel_i - sel_j)

            weights.append(w)
            sel_similarities.append(sel_sim)

    weights = np.array(weights)
    sel_similarities = np.array(sel_similarities)

    if len(weights) < 3:
        return {
            'connection_type': connection_type,
            'correlation': 0.0,
            'p_value': 1.0,
            'n_pairs': len(weights),
            'weights': weights,
            'similarities': sel_similarities
        }

    r, p = stats.pearsonr(weights, sel_similarities)

    return {
        'connection_type': connection_type,
        'correlation': float(r) if not np.isnan(r) else 0.0,
        'p_value': float(p) if not np.isnan(p) else 1.0,
        'n_pairs': len(weights),
        'weights': weights,
        'similarities': sel_similarities
    }


def compute_input_weighted_selectivity_recorded(
    weight_blocks: Dict[str, np.ndarray],
    selectivity_e: np.ndarray,
    selectivity_i: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute weighted input selectivity for recorded neurons.

    Returns:
        E_input_to_E: [n_e] weighted E input to E neurons
        I_input_to_E: [n_e] weighted I input to E neurons
        E_input_to_I: [n_i] weighted E input to I neurons
        I_input_to_I: [n_i] weighted I input to I neurons
    """
    # Input to E neurons
    E_input_to_E = np.abs(weight_blocks['EE']) @ np.nan_to_num(selectivity_e, nan=0.5)
    I_input_to_E = np.abs(weight_blocks['IE']) @ np.nan_to_num(selectivity_i, nan=0.5)

    # Input to I neurons
    E_input_to_I = np.abs(weight_blocks['EI']) @ np.nan_to_num(selectivity_e, nan=0.5)
    I_input_to_I = np.abs(weight_blocks['II']) @ np.nan_to_num(selectivity_i, nan=0.5)

    return E_input_to_E, I_input_to_E, E_input_to_I, I_input_to_I


def fit_effective_connectivity_model_recorded(
    weight_blocks: Dict[str, np.ndarray],
    selectivity_e: np.ndarray,
    selectivity_i: np.ndarray,
    receives_input_e: np.ndarray,
    receives_input_i: np.ndarray
) -> Dict[str, Any]:
    """
    Fit effective connectivity model for recorded neurons.

    Model for E neurons:
    selectivity_e[i] = β0 + β_direct * receives_e[i] + β_EE * E_input[i] + β_IE * I_input[i]

    Returns:
        Dict with coefficients and R^2 for E and I neurons separately
    """
    E_input_E, I_input_E, E_input_I, I_input_I = compute_input_weighted_selectivity_recorded(
        weight_blocks, selectivity_e, selectivity_i
    )

    results = {}

    # Fit for E neurons
    valid_e = ~np.isnan(selectivity_e)
    if valid_e.sum() >= 5:
        X_e = np.column_stack([
            np.ones(len(selectivity_e)),
            receives_input_e.astype(float),
            E_input_E,
            I_input_E
        ])[valid_e]
        y_e = selectivity_e[valid_e]

        try:
            beta_e, _, _, _ = np.linalg.lstsq(X_e, y_e, rcond=None)
            pred_e = X_e @ beta_e
            ss_res = np.sum((y_e - pred_e) ** 2)
            ss_tot = np.sum((y_e - y_e.mean()) ** 2)
            r2_e = 1 - ss_res / (ss_tot + 1e-10)

            results['E_neurons'] = {
                'beta_intercept': float(beta_e[0]),
                'beta_direct': float(beta_e[1]),
                'beta_EE': float(beta_e[2]),
                'beta_IE': float(beta_e[3]),
                'r_squared': float(max(0, r2_e)),
                'n_neurons': int(valid_e.sum())
            }
        except:
            results['E_neurons'] = None
    else:
        results['E_neurons'] = None

    # Fit for I neurons
    valid_i = ~np.isnan(selectivity_i)
    if valid_i.sum() >= 5:
        X_i = np.column_stack([
            np.ones(len(selectivity_i)),
            receives_input_i.astype(float),
            E_input_I,
            I_input_I
        ])[valid_i]
        y_i = selectivity_i[valid_i]

        try:
            beta_i, _, _, _ = np.linalg.lstsq(X_i, y_i, rcond=None)
            pred_i = X_i @ beta_i
            ss_res = np.sum((y_i - pred_i) ** 2)
            ss_tot = np.sum((y_i - y_i.mean()) ** 2)
            r2_i = 1 - ss_res / (ss_tot + 1e-10)

            results['I_neurons'] = {
                'beta_intercept': float(beta_i[0]),
                'beta_direct': float(beta_i[1]),
                'beta_EI': float(beta_i[2]),
                'beta_II': float(beta_i[3]),
                'r_squared': float(max(0, r2_i)),
                'n_neurons': int(valid_i.sum())
            }
        except:
            results['I_neurons'] = None
    else:
        results['I_neurons'] = None

    return results


def graded_ablation_analysis(
    model: EIRNN,
    val_inputs: torch.Tensor,
    val_factors: Dict[str, np.ndarray],
    epoch_bins: Tuple[int, int],
    n_recorded: int,
    device: str = 'cpu',
    scale_factors: np.ndarray = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Analyze how selectivity changes with graded ablation of connection types.

    Returns:
        Dict mapping ablation_type -> factor -> [n_scales] mean AUC deviation
    """
    if scale_factors is None:
        scale_factors = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    n_exc = model.n_exc
    n_total = model.n_total

    # Save original weights
    W_rec_original = model.W_rec.detach().cpu().numpy().copy()
    W_raw_original = model.W_rec_raw.data.clone()

    results = {}

    ablation_configs = {
        'EE': (slice(0, n_exc), slice(0, n_exc)),
        'IE': (slice(0, n_exc), slice(n_exc, n_total)),
        'EI': (slice(n_exc, n_total), slice(0, n_exc)),
        'II': (slice(n_exc, n_total), slice(n_exc, n_total))
    }

    for abl_type, (row_slice, col_slice) in ablation_configs.items():
        print(f"    Running graded ablation for {abl_type}...")

        factor_results = {f: [] for f in ['reward', 'probability', 'salience']}

        for scale in scale_factors:
            # Create scaled weight matrix
            W_scaled = W_rec_original.copy()
            W_scaled[row_slice, col_slice] *= scale

            # Set model weights (W_rec = |W_raw| * sign_mask * diag_mask)
            with torch.no_grad():
                model.W_rec_raw.data = torch.tensor(np.abs(W_scaled), dtype=torch.float32, device=device)

            # Run model
            model.eval()
            with torch.no_grad():
                rates, _ = model(val_inputs.to(device))
                rates_np = rates.cpu().numpy()[:, :, :n_recorded]

            # Compute selectivity for each factor
            for factor_name, factor_values in val_factors.items():
                if factor_name not in factor_results:
                    continue

                roc_results = compute_all_roc_selectivities(
                    rates_np, {factor_name: factor_values}, epoch_bins,
                    min_trials=5, n_perm=50, n_bootstrap=50
                )

                if factor_name in roc_results:
                    auc = roc_results[factor_name].auc
                    mean_dev = np.nanmean(np.abs(auc - 0.5))
                    factor_results[factor_name].append(mean_dev)
                else:
                    factor_results[factor_name].append(np.nan)

        # Restore original weights
        with torch.no_grad():
            model.W_rec_raw.data = W_raw_original.clone()

        results[abl_type] = {k: np.array(v) for k, v in factor_results.items()}

    return results, scale_factors


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_roc_selectivity_summary(
    roc_results: Dict[str, ROCSelectivityResult],
    n_exc: int,
    title: str,
    output_path: str
):
    """Plot ROC selectivity distribution by neuron type."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    factors = ['reward', 'probability', 'salience', 'identity']
    colors = {'E': '#e74c3c', 'I': '#3498db'}

    for idx, factor in enumerate(factors):
        ax = axes[idx // 2, idx % 2]

        if factor not in roc_results:
            ax.set_title(f'{factor.capitalize()} - No data')
            continue

        auc = roc_results[factor].auc
        if len(auc) == 0 or np.all(np.isnan(auc)):
            ax.set_title(f'{factor.capitalize()} - No valid data')
            continue

        auc_e = auc[:n_exc]
        auc_i = auc[n_exc:]

        # Remove NaNs
        auc_e_valid = auc_e[~np.isnan(auc_e)]
        auc_i_valid = auc_i[~np.isnan(auc_i)]

        data = []
        labels = []
        box_colors = []

        if len(auc_e_valid) > 0:
            data.append(auc_e_valid)
            labels.append(f'E (n={len(auc_e_valid)})')
            box_colors.append(colors['E'])

        if len(auc_i_valid) > 0:
            data.append(auc_i_valid)
            labels.append(f'I (n={len(auc_i_valid)})')
            box_colors.append(colors['I'])

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_ylabel('ROC-AUC')
        ax.set_ylim(0, 1)

        # Add significance counts
        n_sig = roc_results[factor].n_significant
        ax.set_title(f'{factor.capitalize()}\n({n_sig}/{len(auc)} significant)')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_weight_selectivity_scatter_recorded(
    weight_blocks: Dict[str, np.ndarray],
    selectivity_e: np.ndarray,
    selectivity_i: np.ndarray,
    weight_sel_results: Dict[str, Dict[str, Any]],
    factor_name: str,
    output_path: str
):
    """Plot weight vs selectivity scatter for recorded neurons."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    configs = [
        ('EE', selectivity_e, selectivity_e, 'E→E'),
        ('IE', selectivity_i, selectivity_e, 'I→E'),
        ('EI', selectivity_e, selectivity_i, 'E→I'),
        ('II', selectivity_i, selectivity_i, 'I→I')
    ]

    for idx, (block_name, source_sel, target_sel, title) in enumerate(configs):
        ax = axes[idx // 2, idx % 2]

        result = weight_sel_results.get(block_name, {})
        weights = result.get('weights', np.array([]))
        similarities = result.get('similarities', np.array([]))
        r = result.get('correlation', 0)
        p = result.get('p_value', 1)
        n = result.get('n_pairs', 0)

        if len(weights) > 0:
            ax.scatter(weights, similarities, alpha=0.4, s=20)

            # Regression line if significant
            if p < 0.05 and len(weights) > 2:
                z = np.polyfit(weights, similarities, 1)
                poly = np.poly1d(z)
                x_line = np.linspace(weights.min(), weights.max(), 100)
                ax.plot(x_line, poly(x_line), 'r-', linewidth=2, label=f'r={r:.3f}*')
                ax.legend(loc='lower right')

        sig_str = '*' if p < 0.05 else ''
        ax.set_title(f'{title}: r={r:.3f}{sig_str} (n={n})')
        ax.set_xlabel('|Weight|')
        ax.set_ylabel('Selectivity Similarity')

    plt.suptitle(f'{factor_name.capitalize()} - Weight vs Selectivity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_effective_connectivity_summary(
    ec_results: Dict[str, Dict[str, Any]],
    output_path: str
):
    """Plot effective connectivity coefficients."""
    factors = [f for f in ec_results.keys() if ec_results[f] is not None]

    if not factors:
        print("    No valid effective connectivity results to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # E neurons
    ax = axes[0]
    e_data = []
    e_labels = []
    for f in factors:
        if ec_results[f] and ec_results[f].get('E_neurons'):
            e_data.append(ec_results[f]['E_neurons'])
            e_labels.append(f.capitalize())

    if e_data:
        x = np.arange(len(e_labels))
        width = 0.2

        ax.bar(x - width, [d['beta_direct'] for d in e_data], width, label='β_direct', color='#3498db')
        ax.bar(x, [d['beta_EE'] for d in e_data], width, label='β_EE', color='#e74c3c')
        ax.bar(x + width, [d['beta_IE'] for d in e_data], width, label='β_IE', color='#2ecc71')

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(e_labels)
        ax.legend()
        ax.set_ylabel('Coefficient')
        ax.set_title('E Neurons: Effective Connectivity')

        # Add R² annotations
        for i, d in enumerate(e_data):
            ax.annotate(f"R²={d['r_squared']:.2f}", (i, ax.get_ylim()[1]*0.9), ha='center', fontsize=9)

    # I neurons
    ax = axes[1]
    i_data = []
    i_labels = []
    for f in factors:
        if ec_results[f] and ec_results[f].get('I_neurons'):
            i_data.append(ec_results[f]['I_neurons'])
            i_labels.append(f.capitalize())

    if i_data:
        x = np.arange(len(i_labels))
        width = 0.2

        ax.bar(x - width, [d['beta_direct'] for d in i_data], width, label='β_direct', color='#3498db')
        ax.bar(x, [d['beta_EI'] for d in i_data], width, label='β_EI', color='#e74c3c')
        ax.bar(x + width, [d['beta_II'] for d in i_data], width, label='β_II', color='#2ecc71')

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(i_labels)
        ax.legend()
        ax.set_ylabel('Coefficient')
        ax.set_title('I Neurons: Effective Connectivity')

        for i, d in enumerate(i_data):
            ax.annotate(f"R²={d['r_squared']:.2f}", (i, ax.get_ylim()[1]*0.9), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_graded_ablation_curves(
    ablation_results: Dict[str, Dict[str, np.ndarray]],
    scale_factors: np.ndarray,
    output_path: str
):
    """Plot selectivity vs ablation level for each connection type."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ablation_types = ['EE', 'IE', 'EI', 'II']
    titles = ['E→E Ablation', 'I→E Ablation', 'E→I Ablation', 'I→I Ablation']
    colors = {'reward': '#e74c3c', 'probability': '#3498db', 'salience': '#2ecc71'}

    for idx, (abl_type, title) in enumerate(zip(ablation_types, titles)):
        ax = axes[idx // 2, idx % 2]

        if abl_type not in ablation_results:
            continue

        for factor, values in ablation_results[abl_type].items():
            if len(values) == len(scale_factors):
                ax.plot(scale_factors, values, 'o-', label=factor.capitalize(),
                       color=colors.get(factor, 'gray'), linewidth=2, markersize=6)

        ax.set_xlabel('Connection Scale Factor')
        ax.set_ylabel('Mean |AUC - 0.5|')
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_emergent_selectivity_comparison(
    roc_results: Dict[str, ROCSelectivityResult],
    output_path: str
):
    """Plot emergent selectivity analysis."""
    factors = [f for f in ['reward', 'probability', 'salience'] if f in roc_results]

    if not factors:
        return

    fig, axes = plt.subplots(1, len(factors), figsize=(5*len(factors), 5))
    if len(factors) == 1:
        axes = [axes]

    colors = {'receivers': '#3498db', 'non_receivers': '#e74c3c'}

    for idx, factor in enumerate(factors):
        ax = axes[idx]
        result = roc_results[factor]

        if result.receives_input is None:
            ax.set_title(f'{factor.capitalize()} - No assignment')
            continue

        emergent = compute_emergent_roc_selectivity(result)

        receiver_mask = result.receives_input
        non_receiver_mask = ~receiver_mask

        data = []
        labels = []
        box_colors = []

        # Receivers
        rec_auc = result.auc[receiver_mask]
        rec_auc = rec_auc[~np.isnan(rec_auc)]
        if len(rec_auc) > 0:
            data.append(rec_auc)
            labels.append(f'Receivers\n(n={len(rec_auc)})')
            box_colors.append(colors['receivers'])

        # Non-receivers (emergent)
        nonrec_auc = result.auc[non_receiver_mask]
        nonrec_auc = nonrec_auc[~np.isnan(nonrec_auc)]
        if len(nonrec_auc) > 0:
            data.append(nonrec_auc)
            labels.append(f'Non-receivers\n(n={len(nonrec_auc)})')
            box_colors.append(colors['non_receivers'])

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('ROC-AUC')
        ax.set_ylim(0, 1)
        ax.set_title(f'{factor.capitalize()}\n{emergent["frac_emergent_significant"]*100:.0f}% emergent significant')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def run_analysis():
    """Run full connectivity-selectivity analysis."""
    print("=" * 70)
    print("CONNECTIVITY-SELECTIVITY ANALYSIS (v2 - Fixed)")
    print("=" * 70)

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    results_dir = Path(CONFIG['results_dir'])
    e0_dir = results_dir / 'experiments' / 'E0_baseline'
    e1_dir = results_dir / 'experiments' / 'E1_biological'

    # Load data
    print("\n[1/6] Loading data...")
    data = load_mat_file(CONFIG['data_path'])
    validate_data(data)

    neuron_type = data['neuron_type']
    n_neurons = int(data['n_neurons'])
    n_exc = int((neuron_type == 1).sum())  # Recorded E
    n_inh = int((neuron_type == 2).sum())  # Recorded I
    bin_size_ms = float(data['bin_size_ms'])

    print(f"  Recorded neurons: {n_neurons} ({n_exc} E, {n_inh} I)")

    # Prepare inputs
    firing_rates = data['firing_rates']
    n_bins = firing_rates.shape[1]
    n_trials = firing_rates.shape[2]

    # Build standard 14-channel input
    input_channels = []
    for key in ['input_fixation_on', 'input_go_signal', 'input_reward_on',
                'input_eye_x', 'input_eye_y', 'input_is_face', 'input_is_nonface',
                'input_is_bullseye', 'input_high_salience', 'input_low_salience']:
        if key in data:
            sig = data[key]
            if sig.ndim == 1:
                sig = np.tile(sig[None, :], (n_bins, 1))
            input_channels.append(sig.T)

    for i in range(4):
        input_channels.append(data['input_target_loc'][i].T)

    inputs = np.stack(input_channels, axis=2).astype(np.float32)
    inputs = torch.tensor(inputs)

    # Trial factors
    trial_factors = {
        'reward': data['trial_reward'].astype(float),
        'probability': data['trial_probability'].astype(float),
        'salience': data['trial_salience'].astype(float),
    }

    # Check identity factor
    identity = data['trial_identity'].astype(float)
    if len(np.unique(identity)) > 1:
        trial_factors['identity'] = identity
    else:
        print("  Note: Identity factor has only one unique value, excluding from analysis")

    # Target onset and epoch
    target_onset_bin = find_target_onset_bin(data['input_target_loc'])
    epoch_bins = find_visual_epoch_bins(target_onset_bin, bin_size_ms,
                                        CONFIG['visual_window_start_ms'],
                                        CONFIG['visual_window_end_ms'])
    print(f"  Visual epoch: bins {epoch_bins[0]}-{epoch_bins[1]} ({CONFIG['visual_window_start_ms']}-{CONFIG['visual_window_end_ms']}ms post-target)")

    # Train/val split
    np.random.seed(42)
    indices = np.arange(n_trials)
    np.random.shuffle(indices)
    n_val = int(n_trials * 0.2)
    val_idx = indices[:n_val]

    val_inputs = inputs[val_idx]
    val_factors = {k: v[val_idx] for k, v in trial_factors.items()}

    device = 'cpu'

    # =========================================================================
    # E0 BASELINE
    # =========================================================================
    print("\n[2/6] Analyzing E0 Baseline...")

    model_e0 = create_model_from_data(
        n_classic=n_exc, n_interneuron=n_inh, n_inputs=14,
        enforce_ratio=True,
        input_embed_dim=CONFIG['input_embed_dim'],
        input_embed_type=CONFIG['input_embed_type'],
        attention_heads=CONFIG['attention_heads'],
        learnable_h0=CONFIG['learnable_h0'],
        device=device
    )
    model_e0.load_state_dict(torch.load(str(e0_dir / 'model_best.pt'), map_location=device))
    model_e0.eval()

    n_model_exc = model_e0.n_exc
    n_model_total = model_e0.n_total

    print(f"  Model: {n_model_total} neurons ({n_model_exc} E, {n_model_total - n_model_exc} I)")

    # Get outputs (recorded neurons only)
    with torch.no_grad():
        rates_e0, _ = model_e0(val_inputs.to(device))
        rates_e0 = rates_e0.cpu().numpy()[:, :, :n_neurons]

    # ROC selectivity
    print("  Computing ROC selectivity...")
    roc_e0 = compute_all_roc_selectivities(
        rates_e0, val_factors, epoch_bins,
        min_trials=CONFIG['min_trials'],
        n_perm=CONFIG['n_perm'],
        n_bootstrap=CONFIG['n_bootstrap']
    )

    for f, r in roc_e0.items():
        print(f"    {f}: AUC={r.mean_auc:.3f}, {r.n_significant}/{r.n_neurons} significant")

    # Weight analysis
    W_rec_e0 = model_e0.W_rec.detach().cpu().numpy()
    weight_blocks_e0 = extract_recorded_weight_blocks(W_rec_e0, n_exc, n_inh, n_model_exc)

    print("  Analyzing weight-selectivity correlations...")
    weight_sel_e0 = {}
    for factor, result in roc_e0.items():
        sel_e = result.auc[:n_exc]
        sel_i = result.auc[n_exc:]

        weight_sel_e0[factor] = {}
        for block_name in ['EE', 'IE', 'EI', 'II']:
            if block_name == 'EE':
                corr = analyze_weight_selectivity_correlation_recorded(
                    weight_blocks_e0[block_name], sel_e, sel_e, block_name)
            elif block_name == 'IE':
                corr = analyze_weight_selectivity_correlation_recorded(
                    weight_blocks_e0[block_name], sel_i, sel_e, block_name)
            elif block_name == 'EI':
                corr = analyze_weight_selectivity_correlation_recorded(
                    weight_blocks_e0[block_name], sel_e, sel_i, block_name)
            else:  # II
                corr = analyze_weight_selectivity_correlation_recorded(
                    weight_blocks_e0[block_name], sel_i, sel_i, block_name)

            weight_sel_e0[factor][block_name] = corr
            sig = '*' if corr['p_value'] < 0.05 else ''
            print(f"    {factor}/{block_name}: r={corr['correlation']:.3f}{sig} (n={corr['n_pairs']})")

    # =========================================================================
    # E1 BIOLOGICAL
    # =========================================================================
    print("\n[3/6] Analyzing E1 Biological...")

    # Create assignment
    rf_locations = np.zeros(n_neurons, dtype=int)
    for i in range(n_exc):
        rf_locations[i] = (i % 4) + 1

    assignment_e1 = create_biological_assignment(
        n_neurons, neuron_type, rf_locations,
        e_fractions={'reward': 0.30, 'probability': 0.30, 'salience': 0.75, 'identity': 0.75},
        i_fractions={'reward': 0.40, 'probability': 0.40, 'salience': 0.25, 'identity': 0.25},
        seed=42
    )

    # Load E1 model
    n_bio_inputs = n_neurons * 6
    model_e1 = create_model_from_data(
        n_classic=n_exc, n_interneuron=n_inh, n_inputs=n_bio_inputs,
        enforce_ratio=True,
        input_embed_dim=CONFIG['input_embed_dim'],
        input_embed_type='linear',
        learnable_h0=CONFIG['learnable_h0'],
        device=device
    )
    model_e1.load_state_dict(torch.load(str(e1_dir / 'model_best.pt'), map_location=device))
    model_e1.eval()

    # Construct biological inputs
    from src.data_loader import construct_restructured_input_tensor
    bio_inputs = construct_restructured_input_tensor(data, assignment_e1)
    bio_inputs_flat = bio_inputs.reshape(bio_inputs.shape[0], bio_inputs.shape[1], -1)
    bio_inputs_tensor = torch.tensor(bio_inputs_flat, dtype=torch.float32)
    val_bio_inputs = bio_inputs_tensor[val_idx]

    # Get outputs
    with torch.no_grad():
        rates_e1, _ = model_e1(val_bio_inputs.to(device))
        rates_e1 = rates_e1.cpu().numpy()[:, :, :n_neurons]

    # ROC selectivity with assignment
    print("  Computing ROC selectivity...")
    roc_e1 = compute_all_roc_selectivities(
        rates_e1, val_factors, epoch_bins,
        assignment=assignment_e1,
        min_trials=CONFIG['min_trials'],
        n_perm=CONFIG['n_perm'],
        n_bootstrap=CONFIG['n_bootstrap']
    )

    for f, r in roc_e1.items():
        emergent = compute_emergent_roc_selectivity(r)
        print(f"    {f}: AUC={r.mean_auc:.3f}, emergent: {emergent['frac_emergent_significant']*100:.0f}% sig")

    # Weight analysis
    W_rec_e1 = model_e1.W_rec.detach().cpu().numpy()
    weight_blocks_e1 = extract_recorded_weight_blocks(W_rec_e1, n_exc, n_inh, model_e1.n_exc)

    print("  Analyzing weight-selectivity correlations...")
    weight_sel_e1 = {}
    for factor, result in roc_e1.items():
        sel_e = result.auc[:n_exc]
        sel_i = result.auc[n_exc:]

        weight_sel_e1[factor] = {}
        for block_name in ['EE', 'IE', 'EI', 'II']:
            if block_name == 'EE':
                corr = analyze_weight_selectivity_correlation_recorded(
                    weight_blocks_e1[block_name], sel_e, sel_e, block_name)
            elif block_name == 'IE':
                corr = analyze_weight_selectivity_correlation_recorded(
                    weight_blocks_e1[block_name], sel_i, sel_e, block_name)
            elif block_name == 'EI':
                corr = analyze_weight_selectivity_correlation_recorded(
                    weight_blocks_e1[block_name], sel_e, sel_i, block_name)
            else:
                corr = analyze_weight_selectivity_correlation_recorded(
                    weight_blocks_e1[block_name], sel_i, sel_i, block_name)

            weight_sel_e1[factor][block_name] = corr
            sig = '*' if corr['p_value'] < 0.05 else ''
            print(f"    {factor}/{block_name}: r={corr['correlation']:.3f}{sig} (n={corr['n_pairs']})")

    # Effective connectivity
    print("  Fitting effective connectivity models...")
    ec_e1 = {}
    for factor, result in roc_e1.items():
        receives = result.receives_input if result.receives_input is not None else np.ones(n_neurons, dtype=bool)
        receives_e = receives[:n_exc]
        receives_i = receives[n_exc:]

        ec_e1[factor] = fit_effective_connectivity_model_recorded(
            weight_blocks_e1,
            result.auc[:n_exc],
            result.auc[n_exc:],
            receives_e,
            receives_i
        )

        if ec_e1[factor].get('E_neurons'):
            e_res = ec_e1[factor]['E_neurons']
            print(f"    {factor} (E): R²={e_res['r_squared']:.3f}, β_EE={e_res['beta_EE']:.4f}, β_IE={e_res['beta_IE']:.4f}")

    # =========================================================================
    # GRADED ABLATION
    # =========================================================================
    print("\n[4/6] Running graded ablation analysis...")

    ablation_e1, scale_factors = graded_ablation_analysis(
        model_e1, val_bio_inputs, val_factors, epoch_bins, n_neurons, device
    )

    # =========================================================================
    # GENERATE FIGURES
    # =========================================================================
    print("\n[5/6] Generating figures...")

    # ROC selectivity summaries
    plot_roc_selectivity_summary(roc_e0, n_exc, 'E0 Baseline - ROC Selectivity',
                                  str(figures_dir / 'roc_selectivity_e0.png'))
    plot_roc_selectivity_summary(roc_e1, n_exc, 'E1 Biological - ROC Selectivity',
                                  str(figures_dir / 'roc_selectivity_e1.png'))

    # Weight-selectivity scatter
    for factor in roc_e0.keys():
        plot_weight_selectivity_scatter_recorded(
            weight_blocks_e0,
            roc_e0[factor].auc[:n_exc], roc_e0[factor].auc[n_exc:],
            weight_sel_e0[factor], factor,
            str(figures_dir / f'weight_sel_scatter_e0_{factor}.png')
        )

    for factor in roc_e1.keys():
        plot_weight_selectivity_scatter_recorded(
            weight_blocks_e1,
            roc_e1[factor].auc[:n_exc], roc_e1[factor].auc[n_exc:],
            weight_sel_e1[factor], factor,
            str(figures_dir / f'weight_sel_scatter_e1_{factor}.png')
        )

    # Effective connectivity
    plot_effective_connectivity_summary(ec_e1, str(figures_dir / 'effective_connectivity_e1.png'))

    # Graded ablation
    plot_graded_ablation_curves(ablation_e1, scale_factors, str(figures_dir / 'graded_ablation_e1.png'))

    # Emergent selectivity
    plot_emergent_selectivity_comparison(roc_e1, str(figures_dir / 'emergent_selectivity_e1.png'))

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n[6/6] Saving results...")

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'data_info': {
            'n_recorded_e': n_exc,
            'n_recorded_i': n_inh,
            'n_model_e': n_model_exc,
            'n_model_total': n_model_total,
            'visual_epoch_bins': list(epoch_bins)
        },
        'e0_baseline': {
            'roc_selectivity': {f: r.to_dict() for f, r in roc_e0.items()},
            'weight_selectivity': {
                f: {b: {'correlation': c['correlation'], 'p_value': c['p_value'], 'n_pairs': c['n_pairs']}
                    for b, c in blocks.items()}
                for f, blocks in weight_sel_e0.items()
            }
        },
        'e1_biological': {
            'roc_selectivity': {f: r.to_dict() for f, r in roc_e1.items()},
            'emergent_selectivity': {f: compute_emergent_roc_selectivity(r) for f, r in roc_e1.items()},
            'weight_selectivity': {
                f: {b: {'correlation': c['correlation'], 'p_value': c['p_value'], 'n_pairs': c['n_pairs']}
                    for b, c in blocks.items()}
                for f, blocks in weight_sel_e1.items()
            },
            'effective_connectivity': ec_e1,
            'graded_ablation': {
                'scale_factors': scale_factors.tolist(),
                'results': {abl: {f: v.tolist() for f, v in factors.items()}
                           for abl, factors in ablation_e1.items()}
            }
        }
    }

    with open(output_dir / 'results_v2.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Generate report
    generate_report_v2(results, output_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - results_v2.json")
    print(f"  - CONNECTIVITY_ANALYSIS_REPORT_v2.md")
    print(f"  - figures/")


def generate_report_v2(results: Dict, output_dir: Path):
    """Generate comprehensive markdown report."""

    report = f"""# Connectivity-Selectivity Analysis Report (v2)

**Date:** {results['timestamp'][:10]}

---

## Overview

This analysis examines the relationship between recurrent connectivity (E→E, I→E, etc.)
and factor selectivity using ROC-AUC, matching the neuronal analysis pipeline.

### Data Configuration
- **Recorded E neurons:** {results['data_info']['n_recorded_e']}
- **Recorded I neurons:** {results['data_info']['n_recorded_i']}
- **Model E neurons:** {results['data_info']['n_model_e']} (includes hidden)
- **Visual epoch:** bins {results['data_info']['visual_epoch_bins'][0]}-{results['data_info']['visual_epoch_bins'][1]}

---

## Methods

### ROC-AUC Selectivity
- **Metric:** Area Under ROC Curve via Mann-Whitney U statistic
- **Window:** {results['config']['visual_window_start_ms']}-{results['config']['visual_window_end_ms']} ms post-target onset
- **Significance:** Permutation test (n={results['config']['n_perm']}) + Bootstrap CI

---

## E0 Baseline Results

### ROC Selectivity
"""

    for f, data in results['e0_baseline']['roc_selectivity'].items():
        report += f"- **{f.capitalize()}:** Mean AUC = {data['mean_auc']:.3f}, {data['n_significant']}/{len(data['auc'])} significant\n"

    report += "\n### Weight-Selectivity Correlations\n\n"
    report += "| Factor | E→E | I→E | E→I | I→I |\n"
    report += "|--------|-----|-----|-----|-----|\n"

    for f in results['e0_baseline']['weight_selectivity'].keys():
        row = f"| {f.capitalize()} "
        for conn in ['EE', 'IE', 'EI', 'II']:
            data = results['e0_baseline']['weight_selectivity'][f][conn]
            sig = '*' if data['p_value'] < 0.05 else ''
            row += f"| {data['correlation']:.3f}{sig} "
        report += row + "|\n"

    report += """
---

## E1 Biological Results

### ROC Selectivity & Emergent Selectivity
"""

    for f, data in results['e1_biological']['roc_selectivity'].items():
        emergent = results['e1_biological']['emergent_selectivity'][f]
        report += f"- **{f.capitalize()}:** Mean AUC = {data['mean_auc']:.3f}, "
        report += f"**{emergent['frac_emergent_significant']*100:.0f}% emergent significant**\n"

    report += "\n### Weight-Selectivity Correlations\n\n"
    report += "| Factor | E→E | I→E | E→I | I→I |\n"
    report += "|--------|-----|-----|-----|-----|\n"

    for f in results['e1_biological']['weight_selectivity'].keys():
        row = f"| {f.capitalize()} "
        for conn in ['EE', 'IE', 'EI', 'II']:
            data = results['e1_biological']['weight_selectivity'][f][conn]
            sig = '*' if data['p_value'] < 0.05 else ''
            row += f"| {data['correlation']:.3f}{sig} "
        report += row + "|\n"

    report += "\n### Effective Connectivity Model\n\n"
    report += "**E Neurons:** `selectivity = β₀ + β_direct×receives + β_EE×E_input + β_IE×I_input`\n\n"
    report += "| Factor | β_direct | β_EE | β_IE | R² |\n"
    report += "|--------|----------|------|------|----|\n"

    for f, data in results['e1_biological']['effective_connectivity'].items():
        if data and data.get('E_neurons'):
            e = data['E_neurons']
            report += f"| {f.capitalize()} | {e['beta_direct']:.4f} | {e['beta_EE']:.4f} | {e['beta_IE']:.4f} | {e['r_squared']:.3f} |\n"

    report += """
---

## Key Findings

### 1. ROC-Based Selectivity Matches Neuronal Pipeline
Selectivity computed using ROC-AUC (Mann-Whitney U) in the 50-250ms visual epoch,
directly comparable to neuronal selectivity measures.

### 2. Strong Emergent Selectivity
In E1 (biological constraints), neurons NOT receiving direct factor input show significant selectivity:
"""

    for f, data in results['e1_biological']['emergent_selectivity'].items():
        report += f"- **{f.capitalize()}:** {data['frac_emergent_significant']*100:.0f}% of non-receivers significant\n"

    report += """
### 3. Weight-Selectivity Relationships
Correlation between connection weights and selectivity similarity indicates
whether neurons with similar selectivity are more strongly connected.

### 4. Effective Connectivity
The regression model decomposes selectivity into:
- **β_direct:** Contribution from direct input
- **β_EE/β_IE:** Contribution from E and I inputs weighted by their selectivity

---

## Graded Ablation Analysis

Shows how mean selectivity changes as connection types are scaled from 0 (ablated) to 1 (intact).
See `figures/graded_ablation_e1.png` for visualization.

---

## Generated Figures

1. `roc_selectivity_e0.png` / `roc_selectivity_e1.png` - ROC selectivity distributions
2. `weight_sel_scatter_*_*.png` - Weight vs selectivity similarity scatter plots
3. `effective_connectivity_e1.png` - Regression coefficients
4. `graded_ablation_e1.png` - Selectivity vs ablation level
5. `emergent_selectivity_e1.png` - Receivers vs non-receivers comparison

---

*Report generated: {results['timestamp']}*
"""

    with open(output_dir / 'CONNECTIVITY_ANALYSIS_REPORT_v2.md', 'w') as f:
        f.write(report)

    print(f"  Report saved: {output_dir / 'CONNECTIVITY_ANALYSIS_REPORT_v2.md'}")


if __name__ == '__main__':
    run_analysis()
