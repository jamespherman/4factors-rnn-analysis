#!/usr/bin/env python3
"""
Connectivity-Selectivity Analysis Script.

Analyzes the relationship between recurrent connectivity (E->E, I->E, etc.)
and factor selectivity using ROC-AUC based selectivity matching the neuronal
analysis pipeline.

Usage:
    python scripts/analyze_connectivity_selectivity.py

Output: results/biological_input_experiments/connectivity_analysis/
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EIRNN, create_model_from_data
from src.data_loader import load_mat_file, validate_data, find_target_onset_bin
from src.input_assignment import InputAssignment, create_biological_assignment
from src.analysis.roc_selectivity import (
    compute_all_roc_selectivities, compute_emergent_roc_selectivity,
    find_visual_epoch_bins, ROCSelectivityResult
)
from src.analysis.connectivity_selectivity import (
    extract_weight_blocks, analyze_all_weight_selectivity_correlations,
    fit_effective_connectivity_model, compute_selectivity_flow,
    summarize_connectivity_selectivity_analysis, EffectiveConnectivityResult
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    # Feb 2026 results used the stale 06:31 export (14 E / 28 I); the canonical
    # export is data/rnn_export_Newton_08_15_2025_SC.mat (41 E / 13 I)
    'data_path': 'data/export_2026-01-22_0631/rnn_export_Newton_08_15_2025_SC.mat',
    'results_dir': 'results/biological_input_experiments',
    'output_dir': 'results/biological_input_experiments/connectivity_analysis',

    # ROC parameters (matching neuronal pipeline)
    'visual_window_start_ms': 50.0,
    'visual_window_end_ms': 250.0,
    'min_trials': 5,
    'n_perm': 200,
    'n_bootstrap': 200,

    # Model parameters (must match training)
    'input_embed_dim': 56,
    'input_embed_type': 'attention',
    'attention_heads': 4,
    'learnable_h0': True,
    'tau': 50.0,
    'dt': 25.0,
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_trained_model(model_path: str, n_exc: int, n_inh: int, n_inputs: int, device: str = 'cpu') -> EIRNN:
    """Load a trained model from checkpoint."""
    model = create_model_from_data(
        arch_version='feb2026',  # model code these results were trained with
        n_classic=n_exc,
        n_interneuron=n_inh,
        n_inputs=n_inputs,
        enforce_ratio=True,
        dt=CONFIG['dt'],
        tau=CONFIG['tau'],
        input_embed_dim=CONFIG['input_embed_dim'],
        input_embed_type=CONFIG['input_embed_type'],
        attention_heads=CONFIG['attention_heads'],
        learnable_h0=CONFIG['learnable_h0'],
        device=device
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def prepare_data(data: dict, config: dict) -> Tuple[torch.Tensor, Dict[str, np.ndarray], int]:
    """Prepare validation data and trial factors."""
    # Get firing rates and trial info
    firing_rates = data['firing_rates']  # [n_neurons, n_bins, n_trials]
    n_neurons, n_bins, n_trials = firing_rates.shape

    # Construct input tensor (standard 14-channel input)
    inputs = []
    input_keys = [
        'input_fixation_on', 'input_go_signal', 'input_reward_on',
        'input_eye_x', 'input_eye_y',
        'input_is_face', 'input_is_nonface', 'input_is_bullseye',
        'input_high_salience', 'input_low_salience'
    ]

    for key in input_keys:
        if key in data:
            signal = data[key]
            if signal.ndim == 1:
                signal = np.tile(signal[None, :], (n_bins, 1))
            inputs.append(signal.T)  # [n_trials, n_bins]

    # Add target location (4 channels)
    target_loc = data['input_target_loc']  # [4, n_bins, n_trials]
    for i in range(4):
        inputs.append(target_loc[i].T)  # [n_trials, n_bins]

    # Stack: [n_trials, n_bins, n_channels]
    inputs = np.stack(inputs, axis=2)
    inputs = torch.tensor(inputs, dtype=torch.float32)

    # Trial factors
    trial_factors = {
        'reward': data['trial_reward'].astype(float),
        'probability': data['trial_probability'].astype(float),
        'salience': data['trial_salience'].astype(float),
        'identity': data['trial_identity'].astype(float)
    }

    # Find target onset bin
    target_onset_bin = find_target_onset_bin(data['input_target_loc'])

    return inputs, trial_factors, target_onset_bin


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_roc_selectivity_comparison(
    roc_results_e0: Dict[str, ROCSelectivityResult],
    roc_results_e1: Dict[str, ROCSelectivityResult],
    output_path: str
):
    """Compare ROC selectivity between baseline and biological experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    factors = ['reward', 'probability', 'salience', 'identity']

    for idx, factor in enumerate(factors):
        ax = axes[idx // 2, idx % 2]

        if factor in roc_results_e0 and factor in roc_results_e1:
            auc_e0 = roc_results_e0[factor].auc
            auc_e1 = roc_results_e1[factor].auc

            # Remove NaNs
            valid = ~np.isnan(auc_e0) & ~np.isnan(auc_e1)

            ax.scatter(auc_e0[valid], auc_e1[valid], alpha=0.6, s=50)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)

            # Correlation
            if valid.sum() > 2:
                r, _ = np.corrcoef(auc_e0[valid], auc_e1[valid])[0, 1], None
                ax.set_title(f'{factor.capitalize()} (r={r:.2f})')
            else:
                ax.set_title(factor.capitalize())

            ax.set_xlabel('E0 Baseline AUC')
            ax.set_ylabel('E1 Biological AUC')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_emergent_selectivity_roc(
    roc_results: Dict[str, ROCSelectivityResult],
    output_path: str
):
    """Plot emergent selectivity analysis using ROC-AUC."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    factors = ['reward', 'probability', 'salience', 'identity']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    for idx, factor in enumerate(factors):
        ax = axes[idx]

        if factor not in roc_results:
            continue

        result = roc_results[factor]
        emergent = compute_emergent_roc_selectivity(result)

        if result.receives_input is not None and emergent['n_non_receivers'] > 0:
            receiver_mask = result.receives_input
            non_receiver_mask = ~receiver_mask

            # Box plot comparing receivers vs non-receivers
            data_to_plot = []
            labels = []

            if receiver_mask.sum() > 0:
                receiver_auc = result.auc[receiver_mask]
                receiver_auc = receiver_auc[~np.isnan(receiver_auc)]
                if len(receiver_auc) > 0:
                    data_to_plot.append(receiver_auc)
                    labels.append(f'Receivers\n(n={len(receiver_auc)})')

            if non_receiver_mask.sum() > 0:
                non_receiver_auc = result.auc[non_receiver_mask]
                non_receiver_auc = non_receiver_auc[~np.isnan(non_receiver_auc)]
                if len(non_receiver_auc) > 0:
                    data_to_plot.append(non_receiver_auc)
                    labels.append(f'Non-receivers\n(n={len(non_receiver_auc)})')

            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                bp['boxes'][0].set_facecolor(colors[idx])
                if len(bp['boxes']) > 1:
                    bp['boxes'][1].set_facecolor('lightgray')

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('ROC-AUC')
        ax.set_title(f'{factor.capitalize()}\n({emergent["frac_emergent_significant"]*100:.0f}% emergent sig.)')
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_weight_selectivity_scatter(
    weight_sel_corr: Dict[str, Dict[str, Any]],
    output_path: str
):
    """Plot weight vs selectivity similarity for each connection type."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    conn_types = ['EE', 'EI', 'IE', 'II']
    titles = ['E → E', 'I → E', 'E → I', 'I → I']

    # Use reward as example factor
    factor = 'reward'

    for idx, (conn_type, title) in enumerate(zip(conn_types, titles)):
        ax = axes[idx // 2, idx % 2]

        if conn_type in weight_sel_corr and factor in weight_sel_corr[conn_type]:
            result = weight_sel_corr[conn_type][factor]

            if result.n_pairs > 0:
                ax.scatter(result.weight_values, result.selectivity_similarity,
                          alpha=0.3, s=10)

                # Add regression line if significant
                if result.p_value < 0.05:
                    z = np.polyfit(result.weight_values, result.selectivity_similarity, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(result.weight_values.min(), result.weight_values.max(), 100)
                    ax.plot(x_line, p(x_line), 'r-', linewidth=2,
                           label=f'r={result.correlation:.2f}*')
                else:
                    ax.text(0.05, 0.95, f'r={result.correlation:.2f} (ns)',
                           transform=ax.transAxes, va='top')

        ax.set_xlabel('|Weight|')
        ax.set_ylabel('Selectivity Similarity\n(negative distance)')
        ax.set_title(f'{title} (n={result.n_pairs if result.n_pairs else 0})')
        if result.p_value < 0.05:
            ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_effective_connectivity_coefficients(
    ec_results: Dict[str, EffectiveConnectivityResult],
    output_path: str
):
    """Plot effective connectivity coefficients for each factor."""
    factors = list(ec_results.keys())
    n_factors = len(factors)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_factors)
    width = 0.25

    beta_direct = [ec_results[f].beta_direct for f in factors]
    beta_E = [ec_results[f].beta_E for f in factors]
    beta_I = [ec_results[f].beta_I for f in factors]

    bars1 = ax.bar(x - width, beta_direct, width, label='β_direct (input)', color='#3498db')
    bars2 = ax.bar(x, beta_E, width, label='β_E (E input)', color='#e74c3c')
    bars3 = ax.bar(x + width, beta_I, width, label='β_I (I input)', color='#2ecc71')

    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Coefficient')
    ax.set_title('Effective Connectivity Model Coefficients')
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in factors])
    ax.legend()

    # Add R² values
    for i, factor in enumerate(factors):
        r2 = ec_results[factor].r_squared
        ax.text(i, ax.get_ylim()[1] * 0.9, f'R²={r2:.2f}',
               ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_connection_selectivity_heatmap(
    W_rec: np.ndarray,
    selectivity: np.ndarray,
    n_exc: int,
    factor_name: str,
    output_path: str
):
    """Plot weight matrix with neurons sorted by selectivity."""
    n_total = W_rec.shape[0]

    # Sort E and I neurons separately by selectivity
    e_order = np.argsort(selectivity[:n_exc])[::-1]
    i_order = np.argsort(selectivity[n_exc:])[::-1] + n_exc

    # Combined order
    order = np.concatenate([e_order, i_order])

    # Reorder weight matrix
    W_sorted = W_rec[order][:, order]
    sel_sorted = selectivity[order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 0.05]})

    # Main heatmap
    ax = axes[0]
    im = ax.imshow(W_sorted, cmap='RdBu_r', aspect='auto',
                   vmin=-np.percentile(np.abs(W_rec), 95),
                   vmax=np.percentile(np.abs(W_rec), 95))

    # Add lines to separate E and I
    ax.axhline(n_exc - 0.5, color='black', linewidth=2)
    ax.axvline(n_exc - 0.5, color='black', linewidth=2)

    ax.set_xlabel('Source neuron (sorted by selectivity)')
    ax.set_ylabel('Target neuron (sorted by selectivity)')
    ax.set_title(f'{factor_name.capitalize()} - W_rec sorted by selectivity')

    # Add E/I labels
    ax.text(n_exc/2, -2, 'E', ha='center', fontsize=12, fontweight='bold')
    ax.text(n_exc + (n_total-n_exc)/2, -2, 'I', ha='center', fontsize=12, fontweight='bold')
    ax.text(-2, n_exc/2, 'E', ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    ax.text(-2, n_exc + (n_total-n_exc)/2, 'I', ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)

    # Colorbar
    plt.colorbar(im, cax=axes[1], label='Weight')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_selectivity_flow_network(
    flow_results: Dict[str, Any],
    selectivity: np.ndarray,
    receives_input: np.ndarray,
    n_exc: int,
    factor_name: str,
    output_path: str
):
    """Visualize selectivity flow as a network graph."""
    try:
        import networkx as nx
    except ImportError:
        print("  NetworkX not installed, skipping network visualization")
        return

    n_total = len(selectivity)

    # Create graph
    G = nx.DiGraph()

    # Add nodes
    for i in range(n_total):
        node_type = 'E' if i < n_exc else 'I'
        is_source = receives_input[i] and np.abs(selectivity[i] - 0.5) > 0.1
        is_target = not receives_input[i] and np.abs(selectivity[i] - 0.5) > 0.1

        G.add_node(i,
                  neuron_type=node_type,
                  selectivity=selectivity[i],
                  is_source=is_source,
                  is_target=is_target)

    # Add edges from flow analysis
    for conn in flow_results.get('connections', []):
        G.add_edge(conn['source'], conn['target'], weight=conn['weight'])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Position nodes: sources on left, targets on right, others in middle
    pos = {}
    sources = [n for n, d in G.nodes(data=True) if d.get('is_source', False)]
    targets = [n for n, d in G.nodes(data=True) if d.get('is_target', False)]
    others = [n for n in G.nodes() if n not in sources and n not in targets]

    for i, n in enumerate(sources):
        pos[n] = (0, i / (len(sources) + 1))
    for i, n in enumerate(targets):
        pos[n] = (2, i / (len(targets) + 1))
    for i, n in enumerate(others):
        pos[n] = (1, i / (len(others) + 1))

    # Draw
    node_colors = []
    for n in G.nodes():
        sel = G.nodes[n]['selectivity']
        node_colors.append(plt.cm.RdBu_r((sel - 0) / 1.0))

    node_sizes = [200 if G.nodes[n].get('is_source') or G.nodes[n].get('is_target') else 50
                 for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)

    if G.edges():
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, alpha=0.5,
                              width=[w * 10 for w in edge_weights],
                              edge_color='gray', ax=ax)

    ax.set_title(f'{factor_name.capitalize()} - Selectivity Flow\n'
                f'({len(sources)} sources → {len(targets)} targets via {G.number_of_edges()} connections)')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def run_analysis():
    """Run full connectivity-selectivity analysis."""
    print("=" * 60)
    print("CONNECTIVITY-SELECTIVITY ANALYSIS")
    print("=" * 60)

    # Setup paths
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(CONFIG['results_dir'])
    e0_dir = results_dir / 'experiments' / 'E0_baseline'
    e1_dir = results_dir / 'experiments' / 'E1_biological'

    # Load data
    print("\nLoading data...")
    data = load_mat_file(CONFIG['data_path'])
    validate_data(data)

    neuron_type = data['neuron_type']
    n_neurons = int(data['n_neurons'])
    n_exc = int((neuron_type == 1).sum())
    n_inh = int((neuron_type == 2).sum())
    bin_size_ms = float(data['bin_size_ms'])

    print(f"  Neurons: {n_neurons} ({n_exc} E, {n_inh} I)")

    # Prepare data
    inputs, trial_factors, target_onset_bin = prepare_data(data, CONFIG)
    print(f"  Target onset bin: {target_onset_bin}")

    # Visual epoch bins
    epoch_bins = find_visual_epoch_bins(
        target_onset_bin, bin_size_ms,
        CONFIG['visual_window_start_ms'],
        CONFIG['visual_window_end_ms']
    )
    print(f"  Visual epoch bins: {epoch_bins}")

    # Train/val split (same as training)
    np.random.seed(42)
    n_trials = inputs.shape[0]
    indices = np.arange(n_trials)
    np.random.shuffle(indices)
    n_val = int(n_trials * 0.2)
    val_idx = indices[:n_val]

    val_inputs = inputs[val_idx]
    val_factors = {k: v[val_idx] for k, v in trial_factors.items()}

    device = 'cpu'

    # =========================================================================
    # E0 BASELINE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60)
    print("E0 BASELINE ANALYSIS")
    print("=" * 60)

    # Load E0 model
    print("\nLoading E0 model...")
    model_e0 = load_trained_model(
        str(e0_dir / 'model_best.pt'),
        n_exc, n_inh, inputs.shape[2], device
    )

    # Get model outputs (all neurons including hidden)
    print("Computing model outputs...")
    n_model_total = model_e0.n_total  # 140 (112 E + 28 I)
    n_model_exc = model_e0.n_exc      # 112

    with torch.no_grad():
        rates_e0, _ = model_e0(val_inputs.to(device))
        rates_e0 = rates_e0.cpu().numpy()
        # For ROC selectivity: only use recorded neurons
        rates_e0_recorded = rates_e0[:, :, :n_neurons]

    # Compute ROC selectivity for recorded neurons only
    print("Computing ROC selectivity (recorded neurons)...")
    roc_results_e0 = compute_all_roc_selectivities(
        rates_e0_recorded, val_factors, epoch_bins,
        assignment=None,  # E0 has no assignment (all receive all)
        min_trials=CONFIG['min_trials'],
        n_perm=CONFIG['n_perm'],
        n_bootstrap=CONFIG['n_bootstrap']
    )

    for factor, result in roc_results_e0.items():
        print(f"  {factor}: mean AUC={result.mean_auc:.3f}, {result.n_significant}/{result.n_neurons} significant")

    # Get weights (for full model)
    W_rec_e0 = model_e0.W_rec.detach().cpu().numpy()

    # For weight-selectivity analysis, expand selectivity to full model size
    # Recorded E: indices 0:n_exc, Recorded I: indices n_exc:n_neurons
    # In model: E indices 0:n_model_exc, I indices n_model_exc:n_model_total
    # Map: recorded E (0:14) -> model E (0:14), recorded I (14:42) -> model I (112:140)

    def expand_selectivity_to_model(sel_recorded, n_recorded_e, n_recorded_i, n_model_e, n_model_i):
        """Expand recorded neuron selectivity to full model dimensions."""
        n_model_total = n_model_e + n_model_i
        sel_full = np.full(n_model_total, np.nan)
        # Recorded E neurons map to first n_recorded_e positions of E block
        sel_full[:n_recorded_e] = sel_recorded[:n_recorded_e]
        # Recorded I neurons map to first n_recorded_i positions of I block
        sel_full[n_model_e:n_model_e + n_recorded_i] = sel_recorded[n_recorded_e:]
        return sel_full

    # Weight-selectivity correlations using full model weights
    print("\nAnalyzing weight-selectivity correlations...")
    selectivity_by_factor_e0_full = {}
    for f, r in roc_results_e0.items():
        selectivity_by_factor_e0_full[f] = expand_selectivity_to_model(
            r.auc, n_exc, n_inh, n_model_exc, n_model_total - n_model_exc
        )

    weight_sel_corr_e0 = analyze_all_weight_selectivity_correlations(
        W_rec_e0, selectivity_by_factor_e0_full, n_model_exc
    )

    for conn_type in ['EE', 'EI', 'IE', 'II']:
        for factor in selectivity_by_factor_e0_full.keys():
            result = weight_sel_corr_e0[conn_type][factor]
            sig = '*' if result.p_value < 0.05 else ''
            print(f"  {conn_type}/{factor}: r={result.correlation:.3f}{sig}")

    # =========================================================================
    # E1 BIOLOGICAL ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60)
    print("E1 BIOLOGICAL ANALYSIS")
    print("=" * 60)

    # Load E1 model
    print("\nLoading E1 model...")

    # Need to handle the different input shape for E1
    # Load E1 assignment
    assignment_path = e1_dir / 'assignment.npy'
    if assignment_path.exists():
        assignment_data = np.load(assignment_path, allow_pickle=True).item()
        rf_locations = np.zeros(n_neurons, dtype=int)
        for i in range(n_exc):
            rf_locations[i] = (i % 4) + 1

        assignment_e1 = create_biological_assignment(
            n_neurons, neuron_type, rf_locations,
            e_fractions={'reward': 0.30, 'probability': 0.30, 'salience': 0.75, 'identity': 0.75},
            i_fractions={'reward': 0.40, 'probability': 0.40, 'salience': 0.25, 'identity': 0.25},
            seed=42,
            experiment_type='E1_BIO'
        )
    else:
        print("  Warning: E1 assignment not found, creating default")
        rf_locations = np.zeros(n_neurons, dtype=int)
        for i in range(n_exc):
            rf_locations[i] = (i % 4) + 1
        assignment_e1 = create_biological_assignment(
            n_neurons, neuron_type, rf_locations, seed=42
        )

    # For E1, we need to use the biological input structure
    # Load E1 model with appropriate input dimension
    # The E1 model was trained with flattened biological inputs: [n_trials, n_bins, n_neurons * 6]
    n_biological_inputs = n_neurons * 6  # 6 factors per neuron

    try:
        model_e1 = create_model_from_data(
            arch_version='feb2026',  # model code these results were trained with
            n_classic=n_exc,
            n_interneuron=n_inh,
            n_inputs=n_biological_inputs,
            enforce_ratio=True,
            dt=CONFIG['dt'],
            tau=CONFIG['tau'],
            input_embed_dim=CONFIG['input_embed_dim'],
            input_embed_type='linear',  # E1 used linear for biological inputs
            learnable_h0=CONFIG['learnable_h0'],
            device=device
        )
        model_e1.load_state_dict(torch.load(str(e1_dir / 'model_best.pt'), map_location=device))
        model_e1.eval()

        # Construct biological inputs for E1
        from src.data_loader import construct_restructured_input_tensor
        bio_inputs = construct_restructured_input_tensor(data, assignment_e1)
        bio_inputs_flat = bio_inputs.reshape(bio_inputs.shape[0], bio_inputs.shape[1], -1)
        bio_inputs_tensor = torch.tensor(bio_inputs_flat, dtype=torch.float32)
        val_bio_inputs = bio_inputs_tensor[val_idx]

        # Get E1 model outputs
        print("Computing E1 model outputs...")
        with torch.no_grad():
            rates_e1, _ = model_e1(val_bio_inputs.to(device))
            rates_e1 = rates_e1.cpu().numpy()
            # Only keep recorded neurons (first n_neurons)
            rates_e1 = rates_e1[:, :, :n_neurons]

        # Compute ROC selectivity with assignment info
        print("Computing ROC selectivity for E1...")
        roc_results_e1 = compute_all_roc_selectivities(
            rates_e1, val_factors, epoch_bins,
            assignment=assignment_e1,
            min_trials=CONFIG['min_trials'],
            n_perm=CONFIG['n_perm'],
            n_bootstrap=CONFIG['n_bootstrap']
        )

        for factor, result in roc_results_e1.items():
            emergent = compute_emergent_roc_selectivity(result)
            print(f"  {factor}: mean AUC={result.mean_auc:.3f}, "
                  f"emergent: {emergent['frac_emergent_significant']*100:.1f}% significant")

        # Get E1 weights
        W_rec_e1 = model_e1.W_rec.detach().cpu().numpy()
        n_model_exc_e1 = model_e1.n_exc
        n_model_total_e1 = model_e1.n_total

        # Weight-selectivity correlations for E1
        print("\nAnalyzing weight-selectivity correlations for E1...")
        selectivity_by_factor_e1_full = {}
        for f, r in roc_results_e1.items():
            selectivity_by_factor_e1_full[f] = expand_selectivity_to_model(
                r.auc, n_exc, n_inh, n_model_exc_e1, n_model_total_e1 - n_model_exc_e1
            )

        weight_sel_corr_e1 = analyze_all_weight_selectivity_correlations(
            W_rec_e1, selectivity_by_factor_e1_full, n_model_exc_e1
        )

        # Effective connectivity models
        print("\nFitting effective connectivity models...")
        ec_results_e1 = {}
        for factor, result in roc_results_e1.items():
            receives = result.receives_input if result.receives_input is not None else np.ones(n_neurons, dtype=bool)
            ec_results_e1[factor] = fit_effective_connectivity_model(
                W_rec_e1, result.auc, receives, n_exc, factor
            )
            print(f"  {factor}: R²={ec_results_e1[factor].r_squared:.3f}, "
                  f"β_E={ec_results_e1[factor].beta_E:.4f}, β_I={ec_results_e1[factor].beta_I:.4f}")

        # Selectivity flow
        print("\nAnalyzing selectivity flow...")
        flow_results_e1 = {}
        for factor, result in roc_results_e1.items():
            receives = result.receives_input if result.receives_input is not None else np.ones(n_neurons, dtype=bool)
            flow_results_e1[factor] = compute_selectivity_flow(
                W_rec_e1, result.auc, receives, n_exc
            )
            print(f"  {factor}: {flow_results_e1[factor]['n_sources']} sources → "
                  f"{flow_results_e1[factor]['n_targets']} targets via "
                  f"{flow_results_e1[factor]['n_connections']} strong connections")

        e1_loaded = True

    except Exception as e:
        print(f"  Warning: Could not load E1 model: {e}")
        e1_loaded = False
        roc_results_e1 = {}
        weight_sel_corr_e1 = {}
        ec_results_e1 = {}
        flow_results_e1 = {}
        W_rec_e1 = None

    # =========================================================================
    # GENERATE FIGURES
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # 1. ROC selectivity comparison
    if e1_loaded:
        print("  Plotting ROC selectivity comparison...")
        plot_roc_selectivity_comparison(
            roc_results_e0, roc_results_e1,
            str(figures_dir / 'roc_selectivity_comparison.png')
        )

    # 2. Emergent selectivity (E1 only)
    if e1_loaded:
        print("  Plotting emergent selectivity...")
        plot_emergent_selectivity_roc(
            roc_results_e1,
            str(figures_dir / 'emergent_selectivity_roc.png')
        )

    # 3. Weight-selectivity scatter (E0)
    print("  Plotting weight-selectivity scatter (E0)...")
    plot_weight_selectivity_scatter(
        weight_sel_corr_e0,
        str(figures_dir / 'weight_selectivity_scatter_e0.png')
    )

    # 4. Weight-selectivity scatter (E1)
    if e1_loaded:
        print("  Plotting weight-selectivity scatter (E1)...")
        plot_weight_selectivity_scatter(
            weight_sel_corr_e1,
            str(figures_dir / 'weight_selectivity_scatter_e1.png')
        )

    # 5. Effective connectivity coefficients (E1)
    if e1_loaded and ec_results_e1:
        print("  Plotting effective connectivity coefficients...")
        plot_effective_connectivity_coefficients(
            ec_results_e1,
            str(figures_dir / 'effective_connectivity_coefficients.png')
        )

    # 6. Connection-selectivity heatmaps
    print("  Plotting connection-selectivity heatmaps...")
    for factor in ['reward', 'salience']:
        if factor in roc_results_e0:
            plot_connection_selectivity_heatmap(
                W_rec_e0, roc_results_e0[factor].auc, n_exc, factor,
                str(figures_dir / f'connection_heatmap_e0_{factor}.png')
            )

        if e1_loaded and factor in roc_results_e1:
            plot_connection_selectivity_heatmap(
                W_rec_e1, roc_results_e1[factor].auc, n_exc, factor,
                str(figures_dir / f'connection_heatmap_e1_{factor}.png')
            )

    # 7. Selectivity flow networks (E1)
    if e1_loaded:
        print("  Plotting selectivity flow networks...")
        for factor in ['reward', 'salience']:
            if factor in roc_results_e1 and factor in flow_results_e1:
                result = roc_results_e1[factor]
                receives = result.receives_input if result.receives_input is not None else np.ones(n_neurons, dtype=bool)
                plot_selectivity_flow_network(
                    flow_results_e1[factor], result.auc, receives, n_exc, factor,
                    str(figures_dir / f'selectivity_flow_{factor}.png')
                )

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Compile results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'e0_baseline': {
            'roc_selectivity': {f: r.to_dict() for f, r in roc_results_e0.items()},
            'weight_selectivity_correlations': {
                conn: {f: {'correlation': r.correlation, 'p_value': r.p_value, 'n_pairs': r.n_pairs}
                       for f, r in factors.items()}
                for conn, factors in weight_sel_corr_e0.items()
            }
        }
    }

    if e1_loaded:
        results['e1_biological'] = {
            'roc_selectivity': {f: r.to_dict() for f, r in roc_results_e1.items()},
            'emergent_selectivity': {f: compute_emergent_roc_selectivity(r) for f, r in roc_results_e1.items()},
            'weight_selectivity_correlations': {
                conn: {f: {'correlation': r.correlation, 'p_value': r.p_value, 'n_pairs': r.n_pairs}
                       for f, r in factors.items()}
                for conn, factors in weight_sel_corr_e1.items()
            },
            'effective_connectivity': {
                f: {'beta_direct': r.beta_direct, 'beta_E': r.beta_E, 'beta_I': r.beta_I, 'r_squared': r.r_squared}
                for f, r in ec_results_e1.items()
            },
            'selectivity_flow': {
                f: {'n_sources': flow['n_sources'], 'n_targets': flow['n_targets'],
                    'n_connections': flow['n_connections'], 'connection_types': flow['connection_type_counts']}
                for f, flow in flow_results_e1.items()
            }
        }

    # Save JSON results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Results saved to {output_dir / 'results.json'}")
    print(f"  Figures saved to {figures_dir}")

    # =========================================================================
    # GENERATE REPORT
    # =========================================================================
    print("\nGenerating report...")
    generate_report(results, output_dir, e1_loaded)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


def generate_report(results: Dict, output_dir: Path, e1_loaded: bool):
    """Generate markdown report."""
    report = f"""# Connectivity-Selectivity Analysis Report

**Date:** {results['timestamp'][:10]}

---

## Overview

This analysis examines the relationship between recurrent connectivity (E→E, I→E, etc.)
and factor selectivity in the trained RNN models. Selectivity is computed using ROC-AUC,
matching the neuronal analysis pipeline.

---

## Methods

### ROC-AUC Selectivity
- **Metric**: Area Under ROC Curve via Mann-Whitney U statistic
- **Range**: [0, 1], where 0.5 = chance
- **Significance**: Permutation test (n={results['config']['n_perm']}) + Bootstrap CI

### Visual Epoch Window
- **Start**: {results['config']['visual_window_start_ms']} ms post-target onset
- **End**: {results['config']['visual_window_end_ms']} ms post-target onset
- Matches SC visual epoch in neuronal analysis

---

## E0 Baseline Results

### ROC Selectivity Summary
"""

    # E0 ROC results
    for factor, data in results['e0_baseline']['roc_selectivity'].items():
        report += f"- **{factor.capitalize()}**: Mean AUC = {data['mean_auc']:.3f}, "
        report += f"{data['n_significant']}/{len(data['auc'])} significant\n"

    report += "\n### Weight-Selectivity Correlations (E0)\n\n"
    report += "| Connection | Factor | Correlation | p-value | Significant |\n"
    report += "|------------|--------|-------------|---------|-------------|\n"

    for conn_type in ['EE', 'EI', 'IE', 'II']:
        for factor, data in results['e0_baseline']['weight_selectivity_correlations'][conn_type].items():
            sig = '✓' if data['p_value'] < 0.05 else ''
            report += f"| {conn_type} | {factor} | {data['correlation']:.3f} | {data['p_value']:.3f} | {sig} |\n"

    if e1_loaded:
        report += "\n---\n\n## E1 Biological Results\n\n"

        # Emergent selectivity
        report += "### Emergent Selectivity (ROC-AUC)\n\n"
        report += "| Factor | Non-receivers | Significant | Fraction |\n"
        report += "|--------|---------------|-------------|----------|\n"

        for factor, data in results['e1_biological']['emergent_selectivity'].items():
            report += f"| {factor.capitalize()} | {data['n_non_receivers']} | "
            report += f"{data['n_emergent_significant']} | {data['frac_emergent_significant']*100:.1f}% |\n"

        # Effective connectivity
        report += "\n### Effective Connectivity Model\n\n"
        report += "Model: `selectivity = β₀ + β_direct × receives_input + β_E × E_input + β_I × I_input`\n\n"
        report += "| Factor | β_direct | β_E | β_I | R² |\n"
        report += "|--------|----------|-----|-----|----|\n"

        for factor, data in results['e1_biological']['effective_connectivity'].items():
            report += f"| {factor.capitalize()} | {data['beta_direct']:.4f} | "
            report += f"{data['beta_E']:.4f} | {data['beta_I']:.4f} | {data['r_squared']:.3f} |\n"

        # Selectivity flow
        report += "\n### Selectivity Flow\n\n"
        report += "| Factor | Sources | Targets | Strong Connections |\n"
        report += "|--------|---------|---------|-------------------|\n"

        for factor, data in results['e1_biological']['selectivity_flow'].items():
            report += f"| {factor.capitalize()} | {data['n_sources']} | "
            report += f"{data['n_targets']} | {data['n_connections']} |\n"

    report += """
---

## Interpretation

### Weight-Selectivity Correlations
- **Positive correlation** in E→E: Neurons with similar selectivity have stronger excitatory connections
- **Negative correlation** in I→E: Inhibitory connections may suppress similar selectivity (competition)

### Effective Connectivity
- **β_E > 0**: Excitatory inputs from selective neurons increase target selectivity
- **β_I < 0**: Inhibitory inputs from selective neurons decrease target selectivity
- **β_direct**: Direct input contribution independent of recurrence

### Emergent Selectivity
Neurons that do NOT receive direct factor input but show significant selectivity
demonstrate that recurrent dynamics propagate factor information through the network.

---

## Generated Figures

1. `roc_selectivity_comparison.png` - E0 vs E1 selectivity scatter
2. `emergent_selectivity_roc.png` - Emergent selectivity boxplots
3. `weight_selectivity_scatter_*.png` - Weight vs selectivity by connection type
4. `effective_connectivity_coefficients.png` - β coefficients by factor
5. `connection_heatmap_*.png` - W_rec sorted by selectivity
6. `selectivity_flow_*.png` - Network visualization of selectivity propagation

---

*Report generated: {results['timestamp']}*
"""

    with open(output_dir / 'CONNECTIVITY_ANALYSIS_REPORT.md', 'w') as f:
        f.write(report)

    print(f"  Report saved to {output_dir / 'CONNECTIVITY_ANALYSIS_REPORT.md'}")


if __name__ == '__main__':
    run_analysis()
