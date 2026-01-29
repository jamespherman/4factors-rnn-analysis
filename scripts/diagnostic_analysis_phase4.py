"""
Phase 4 Diagnostic Analysis Script.

Conducts deep analysis to understand model performance:
1. Per-neuron analysis (scatter plots, E vs I comparison)
2. Temporal error analysis (time-resolved error, residual timecourse)
3. Input-driven analysis
4. Population structure (PCA comparison)
5. Trial-to-trial variability (Fano factors)

Usage:
    python scripts/diagnostic_analysis_phase4.py --data data/rnn_export_Newton_08_15_2025_SC.mat --model results/phase4/attention_learnable_h0_model_best.pt
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EIRNN, create_model_from_data
from src.data_loader import load_session, train_val_split


def load_model_and_data(model_path: str, data_path: str, device: str = 'cpu', config_path: str = None):
    """Load trained model and data."""
    # Load checkpoint (just the state dict)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    # Try to load config from result json if available
    config = {}
    if config_path:
        with open(config_path, 'r') as f:
            result = json.load(f)
            config = result.get('config', {})
    else:
        # Try to find result.json in same directory
        result_path = model_path.replace('_model_best.pt', '_result.json')
        try:
            with open(result_path, 'r') as f:
                result = json.load(f)
                config = result.get('config', {})
        except:
            pass

    # Load data
    dataset = load_session(data_path)
    train_idx, val_idx = train_val_split(dataset, val_fraction=0.2, seed=42)

    all_data = dataset.get_all_trials()
    train_data = {
        'inputs': all_data['inputs'][train_idx],
        'targets': all_data['targets'][train_idx],
        'mask': all_data['mask'][train_idx],
    }
    val_data = {
        'inputs': all_data['inputs'][val_idx],
        'targets': all_data['targets'][val_idx],
        'mask': all_data['mask'][val_idx],
    }

    neuron_info = dataset.get_neuron_info()
    n_inputs = dataset.get_input_dim()
    bin_size_ms = dataset.bin_size_ms

    # Recreate model architecture
    model = create_model_from_data(
        n_classic=neuron_info['n_exc'],
        n_interneuron=neuron_info['n_inh'],
        n_inputs=n_inputs,
        enforce_ratio=True,
        dt=bin_size_ms,
        input_embed_dim=config.get('input_embed_dim', 56),
        input_embed_type=config.get('input_embed_type', 'attention'),
        attention_heads=config.get('attention_heads', 4),
        learnable_alpha=config.get('learnable_alpha', 'none'),
        alpha_e_init=config.get('alpha_e_init', 0.5),
        alpha_i_init=config.get('alpha_i_init', 0.5),
        learnable_h0=config.get('learnable_h0', True),
        h0_init=config.get('h0_init', 0.1),
    )
    model = model.to(device)

    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()

    return model, train_data, val_data, neuron_info, n_inputs, bin_size_ms


def compute_predictions(model: EIRNN, data: dict, device: str, neuron_info: dict):
    """Compute model predictions.

    Note: Model outputs all neurons (including hidden E for 4:1 ratio).
    We slice to get only recorded neurons to match targets.
    """
    model.eval()
    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        model_rates_full, activations = model(inputs)

        # Get recorded neuron counts
        n_e_recorded = neuron_info['n_exc']  # Recorded E (classic)
        n_i_recorded = neuron_info['n_inh']  # Recorded I (interneuron)
        n_e_total = model.n_exc  # Total E in model (includes hidden)

        # Slice to get only recorded neurons
        # Model output: [E_recorded, E_hidden, I_recorded]
        # Target format: [E_recorded, I_recorded]
        e_recorded = model_rates_full[:, :, :n_e_recorded]  # First n_e_recorded are recorded E
        i_recorded = model_rates_full[:, :, n_e_total:n_e_total + n_i_recorded]  # I starts at n_e_total
        model_rates = torch.cat([e_recorded, i_recorded], dim=2)

        return {
            'model_rates': model_rates.cpu().numpy(),  # [trials, time, n_recorded]
            'targets': targets.cpu().numpy(),
            'activations': activations.cpu().numpy(),  # [trials, time, n_total]
            'inputs': inputs.cpu().numpy()
        }


def analyze_per_neuron(predictions: dict, neuron_info: dict, output_dir: Path):
    """
    Per-neuron correlation analysis.

    Outputs:
    - Scatter plot of model vs real PSTH per neuron
    - Histogram of per-neuron correlations
    - E vs I comparison
    """
    model_rates = predictions['model_rates']  # [trials, time, n_recorded]
    targets = predictions['targets']

    # Trial-averaged PSTHs
    model_psth = model_rates.mean(axis=0)  # [time, n_recorded]
    target_psth = targets.mean(axis=0)

    n_recorded = target_psth.shape[1]

    # Compute per-neuron correlations
    correlations = np.zeros(n_recorded)
    for i in range(n_recorded):
        r = np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1]
        correlations[i] = r if not np.isnan(r) else 0.0

    # Get E vs I labels
    n_e_recorded = neuron_info['n_exc']
    e_mask = np.arange(n_recorded) < n_e_recorded
    i_mask = ~e_mask

    # Figure 1: Histogram of correlations
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-0.5, 1.0, 31)
    ax.hist(correlations[e_mask], bins=bins, alpha=0.7, label=f'E neurons (n={e_mask.sum()})', color='blue')
    ax.hist(correlations[i_mask], bins=bins, alpha=0.7, label=f'I neurons (n={i_mask.sum()})', color='red')
    ax.axvline(correlations.mean(), color='k', linestyle='--', label=f'Mean: {correlations.mean():.3f}')
    ax.set_xlabel('PSTH Correlation')
    ax.set_ylabel('Count')
    ax.set_title('Per-Neuron PSTH Correlations')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'per_neuron_correlation_histogram.png', dpi=150)
    plt.close()

    # Figure 2: E vs I boxplot comparison
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([correlations[e_mask], correlations[i_mask]], labels=['E neurons', 'I neurons'])
    ax.set_ylabel('PSTH Correlation')
    ax.set_title(f'E vs I Correlation Comparison\nE: {correlations[e_mask].mean():.3f}, I: {correlations[i_mask].mean():.3f}')
    plt.tight_layout()
    plt.savefig(output_dir / 'e_vs_i_correlation.png', dpi=150)
    plt.close()

    # Figure 3: Scatter plot of best/worst neurons
    # Best 4 and worst 4
    sorted_idx = np.argsort(correlations)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, idx in enumerate(sorted_idx[-4:]):  # Best 4
        ax = axes[0, i]
        ax.plot(target_psth[:, idx], 'b-', label='Real', alpha=0.8)
        ax.plot(model_psth[:, idx], 'r-', label='Model', alpha=0.8)
        neuron_type = 'E' if idx < n_e_recorded else 'I'
        ax.set_title(f'Best #{4-i} ({neuron_type}, r={correlations[idx]:.3f})')
        if i == 0:
            ax.legend()
        ax.set_xlabel('Time bin')
        ax.set_ylabel('Firing rate')

    for i, idx in enumerate(sorted_idx[:4]):  # Worst 4
        ax = axes[1, i]
        ax.plot(target_psth[:, idx], 'b-', label='Real', alpha=0.8)
        ax.plot(model_psth[:, idx], 'r-', label='Model', alpha=0.8)
        neuron_type = 'E' if idx < n_e_recorded else 'I'
        ax.set_title(f'Worst #{i+1} ({neuron_type}, r={correlations[idx]:.3f})')
        ax.set_xlabel('Time bin')
        ax.set_ylabel('Firing rate')

    plt.suptitle('Best and Worst Fit Neurons (PSTH)')
    plt.tight_layout()
    plt.savefig(output_dir / 'best_worst_neurons.png', dpi=150)
    plt.close()

    return {
        'per_neuron_correlations': correlations.tolist(),
        'mean_correlation': float(correlations.mean()),
        'e_mean_correlation': float(correlations[e_mask].mean()),
        'i_mean_correlation': float(correlations[i_mask].mean()),
        'e_std_correlation': float(correlations[e_mask].std()),
        'i_std_correlation': float(correlations[i_mask].std())
    }


def analyze_temporal_error(predictions: dict, output_dir: Path):
    """
    Temporal error analysis.

    Outputs:
    - Time-resolved MSE
    - Time-resolved correlation
    - Residual timecourse (mean and std)
    """
    model_rates = predictions['model_rates']  # [trials, time, n_recorded]
    targets = predictions['targets']

    n_trials, n_time, n_neurons = targets.shape

    # Trial-averaged PSTHs
    model_psth = model_rates.mean(axis=0)  # [time, n_recorded]
    target_psth = targets.mean(axis=0)

    # Time-resolved MSE (averaged over neurons)
    mse_per_time = np.mean((model_psth - target_psth) ** 2, axis=1)  # [time]

    # Time-resolved correlation (per time point across neurons)
    corr_per_time = np.zeros(n_time)
    for t in range(n_time):
        r = np.corrcoef(model_psth[t, :], target_psth[t, :])[0, 1]
        corr_per_time[t] = r if not np.isnan(r) else 0.0

    # Residual timecourse
    residuals = model_psth - target_psth  # [time, n_neurons]
    mean_residual = residuals.mean(axis=1)  # [time]
    std_residual = residuals.std(axis=1)

    # Figure 1: Time-resolved MSE
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(mse_per_time, 'k-', linewidth=1.5)
    ax.set_ylabel('MSE')
    ax.set_title('Time-Resolved Mean Squared Error')
    ax.axhline(mse_per_time.mean(), color='r', linestyle='--', alpha=0.5, label=f'Mean: {mse_per_time.mean():.4f}')
    ax.legend()

    ax = axes[1]
    ax.plot(corr_per_time, 'b-', linewidth=1.5)
    ax.set_ylabel('Correlation')
    ax.set_title('Time-Resolved Spatial Correlation (across neurons)')
    ax.axhline(corr_per_time.mean(), color='r', linestyle='--', alpha=0.5, label=f'Mean: {corr_per_time.mean():.3f}')
    ax.legend()

    ax = axes[2]
    ax.fill_between(range(n_time), mean_residual - std_residual, mean_residual + std_residual,
                    alpha=0.3, color='blue')
    ax.plot(mean_residual, 'b-', linewidth=1.5)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_ylabel('Residual (Model - Real)')
    ax.set_xlabel('Time bin')
    ax.set_title('Mean Residual Timecourse (±1 std)')

    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_error_analysis.png', dpi=150)
    plt.close()

    # Identify periods of high/low error
    early_mse = mse_per_time[:n_time//5].mean()  # First 20%
    middle_mse = mse_per_time[n_time//5:4*n_time//5].mean()  # Middle 60%
    late_mse = mse_per_time[4*n_time//5:].mean()  # Last 20%

    return {
        'mse_per_time': mse_per_time.tolist(),
        'corr_per_time': corr_per_time.tolist(),
        'mean_residual': mean_residual.tolist(),
        'early_mse': float(early_mse),
        'middle_mse': float(middle_mse),
        'late_mse': float(late_mse),
        'mean_mse': float(mse_per_time.mean()),
        'mean_spatial_corr': float(corr_per_time.mean())
    }


def analyze_population_structure(predictions: dict, neuron_info: dict, output_dir: Path):
    """
    Population structure analysis using PCA.

    Compare principal component trajectories between model and real data.
    """
    model_rates = predictions['model_rates']  # [trials, time, n_recorded]
    targets = predictions['targets']

    # Trial-averaged PSTHs
    model_psth = model_rates.mean(axis=0)  # [time, n_recorded]
    target_psth = targets.mean(axis=0)

    # Fit PCA on real data
    pca = PCA(n_components=10)
    pca.fit(target_psth)

    # Project both onto real data PCA space
    real_pcs = pca.transform(target_psth)  # [time, 10]
    model_pcs = pca.transform(model_psth)  # [time, 10]

    # Figure 1: First 3 PCs over time
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(real_pcs[:, i], 'b-', linewidth=2, label='Real', alpha=0.8)
        ax.plot(model_pcs[:, i], 'r-', linewidth=2, label='Model', alpha=0.8)
        r = np.corrcoef(real_pcs[:, i], model_pcs[:, i])[0, 1]
        ax.set_ylabel(f'PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)')
        ax.set_title(f'PC{i+1} Trajectory (r = {r:.3f})')
        if i == 0:
            ax.legend()

    axes[-1].set_xlabel('Time bin')
    plt.suptitle('Population Activity Trajectories (PCA)')
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_trajectories.png', dpi=150)
    plt.close()

    # Figure 2: 2D state space trajectory
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PC1 vs PC2
    ax = axes[0]
    ax.plot(real_pcs[:, 0], real_pcs[:, 1], 'b-', linewidth=2, label='Real', alpha=0.8)
    ax.plot(model_pcs[:, 0], model_pcs[:, 1], 'r-', linewidth=2, label='Model', alpha=0.8)
    ax.scatter(real_pcs[0, 0], real_pcs[0, 1], c='b', s=100, marker='o', zorder=5)
    ax.scatter(model_pcs[0, 0], model_pcs[0, 1], c='r', s=100, marker='o', zorder=5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.legend()
    ax.set_title('PC1 vs PC2 (circles = trial start)')

    # PC2 vs PC3
    ax = axes[1]
    ax.plot(real_pcs[:, 1], real_pcs[:, 2], 'b-', linewidth=2, label='Real', alpha=0.8)
    ax.plot(model_pcs[:, 1], model_pcs[:, 2], 'r-', linewidth=2, label='Model', alpha=0.8)
    ax.scatter(real_pcs[0, 1], real_pcs[0, 2], c='b', s=100, marker='o', zorder=5)
    ax.scatter(model_pcs[0, 1], model_pcs[0, 2], c='r', s=100, marker='o', zorder=5)
    ax.set_xlabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
    ax.legend()
    ax.set_title('PC2 vs PC3')

    plt.tight_layout()
    plt.savefig(output_dir / 'pca_state_space.png', dpi=150)
    plt.close()

    # Compute PC correlations
    pc_correlations = []
    for i in range(min(10, real_pcs.shape[1])):
        r = np.corrcoef(real_pcs[:, i], model_pcs[:, i])[0, 1]
        pc_correlations.append(float(r) if not np.isnan(r) else 0.0)

    return {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'pc_correlations': pc_correlations,
        'cumulative_variance_top3': float(pca.explained_variance_ratio_[:3].sum())
    }


def analyze_trial_variability(predictions: dict, output_dir: Path):
    """
    Trial-to-trial variability analysis.

    Compare Fano factors between model and real data.
    """
    model_rates = predictions['model_rates']  # [trials, time, n_recorded]
    targets = predictions['targets']

    n_trials, n_time, n_neurons = targets.shape

    # Compute Fano factor per neuron (averaged across time)
    # Fano factor = variance / mean

    # Real data Fano factors
    real_mean = targets.mean(axis=0)  # [time, neurons]
    real_var = targets.var(axis=0)
    real_fano = np.where(real_mean > 0.1, real_var / real_mean, np.nan)  # Avoid div by 0
    real_fano_mean = np.nanmean(real_fano, axis=0)  # [neurons]

    # Model Fano factors
    model_mean = model_rates.mean(axis=0)
    model_var = model_rates.var(axis=0)
    model_fano = np.where(model_mean > 0.1, model_var / model_mean, np.nan)
    model_fano_mean = np.nanmean(model_fano, axis=0)

    # Figure 1: Fano factor comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    valid_mask = ~np.isnan(real_fano_mean) & ~np.isnan(model_fano_mean)
    ax.scatter(real_fano_mean[valid_mask], model_fano_mean[valid_mask], alpha=0.5)
    ax.plot([0, 3], [0, 3], 'k--', alpha=0.5)
    ax.set_xlabel('Real Fano Factor')
    ax.set_ylabel('Model Fano Factor')
    ax.set_title('Per-Neuron Fano Factor Comparison')
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 3])

    ax = axes[1]
    bins = np.linspace(0, 3, 31)
    ax.hist(real_fano_mean[~np.isnan(real_fano_mean)], bins=bins, alpha=0.7, label='Real', color='blue')
    ax.hist(model_fano_mean[~np.isnan(model_fano_mean)], bins=bins, alpha=0.7, label='Model', color='red')
    ax.set_xlabel('Fano Factor')
    ax.set_ylabel('Count')
    ax.set_title(f'Fano Factor Distribution\nReal: {np.nanmean(real_fano_mean):.2f}, Model: {np.nanmean(model_fano_mean):.2f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'fano_factor_analysis.png', dpi=150)
    plt.close()

    # Figure 2: Time-resolved Fano factors
    real_fano_time = np.nanmean(real_fano, axis=1)  # [time]
    model_fano_time = np.nanmean(model_fano, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(real_fano_time, 'b-', linewidth=2, label='Real', alpha=0.8)
    ax.plot(model_fano_time, 'r-', linewidth=2, label='Model', alpha=0.8)
    ax.set_xlabel('Time bin')
    ax.set_ylabel('Mean Fano Factor')
    ax.set_title('Time-Resolved Fano Factor')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'fano_factor_time.png', dpi=150)
    plt.close()

    return {
        'real_fano_mean': float(np.nanmean(real_fano_mean)),
        'model_fano_mean': float(np.nanmean(model_fano_mean)),
        'fano_correlation': float(np.corrcoef(
            real_fano_mean[valid_mask], model_fano_mean[valid_mask]
        )[0, 1]) if valid_mask.sum() > 2 else 0.0
    }


def analyze_initial_state(model: EIRNN, neuron_info: dict, output_dir: Path):
    """
    Analyze the learned initial state h0.
    """
    if model.h0 is None:
        return {'learnable_h0': False}

    h0 = model.h0.detach().cpu().numpy()
    n_e = model.n_exc  # Use model's E count (includes hidden E neurons)
    n_i = model.n_inh  # Use model's I count

    h0_e = h0[:n_e]
    h0_i = h0[n_e:]

    # Figure: h0 distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(h0_e, bins=30, alpha=0.7, label=f'E (mean={h0_e.mean():.3f})', color='blue')
    ax.hist(h0_i, bins=30, alpha=0.7, label=f'I (mean={h0_i.mean():.3f})', color='red')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Initial State (h0)')
    ax.set_ylabel('Count')
    ax.set_title('Learned Initial State Distribution')
    ax.legend()

    ax = axes[1]
    ax.plot(h0, 'k.', alpha=0.5)
    ax.axvline(n_e, color='g', linestyle='--', label='E/I boundary')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Initial State (h0)')
    ax.set_title('Learned Initial State by Neuron')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'initial_state_h0.png', dpi=150)
    plt.close()

    return {
        'learnable_h0': True,
        'h0_e_mean': float(h0_e.mean()),
        'h0_e_std': float(h0_e.std()),
        'h0_i_mean': float(h0_i.mean()),
        'h0_i_std': float(h0_i.std()),
        'h0_min': float(h0.min()),
        'h0_max': float(h0.max()),
        'h0_positive_fraction': float((h0 > 0).mean())
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 4 Diagnostic Analysis')
    parser.add_argument('--data', type=str, default='data/rnn_export_Newton_08_15_2025_SC.mat')
    parser.add_argument('--model', type=str, default='results/phase4/attention_learnable_h0_model_best.pt')
    parser.add_argument('--output', type=str, default='results/phase4/diagnostics/')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    print("Loading model and data...")
    model, train_data, val_data, neuron_info, n_inputs, bin_size_ms = load_model_and_data(
        args.model, args.data, device
    )

    # Compute predictions on validation data
    print("Computing predictions...")
    predictions = compute_predictions(model, val_data, device, neuron_info)

    # Run analyses
    results = {}

    print("\n1. Per-neuron analysis...")
    results['per_neuron'] = analyze_per_neuron(predictions, neuron_info, output_dir)
    print(f"   Mean correlation: {results['per_neuron']['mean_correlation']:.4f}")
    print(f"   E neurons: {results['per_neuron']['e_mean_correlation']:.4f}")
    print(f"   I neurons: {results['per_neuron']['i_mean_correlation']:.4f}")

    print("\n2. Temporal error analysis...")
    results['temporal'] = analyze_temporal_error(predictions, output_dir)
    print(f"   Early MSE: {results['temporal']['early_mse']:.4f}")
    print(f"   Middle MSE: {results['temporal']['middle_mse']:.4f}")
    print(f"   Late MSE: {results['temporal']['late_mse']:.4f}")

    print("\n3. Population structure (PCA)...")
    results['population'] = analyze_population_structure(predictions, neuron_info, output_dir)
    print(f"   PC1 correlation: {results['population']['pc_correlations'][0]:.4f}")
    print(f"   PC2 correlation: {results['population']['pc_correlations'][1]:.4f}")
    print(f"   PC3 correlation: {results['population']['pc_correlations'][2]:.4f}")

    print("\n4. Trial variability (Fano factors)...")
    results['variability'] = analyze_trial_variability(predictions, output_dir)
    print(f"   Real Fano: {results['variability']['real_fano_mean']:.3f}")
    print(f"   Model Fano: {results['variability']['model_fano_mean']:.3f}")

    print("\n5. Initial state (h0) analysis...")
    results['h0'] = analyze_initial_state(model, neuron_info, output_dir)
    if results['h0']['learnable_h0']:
        print(f"   h0 E mean: {results['h0']['h0_e_mean']:.4f}")
        print(f"   h0 I mean: {results['h0']['h0_i_mean']:.4f}")
        print(f"   h0 positive fraction: {results['h0']['h0_positive_fraction']:.3f}")

    # Save summary
    summary = {
        'model_path': args.model,
        'data_path': args.data,
        **results
    }

    with open(output_dir / 'diagnostic_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDiagnostic analysis complete!")
    print(f"Results saved to {output_dir}")
    print("\nGenerated figures:")
    for f in output_dir.glob('*.png'):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
