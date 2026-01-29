#!/usr/bin/env python3
"""
Phase 6: Cross-Session Replication for Animal 1 (Newton)

This script trains E-I RNN models on additional Newton sessions and runs
the full connectivity analysis to test replication of key findings.

Datasets:
- Newton_08_13_2025_SC
- Newton_08_14_2025_SC

Usage:
    python scripts/run_replication_animal1.py           # Full run
    python scripts/run_replication_animal1.py --test    # Quick test (5 epochs)
    python scripts/run_replication_animal1.py --dataset Newton_08_14_2025_SC  # Single dataset

Output: results/replication/{dataset_name}/
"""

import argparse
import json
import time
import signal
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
from scipy.spatial.distance import pdist
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import h5py

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EIRNN, create_model_from_data
from src.losses import compute_L_neuron, compute_L_trial, compute_L_reg
from src.data_loader import load_session, train_val_split

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results/replication"

# Datasets to process (excluding the original 08_15)
DATASETS = [
    "Newton_08_13_2025_SC",
    "Newton_08_14_2025_SC",
]

# Training configuration (same as train_final_model.py)
CONFIG = {
    # Training parameters
    'max_epochs': 2000,
    'patience': 300,
    'lr': 1e-3,
    'lr_scheduler_patience': 50,
    'lr_scheduler_factor': 0.5,
    'min_lr': 1e-5,
    'gradient_clip': 1.0,
    'seed': 42,

    # Model parameters (best Phase 5 config)
    'tau': 50.0,
    'dt': 25.0,
    'noise_scale': 0.1,
    'spectral_radius': 0.9,
    'input_embed_dim': 56,
    'input_embed_type': 'attention',
    'attention_heads': 4,
    'learnable_h0': True,
    'h0_init': 0.1,

    # Loss parameters
    'use_grad_balancing': True,
    'ltrial_scale': 0.5,
    'lambda_reg': 1e-4,
    'lambda_scale': 0.1,
    'lambda_var': 0.05,

    # Logging parameters
    'checkpoint_every': 100,
    'detailed_log_every': 100,
    'print_every': 10,
}


# ==============================================================================
# TRAINING FUNCTIONS (from train_final_model.py)
# ==============================================================================

def compute_psth_correlation(model: EIRNN, data: Dict, device: str) -> float:
    """Compute mean PSTH correlation across neurons."""
    model.eval()
    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        model_rates, _ = model(inputs)

        model_psth = model_rates.mean(dim=0).cpu().numpy()
        target_psth = targets.mean(dim=0).cpu().numpy()

        n_recorded = target_psth.shape[1]
        correlations = []
        for i in range(n_recorded):
            r = np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1]
            if not np.isnan(r):
                correlations.append(r)

        return np.mean(correlations) if correlations else 0.0


def compute_per_neuron_correlations(model: EIRNN, data: Dict, device: str) -> np.ndarray:
    """Compute PSTH correlation for each neuron."""
    model.eval()
    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        model_rates, _ = model(inputs)

        model_psth = model_rates.mean(dim=0).cpu().numpy()
        target_psth = targets.mean(dim=0).cpu().numpy()

        n_recorded = target_psth.shape[1]
        correlations = np.zeros(n_recorded)
        for i in range(n_recorded):
            r = np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1]
            correlations[i] = r if not np.isnan(r) else 0.0

        return correlations


def compute_fano_factors(model: EIRNN, data: Dict, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Fano factors for model and real data per neuron."""
    model.eval()
    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        model_rates, _ = model(inputs)

        n_recorded = targets.shape[2]
        model_rates = model_rates[:, :, :n_recorded]

        model_var = model_rates.var(dim=0).cpu().numpy()
        model_mean = model_rates.mean(dim=0).cpu().numpy()
        target_var = targets.var(dim=0).cpu().numpy()
        target_mean = targets.mean(dim=0).cpu().numpy()

        model_fano = np.where(model_mean.mean(axis=0) > 0.1,
                              model_var.mean(axis=0) / (model_mean.mean(axis=0) + 1e-8), 0)
        target_fano = np.where(target_mean.mean(axis=0) > 0.1,
                               target_var.mean(axis=0) / (target_mean.mean(axis=0) + 1e-8), 0)

        return model_fano, target_fano


def get_model_outputs(model: EIRNN, data: Dict, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get model outputs for saving."""
    model.eval()
    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        model_rates, _ = model(inputs)

        model_rates_np = model_rates.cpu().numpy()
        target_rates_np = targets.cpu().numpy()

        n_recorded = target_rates_np.shape[2]
        model_rates_np = model_rates_np[:, :, :n_recorded]

        model_psth = model_rates_np.mean(axis=0)
        target_psth = target_rates_np.mean(axis=0)

        return model_rates_np, target_rates_np, model_psth, target_psth


# ==============================================================================
# CONNECTIVITY ANALYSIS FUNCTIONS (from analyze_connectivity.py)
# ==============================================================================

def compute_gini(x):
    """Compute Gini coefficient of array x."""
    x = np.abs(x)
    x = np.sort(x)
    n = len(x)
    if x.sum() == 0:
        return 0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))


def compute_entropy(x):
    """Compute normalized entropy of distribution."""
    x = np.abs(x)
    if x.sum() == 0:
        return 0
    p = x / x.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(x))
    return entropy / max_entropy if max_entropy > 0 else 0


def compute_factor_selectivity(val_target_rates, trial_factors, n_recorded_e, n_i):
    """Compute factor selectivity for each neuron."""
    rates = val_target_rates
    mean_rates = rates.mean(axis=1)
    n_recorded = mean_rates.shape[1]

    factor_names = ['reward', 'location', 'identity', 'salience']
    selectivity = np.zeros((n_recorded, 4))
    selectivity_pval = np.zeros((n_recorded, 4))

    for i in range(n_recorded):
        y = mean_rates[:, i]
        X = np.column_stack([
            trial_factors['reward'],
            trial_factors['location'],
            trial_factors['identity'],
            trial_factors['salience'],
        ])
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        X_std = sm.add_constant(X_std)

        try:
            model = sm.OLS(y, X_std).fit()
            for j in range(4):
                t_stat = model.tvalues[j + 1]
                df_resid = model.df_resid
                selectivity[i, j] = t_stat**2 / (t_stat**2 + df_resid)
                selectivity_pval[i, j] = model.pvalues[j + 1]
        except:
            selectivity[i, :] = np.nan
            selectivity_pval[i, :] = np.nan

    return {
        'selectivity': selectivity,
        'selectivity_pval': selectivity_pval,
        'factor_names': factor_names,
        'e_selectivity': selectivity[:n_recorded_e, :],
        'i_selectivity': selectivity[n_recorded_e:, :],
    }


def analyze_ie_connectivity(W_rec, n_e, n_i):
    """Analyze I→E connectivity structure."""
    W_IE = W_rec[:n_e, n_e:n_e+n_i]

    i_stats = []
    for i in range(n_i):
        weights = W_IE[:, i]
        abs_weights = np.abs(weights)
        stats_dict = {
            'i_neuron': i + n_e,
            'mean_abs_weight': abs_weights.mean(),
            'std_weight': weights.std(),
            'max_abs_weight': abs_weights.max(),
            'min_weight': weights.min(),
            'sparsity': (abs_weights < 0.01).mean(),
            'n_strong_targets': (abs_weights > abs_weights.max() * 0.5).sum(),
            'gini': compute_gini(abs_weights),
        }
        i_stats.append(stats_dict)

    return {
        'W_IE': W_IE,
        'i_stats': pd.DataFrame(i_stats),
    }


def cluster_i_neurons(W_IE, n_i):
    """Cluster I neurons by connectivity patterns."""
    X = W_IE.T
    X_std = StandardScaler().fit_transform(X)

    linkage_matrix = linkage(X_std, method='ward')

    kmeans_results = {}
    silhouette_scores = {}
    for k in [2, 3, 4]:
        if k < n_i:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_std)
            score = silhouette_score(X_std, labels) if len(np.unique(labels)) > 1 else 0
            kmeans_results[k] = {'labels': labels, 'centers': kmeans.cluster_centers_}
            silhouette_scores[k] = score

    best_k = max(silhouette_scores, key=silhouette_scores.get) if silhouette_scores else 2

    # PCA for visualization
    pca = PCA(n_components=min(2, n_i))
    X_pca = pca.fit_transform(X_std)

    # Shuffle control
    n_shuffles = 1000
    shuffled_silhouettes = []
    for _ in range(n_shuffles):
        X_shuffled = np.array([np.random.permutation(row) for row in X_std])
        if best_k < n_i:
            kmeans_shuf = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels_shuf = kmeans_shuf.fit_predict(X_shuffled)
            if len(np.unique(labels_shuf)) > 1:
                score_shuf = silhouette_score(X_shuffled, labels_shuf)
                shuffled_silhouettes.append(score_shuf)

    p_value = (np.array(shuffled_silhouettes) >= silhouette_scores.get(best_k, 0)).mean() if shuffled_silhouettes else 1.0

    return {
        'linkage_matrix': linkage_matrix,
        'kmeans_results': kmeans_results,
        'silhouette_scores': silhouette_scores,
        'best_k': best_k,
        'cluster_labels': kmeans_results.get(best_k, {}).get('labels', np.zeros(n_i)),
        'X_pca': X_pca,
        'pca': pca,
        'X_std': X_std,
        'shuffled_silhouettes': shuffled_silhouettes,
        'permutation_pval': p_value,
    }


def analyze_inhibition_specificity(W_IE, n_e, n_i):
    """Analyze local vs distributed inhibition."""
    gini_values = []
    entropy_values = []
    n_effective = []

    for i in range(n_i):
        weights = np.abs(W_IE[:, i])
        gini_values.append(compute_gini(weights))
        entropy_values.append(compute_entropy(weights))

        p = weights / (weights.sum() + 1e-8)
        n_eff = 1 / (np.sum(p**2) + 1e-8)
        n_effective.append(n_eff)

    gini_values = np.array(gini_values)
    entropy_values = np.array(entropy_values)
    n_effective = np.array(n_effective)

    # Shuffle control
    n_shuffles = 1000
    shuffled_gini = []
    shuffled_entropy = []

    for _ in range(n_shuffles):
        W_flat = W_IE.flatten()
        np.random.shuffle(W_flat)
        W_shuffled = W_flat.reshape(W_IE.shape)

        gini_shuf = [compute_gini(np.abs(W_shuffled[:, i])) for i in range(n_i)]
        entropy_shuf = [compute_entropy(np.abs(W_shuffled[:, i])) for i in range(n_i)]

        shuffled_gini.append(np.mean(gini_shuf))
        shuffled_entropy.append(np.mean(entropy_shuf))

    p_gini = (np.array(shuffled_gini) >= gini_values.mean()).mean()
    p_entropy = (np.array(shuffled_entropy) <= entropy_values.mean()).mean()

    return {
        'gini_values': gini_values,
        'entropy_values': entropy_values,
        'n_effective': n_effective,
        'shuffled_gini': shuffled_gini,
        'shuffled_entropy': shuffled_entropy,
        'p_gini': p_gini,
        'p_entropy': p_entropy,
    }


def analyze_weight_selectivity_correlation(W_IE, e_selectivity, n_i, factor_names):
    """Correlate I→E weights with E neuron selectivity."""
    n_recorded_e = e_selectivity.shape[0]
    weight_selectivity_corr = np.zeros((n_i, 4))
    weight_selectivity_pval = np.zeros((n_i, 4))

    for i in range(n_i):
        weights = np.abs(W_IE[:n_recorded_e, i])
        for j in range(4):
            sel = e_selectivity[:, j]
            valid = ~np.isnan(sel)
            if valid.sum() > 5:
                r, p = stats.spearmanr(weights[valid], sel[valid])
                weight_selectivity_corr[i, j] = r
                weight_selectivity_pval[i, j] = p

    return {
        'weight_selectivity_corr': weight_selectivity_corr,
        'weight_selectivity_pval': weight_selectivity_pval,
    }


def analyze_h0_patterns(h0, n_e, n_recorded_e, n_i):
    """Analyze learned h0 patterns."""
    h0 = h0.flatten()
    h0_E_rec = h0[:n_recorded_e]
    h0_I_rec = h0[n_e:n_e+n_i]

    stat, pval = stats.mannwhitneyu(h0_E_rec, h0_I_rec, alternative='two-sided')

    return {
        'h0_E': h0_E_rec,
        'h0_I': h0_I_rec,
        'h0_ei_pval': pval,
        'h0_E_mean': h0_E_rec.mean(),
        'h0_I_mean': h0_I_rec.mean(),
    }


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def create_training_figures(history, output_dir, neuron_info, per_neuron_corr):
    """Create training visualization figures."""
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = np.arange(1, len(history['train_loss']) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', alpha=0.7, label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-', alpha=0.7, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    ax = axes[0, 1]
    ax.plot(epochs, history['val_correlation'], 'g-', linewidth=2)
    best_epoch = np.argmax(history['val_correlation']) + 1
    best_corr = max(history['val_correlation'])
    ax.axvline(x=best_epoch, color='orange', linestyle='--', alpha=0.7,
               label=f'Best: {best_corr:.4f} @ epoch {best_epoch}')
    ax.scatter([best_epoch], [best_corr], color='orange', s=100, zorder=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSTH Correlation')
    ax.set_title('Validation PSTH Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history['learning_rate'], 'purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, history['L_neuron'], 'b-', alpha=0.7, label='L_neuron')
    ax.plot(epochs, history['L_trial'], 'r-', alpha=0.7, label='L_trial')
    ax.plot(epochs, history['L_reg'], 'g-', alpha=0.7, label='L_reg')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Component')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(figures_dir / 'training_curves.png', dpi=150)
    plt.close()

    # Per-neuron correlation histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    n_exc = neuron_info['n_exc']
    n_recorded = len(per_neuron_corr)
    e_indices = np.arange(min(n_exc, n_recorded))
    i_indices = np.arange(n_exc, n_recorded) if n_recorded > n_exc else np.array([])

    bins = np.linspace(-0.2, 1.0, 25)
    ax.hist(per_neuron_corr[e_indices], bins=bins, alpha=0.6, color='blue',
            label=f'E neurons (n={len(e_indices)}, mean={per_neuron_corr[e_indices].mean():.3f})')
    if len(i_indices) > 0:
        ax.hist(per_neuron_corr[i_indices], bins=bins, alpha=0.6, color='red',
                label=f'I neurons (n={len(i_indices)}, mean={per_neuron_corr[i_indices].mean():.3f})')
    ax.axvline(x=per_neuron_corr.mean(), color='black', linestyle='--', linewidth=2,
               label=f'Overall mean: {per_neuron_corr.mean():.3f}')
    ax.set_xlabel('PSTH Correlation')
    ax.set_ylabel('Count')
    ax.set_title('Per-Neuron PSTH Correlation Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'per_neuron_correlation_histogram.png', dpi=150)
    plt.close()


def create_connectivity_figures(output_dir, selectivity_results, ie_results,
                                 cluster_results, specificity_results, h0_results,
                                 n_recorded_e, n_i, n_e):
    """Create connectivity analysis figures."""
    conn_dir = output_dir / 'connectivity_analysis'
    conn_dir.mkdir(exist_ok=True)

    factor_names = selectivity_results['factor_names']

    # Figure 1: Factor selectivity E vs I comparison
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    e_sel = selectivity_results['e_selectivity']
    i_sel = selectivity_results['i_selectivity']

    for j, (ax, factor) in enumerate(zip(axes, factor_names)):
        e_data = e_sel[:, j][~np.isnan(e_sel[:, j])]
        i_data = i_sel[:, j][~np.isnan(i_sel[:, j])]

        if len(e_data) > 0 and len(i_data) > 0:
            parts = ax.violinplot([e_data, i_data], positions=[0, 1], showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(['tab:blue', 'tab:red'][i])
                pc.set_alpha(0.7)

            stat, pval = stats.mannwhitneyu(e_data, i_data, alternative='two-sided')
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            ax.text(0.5, ax.get_ylim()[1] * 0.95, f'p={pval:.3f} ({sig})', ha='center', fontsize=9)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['E', 'I'])
        ax.set_ylabel('Partial eta-squared')
        ax.set_title(f'{factor.capitalize()}')

    plt.suptitle('Factor Selectivity: E vs I Neurons')
    plt.tight_layout()
    plt.savefig(conn_dir / 'factor_selectivity_ei_comparison.png', dpi=150)
    plt.close()

    # Figure 2: I→E weight matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    W_IE = ie_results['W_IE'][:n_recorded_e, :]
    im = ax.imshow(W_IE, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(W_IE).max(), vmax=np.abs(W_IE).max())
    ax.set_xlabel('I Neuron')
    ax.set_ylabel('E Neuron (target)')
    ax.set_title('I→E Weights')
    plt.colorbar(im, ax=ax, label='Weight')
    plt.tight_layout()
    plt.savefig(conn_dir / 'ie_weight_matrix.png', dpi=150)
    plt.close()

    # Figure 3: I neuron clustering
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    if n_i > 1:
        dendrogram(cluster_results['linkage_matrix'], ax=ax,
                   labels=[f'I{i}' for i in range(n_i)], leaf_rotation=90)
    ax.set_ylabel('Distance')
    ax.set_title('Hierarchical Clustering of I Neurons')

    ax = axes[1]
    X_pca = cluster_results['X_pca']
    cluster_labels = cluster_results['cluster_labels']
    if len(X_pca) > 0 and X_pca.shape[1] >= 2:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
                            cmap='Set1', s=100, edgecolors='black')
        for i in range(min(n_i, len(X_pca))):
            ax.annotate(f'I{i}', (X_pca[i, 0], X_pca[i, 1]), fontsize=9, ha='center', va='bottom')
    ax.set_xlabel(f'PC1')
    ax.set_ylabel(f'PC2')
    ax.set_title(f'PCA of I Neuron Connectivity (k={cluster_results["best_k"]})')

    plt.tight_layout()
    plt.savefig(conn_dir / 'i_neuron_clustering.png', dpi=150)
    plt.close()

    # Figure 4: Inhibition specificity
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    if len(specificity_results['shuffled_gini']) > 0:
        ax.hist(specificity_results['shuffled_gini'], bins=30, alpha=0.5, label='Shuffled', density=True)
    ax.axvline(specificity_results['gini_values'].mean(), color='red', linewidth=2,
               label=f'Real (mean={specificity_results["gini_values"].mean():.3f})')
    ax.set_xlabel('Gini Coefficient')
    ax.set_ylabel('Density')
    ax.set_title(f'Gini Coefficient (p={specificity_results["p_gini"]:.3f})')
    ax.legend()

    ax = axes[1]
    if len(specificity_results['shuffled_entropy']) > 0:
        ax.hist(specificity_results['shuffled_entropy'], bins=30, alpha=0.5, label='Shuffled', density=True)
    ax.axvline(specificity_results['entropy_values'].mean(), color='red', linewidth=2,
               label=f'Real (mean={specificity_results["entropy_values"].mean():.3f})')
    ax.set_xlabel('Normalized Entropy')
    ax.set_ylabel('Density')
    ax.set_title(f'Normalized Entropy (p={specificity_results["p_entropy"]:.3f})')
    ax.legend()

    ax = axes[2]
    ax.bar(range(n_i), specificity_results['n_effective'])
    ax.axhline(n_e, color='gray', linestyle='--', label=f'Max (N_E={n_e})')
    ax.set_xlabel('I Neuron')
    ax.set_ylabel('Effective Targets')
    ax.set_title(f'Effective Number of E Targets (mean={specificity_results["n_effective"].mean():.1f})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(conn_dir / 'inhibition_specificity.png', dpi=150)
    plt.close()

    # Figure 5: h0 comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    h0_data = [h0_results['h0_E'], h0_results['h0_I']]
    parts = ax.violinplot(h0_data, positions=[0, 1], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['tab:blue', 'tab:red'][i])
        pc.set_alpha(0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['E', 'I'])
    ax.set_ylabel('h0')
    ax.set_title(f'Initial State (h0) by Cell Type (p={h0_results["h0_ei_pval"]:.4f})')
    plt.tight_layout()
    plt.savefig(conn_dir / 'h0_vs_selectivity.png', dpi=150)
    plt.close()

    # Summary figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Panel A: I→E weight matrix
    ax = fig.add_subplot(gs[0, 0])
    W_IE = ie_results['W_IE'][:n_recorded_e, :]
    im = ax.imshow(W_IE, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(W_IE).max(), vmax=np.abs(W_IE).max())
    ax.set_xlabel('I Neuron')
    ax.set_ylabel('E Neuron')
    ax.set_title('A. I→E Connectivity')

    # Panel B: PCA clustering
    ax = fig.add_subplot(gs[0, 1])
    if len(X_pca) > 0 and X_pca.shape[1] >= 2:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='Set1', s=100, edgecolors='black')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('B. I Neuron Clusters')

    # Panel C: Gini distribution
    ax = fig.add_subplot(gs[0, 2])
    if len(specificity_results['shuffled_gini']) > 0:
        ax.hist(specificity_results['shuffled_gini'], bins=20, alpha=0.5, label='Shuffled', density=True)
    ax.axvline(specificity_results['gini_values'].mean(), color='red', linewidth=2, label='Real')
    ax.set_xlabel('Gini')
    ax.set_title('C. Inhibition Specificity')
    ax.legend(fontsize=8)

    # Panel D: Factor selectivity E vs I
    ax = fig.add_subplot(gs[1, :])
    e_means = [np.nanmean(selectivity_results['e_selectivity'][:, j]) for j in range(4)]
    i_means = [np.nanmean(selectivity_results['i_selectivity'][:, j]) for j in range(4)]
    e_sems = [stats.sem(selectivity_results['e_selectivity'][:, j], nan_policy='omit') for j in range(4)]
    i_sems = [stats.sem(selectivity_results['i_selectivity'][:, j], nan_policy='omit') for j in range(4)]

    x = np.arange(4)
    width = 0.35
    ax.bar(x - width/2, e_means, width, yerr=e_sems, label='E', color='tab:blue', alpha=0.8)
    ax.bar(x + width/2, i_means, width, yerr=i_sems, label='I', color='tab:red', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(factor_names)
    ax.set_ylabel('Partial eta-squared')
    ax.set_title('D. Factor Selectivity by Cell Type')
    ax.legend()

    # Panel E: h0 E vs I
    ax = fig.add_subplot(gs[2, 0])
    ax.bar([0, 1], [h0_results['h0_E'].mean(), h0_results['h0_I'].mean()],
           yerr=[stats.sem(h0_results['h0_E']), stats.sem(h0_results['h0_I'])],
           color=['tab:blue', 'tab:red'], alpha=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['E', 'I'])
    ax.set_ylabel('h0')
    ax.set_title('E. Initial State by Cell Type')

    # Panel F: Effective targets per I neuron
    ax = fig.add_subplot(gs[2, 1:])
    ax.bar(range(n_i), specificity_results['n_effective'], color='gray', alpha=0.8)
    ax.axhline(n_e, color='red', linestyle='--', label=f'Max={n_e}')
    ax.set_xlabel('I Neuron')
    ax.set_ylabel('Effective Targets')
    ax.set_title(f'F. Effective Targets (mean={specificity_results["n_effective"].mean():.1f})')
    ax.legend()

    plt.savefig(conn_dir / 'summary_figure.png', dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN TRAINING AND ANALYSIS FUNCTION
# ==============================================================================

def train_and_analyze(dataset_name: str, config: Dict, test_mode: bool = False):
    """Train model and run connectivity analysis for a single dataset."""

    print("\n" + "=" * 80)
    print(f"PROCESSING: {dataset_name}")
    print("=" * 80)

    # Override config for test mode
    if test_mode:
        config = config.copy()
        config['max_epochs'] = 5
        config['patience'] = 3
        config['checkpoint_every'] = 2
        config['detailed_log_every'] = 2
        config['print_every'] = 1

    # Setup paths
    data_path = DATA_DIR / f"rnn_export_{dataset_name}.mat"
    output_dir = RESULTS_DIR / dataset_name

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return None

    # Setup
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'weights').mkdir(exist_ok=True)
    (output_dir / 'outputs').mkdir(exist_ok=True)
    (output_dir / 'metrics').mkdir(exist_ok=True)
    (output_dir / 'population').mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'connectivity_analysis').mkdir(exist_ok=True)

    # Load data
    print(f"\nLoading data from {data_path}...")
    dataset = load_session(str(data_path))
    train_idx, val_idx = train_val_split(dataset, val_fraction=0.2, seed=config['seed'])

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

    print(f"Train trials: {len(train_idx)}, Val trials: {len(val_idx)}")
    print(f"Neurons: {neuron_info['n_total']} ({neuron_info['n_exc']} E, {neuron_info['n_inh']} I)")
    print(f"Input dim: {n_inputs}, Time bins: {dataset.n_bins}, Bin size: {bin_size_ms}ms")

    # Network structure
    n_e = neuron_info['n_exc']
    n_i = neuron_info['n_inh']
    n_recorded = neuron_info['n_total']  # All neurons in this dataset are recorded
    n_recorded_e = n_e  # E neurons count

    # Create model
    print("\nCreating model...")
    model = create_model_from_data(
        n_classic=n_e,
        n_interneuron=n_i,
        n_inputs=n_inputs,
        enforce_ratio=True,
        dt=config['dt'],
        tau=config['tau'],
        noise_scale=config['noise_scale'],
        spectral_radius=config['spectral_radius'],
        input_embed_dim=config['input_embed_dim'],
        input_embed_type=config['input_embed_type'],
        attention_heads=config['attention_heads'],
        learnable_h0=config['learnable_h0'],
        h0_init=config['h0_init'],
        device=device
    )

    model.verify_constraints()

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=config['lr_scheduler_factor'],
        patience=config['lr_scheduler_patience'],
        min_lr=config['min_lr']
    )

    # Loss configuration
    loss_ema = {'L_neuron': 1.0, 'L_trial': 1.0}
    ema_decay = 0.99

    # Training state
    best_val_corr = float('-inf')
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_state = None

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_correlation': [],
        'learning_rate': [],
        'epoch_time': [],
        'L_neuron': [],
        'L_trial': [],
        'L_reg': [],
    }

    # Training loop
    print(f"\nStarting training for up to {config['max_epochs']} epochs...")
    start_time = time.time()

    pbar = tqdm(range(config['max_epochs']), desc=f'Training {dataset_name}')
    for epoch in pbar:
        epoch_start = time.time()

        # Training step
        model.train()
        inputs = train_data['inputs'].to(device)
        targets = train_data['targets'].to(device)
        mask = train_data['mask'].to(device)

        optimizer.zero_grad()
        model_rates, _ = model(inputs)
        n_rec = targets.shape[2]
        model_rates_recorded = model_rates[:, :, :n_rec]

        # Compute losses
        L_neuron = compute_L_neuron(
            model_rates_recorded, targets, bin_size_ms,
            mask=mask, lambda_scale=config['lambda_scale'], lambda_var=config['lambda_var']
        )

        L_trial_raw = compute_L_trial(
            model_rates_recorded, targets, bin_size_ms,
            mask=mask, sinkhorn_iters=20, sinkhorn_epsilon=0.1
        )

        L_reg = compute_L_reg(model, config['lambda_reg'])

        # Gradient balancing
        if config['use_grad_balancing']:
            with torch.no_grad():
                loss_ema['L_neuron'] = ema_decay * loss_ema['L_neuron'] + (1 - ema_decay) * L_neuron.item()
                loss_ema['L_trial'] = ema_decay * loss_ema['L_trial'] + (1 - ema_decay) * L_trial_raw.item()

            L_neuron_norm = L_neuron / (loss_ema['L_neuron'] + 1e-8)
            L_trial_norm = L_trial_raw / (loss_ema['L_trial'] + 1e-8)
            loss = L_neuron_norm + config['ltrial_scale'] * L_trial_norm + L_reg
        else:
            loss = L_neuron + config['ltrial_scale'] * L_trial_raw + L_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
        optimizer.step()

        train_loss = loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            inputs_val = val_data['inputs'].to(device)
            targets_val = val_data['targets'].to(device)
            mask_val = val_data['mask'].to(device)

            model_rates_val, _ = model(inputs_val)
            model_rates_val_recorded = model_rates_val[:, :, :n_rec]

            L_neuron_val = compute_L_neuron(
                model_rates_val_recorded, targets_val, bin_size_ms,
                mask=mask_val, lambda_scale=config['lambda_scale'], lambda_var=config['lambda_var']
            )
            val_loss = L_neuron_val.item()

        val_corr = compute_psth_correlation(model, val_data, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_correlation'].append(val_corr)
        history['learning_rate'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        history['L_neuron'].append(L_neuron.item())
        history['L_trial'].append(L_trial_raw.item())
        history['L_reg'].append(L_reg.item())

        pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val_corr': f'{val_corr:.4f}',
            'best': f'{best_val_corr:.4f}',
            'lr': f'{current_lr:.1e}'
        })

        # Track best model
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        # Checkpoint
        if (epoch + 1) % config['checkpoint_every'] == 0:
            checkpoint_path = output_dir / 'checkpoints' / f'model_epoch{epoch+1}.pt'
            torch.save(model.state_dict(), checkpoint_path)

    total_time = time.time() - start_time
    epochs_trained = epoch + 1

    print(f"\nTraining complete: best_val_corr={best_val_corr:.4f} at epoch {best_epoch}")
    print(f"Total time: {str(timedelta(seconds=int(total_time)))}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # =========================================================================
    # SAVE TRAINING OUTPUTS
    # =========================================================================

    print("\nSaving training outputs...")

    # Model checkpoints
    torch.save(best_model_state, output_dir / 'model_best.pt')
    torch.save(model.state_dict(), output_dir / 'model_final.pt')

    # Weight matrices
    W_rec = model.W_rec.detach().cpu().numpy()
    W_in = model.W_in.detach().cpu().numpy()
    W_out = model.W_out.detach().cpu().numpy()

    np.save(output_dir / 'weights' / 'W_rec.npy', W_rec)
    np.save(output_dir / 'weights' / 'W_in.npy', W_in)
    np.save(output_dir / 'weights' / 'W_out.npy', W_out)

    if model.h0 is not None:
        h0 = model.h0.detach().cpu().numpy()
        np.save(output_dir / 'weights' / 'h0.npy', h0)

    if model.input_embed is not None and model.input_embed.embed_type == 'attention':
        embed = model.input_embed
        attention_weights = {
            'qkv_proj': embed.qkv_proj.weight.detach().cpu().numpy(),
            'out_proj': embed.out_proj.weight.detach().cpu().numpy(),
        }
        np.save(output_dir / 'weights' / 'attention_weights.npy', attention_weights)

    # E/I masks
    n_total = model.n_total
    E_mask = np.zeros(n_total, dtype=bool)
    E_mask[:n_e] = True
    I_mask = ~E_mask
    np.save(output_dir / 'weights' / 'E_mask.npy', E_mask)
    np.save(output_dir / 'weights' / 'I_mask.npy', I_mask)

    # Model outputs
    model_rates, target_rates, model_psth, target_psth = get_model_outputs(model, val_data, device)

    np.save(output_dir / 'outputs' / 'val_model_rates.npy', model_rates)
    np.save(output_dir / 'outputs' / 'val_target_rates.npy', target_rates)
    np.save(output_dir / 'outputs' / 'val_model_psth.npy', model_psth)
    np.save(output_dir / 'outputs' / 'val_target_psth.npy', target_psth)

    val_trial_conditions = dataset.trial_reward[val_idx]
    np.save(output_dir / 'outputs' / 'val_trial_conditions.npy', val_trial_conditions)

    # Per-neuron metrics
    per_neuron_corr = compute_per_neuron_correlations(model, val_data, device)
    fano_model, fano_target = compute_fano_factors(model, val_data, device)

    np.save(output_dir / 'metrics' / 'per_neuron_correlation.npy', per_neuron_corr)
    np.save(output_dir / 'metrics' / 'per_neuron_mean_rate_model.npy', model_psth.mean(axis=0))
    np.save(output_dir / 'metrics' / 'per_neuron_mean_rate_target.npy', target_psth.mean(axis=0))
    np.save(output_dir / 'metrics' / 'per_neuron_fano_model.npy', fano_model)
    np.save(output_dir / 'metrics' / 'per_neuron_fano_target.npy', fano_target)

    neuron_ei_labels = np.zeros(n_recorded, dtype=int)
    neuron_ei_labels[n_recorded_e:] = 1
    np.save(output_dir / 'metrics' / 'neuron_ei_labels.npy', neuron_ei_labels)

    # PCA
    n_pcs = min(10, n_recorded)
    pca = PCA(n_components=n_pcs)
    pca_real = pca.fit_transform(target_psth)
    pca_model = pca.transform(model_psth)

    np.save(output_dir / 'population' / 'pca_real.npy', pca_real)
    np.save(output_dir / 'population' / 'pca_model.npy', pca_model)
    np.save(output_dir / 'population' / 'pca_explained_variance.npy', pca.explained_variance_ratio_)

    # Training history
    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training figures
    create_training_figures(history, output_dir, neuron_info, per_neuron_corr)

    # =========================================================================
    # CONNECTIVITY ANALYSIS
    # =========================================================================

    print("\nRunning connectivity analysis...")

    # Load trial factors
    with h5py.File(data_path, 'r') as f:
        n_trials = f['trial_reward'].shape[1]
        n_val = int(0.2 * n_trials)
        val_start = n_trials - n_val

        trial_factors = {
            'reward': f['trial_reward'][0, val_start:].astype(int),
            'location': f['trial_location'][0, val_start:].astype(int),
            'identity': f['trial_identity'][0, val_start:].astype(int),
            'salience': f['trial_salience'][0, val_start:].astype(int),
        }

    # Factor selectivity
    print("  Computing factor selectivity...")
    selectivity_results = compute_factor_selectivity(
        target_rates, trial_factors, n_recorded_e, n_i
    )

    # I→E connectivity
    print("  Analyzing I→E connectivity...")
    ie_results = analyze_ie_connectivity(W_rec, n_e, n_i)

    # I neuron clustering
    print("  Clustering I neurons...")
    cluster_results = cluster_i_neurons(ie_results['W_IE'], n_i)

    # Inhibition specificity
    print("  Analyzing inhibition specificity...")
    specificity_results = analyze_inhibition_specificity(ie_results['W_IE'], n_e, n_i)

    # Weight-selectivity correlation
    print("  Computing weight-selectivity correlations...")
    weight_sel_results = analyze_weight_selectivity_correlation(
        ie_results['W_IE'], selectivity_results['e_selectivity'], n_i,
        selectivity_results['factor_names']
    )

    # h0 analysis
    print("  Analyzing h0 patterns...")
    h0_results = analyze_h0_patterns(h0, n_e, n_recorded_e, n_i)

    # Create connectivity figures
    print("  Creating connectivity figures...")
    create_connectivity_figures(
        output_dir, selectivity_results, ie_results, cluster_results,
        specificity_results, h0_results, n_recorded_e, n_i, n_e
    )

    # Save connectivity results
    conn_dir = output_dir / 'connectivity_analysis'

    selectivity_df = pd.DataFrame(
        selectivity_results['selectivity'],
        columns=selectivity_results['factor_names']
    )
    selectivity_df['neuron_type'] = ['E'] * n_recorded_e + ['I'] * n_i
    selectivity_df.to_csv(conn_dir / 'factor_selectivity.csv', index=False)

    cluster_df = pd.DataFrame({
        'i_neuron': range(n_i),
        'cluster': cluster_results['cluster_labels'],
        'gini': specificity_results['gini_values'],
        'entropy': specificity_results['entropy_values'],
        'n_effective': specificity_results['n_effective'],
    })
    cluster_df.to_csv(conn_dir / 'i_neuron_clusters.csv', index=False)

    summary = {
        'dataset': dataset_name,
        'n_e_neurons': n_e,
        'n_i_neurons': n_i,
        'n_recorded_e': n_recorded_e,
        'n_recorded': n_recorded,
        'n_train_trials': len(train_idx),
        'n_val_trials': len(val_idx),
        'best_val_corr': best_val_corr,
        'best_epoch': best_epoch,
        'epochs_trained': epochs_trained,
        'training_time_sec': total_time,
        'best_k_clusters': cluster_results['best_k'],
        'cluster_silhouette': cluster_results['silhouette_scores'].get(cluster_results['best_k'], 0),
        'cluster_permutation_p': cluster_results['permutation_pval'],
        'mean_gini': specificity_results['gini_values'].mean(),
        'gini_vs_shuffle_p': specificity_results['p_gini'],
        'mean_entropy': specificity_results['entropy_values'].mean(),
        'entropy_vs_shuffle_p': specificity_results['p_entropy'],
        'mean_effective_targets': specificity_results['n_effective'].mean(),
        'h0_E_mean': h0_results['h0_E_mean'],
        'h0_I_mean': h0_results['h0_I_mean'],
        'h0_ei_pval': h0_results['h0_ei_pval'],
        'E_corr_mean': per_neuron_corr[:n_recorded_e].mean(),
        'I_corr_mean': per_neuron_corr[n_recorded_e:].mean() if n_i > 0 else 0,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(conn_dir / 'analysis_summary.csv', index=False)

    # =========================================================================
    # GENERATE REPORT
    # =========================================================================

    print("\nGenerating connectivity report...")

    report = f"""# Connectivity Analysis Report: {dataset_name}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary

| Metric | Value |
|--------|-------|
| E neurons | {n_e} ({n_recorded_e} recorded) |
| I neurons | {n_i} |
| Train trials | {len(train_idx)} |
| Val trials | {len(val_idx)} |

## Model Performance

| Metric | Value |
|--------|-------|
| Best validation correlation | {best_val_corr:.4f} |
| Best epoch | {best_epoch} |
| E neuron mean correlation | {per_neuron_corr[:n_recorded_e].mean():.4f} |
| I neuron mean correlation | {per_neuron_corr[n_recorded_e:].mean():.4f} |
| Training time | {str(timedelta(seconds=int(total_time)))} |

## Connectivity Analysis Results

### Inhibition Specificity

| Metric | Value | vs Shuffle p |
|--------|-------|--------------|
| Mean Gini coefficient | {specificity_results['gini_values'].mean():.4f} | {specificity_results['p_gini']:.4f} |
| Mean entropy | {specificity_results['entropy_values'].mean():.4f} | {specificity_results['p_entropy']:.4f} |
| Mean effective targets | {specificity_results['n_effective'].mean():.1f}/{n_e} ({100*specificity_results['n_effective'].mean()/n_e:.1f}%) | - |

### I Neuron Clustering

| Metric | Value |
|--------|-------|
| Best k | {cluster_results['best_k']} |
| Silhouette score | {cluster_results['silhouette_scores'].get(cluster_results['best_k'], 0):.4f} |
| Permutation p-value | {cluster_results['permutation_pval']:.4f} |

### h0 Analysis

| Metric | Value |
|--------|-------|
| h0 E mean | {h0_results['h0_E_mean']:.4f} |
| h0 I mean | {h0_results['h0_I_mean']:.4f} |
| E vs I p-value | {h0_results['h0_ei_pval']:.4f} |

### Key Finding Replication

| Finding | Original (08_15) | This Session | Replicated? |
|---------|------------------|--------------|-------------|
| Global inhibition (Gini p > 0.05) | p=0.712 | p={specificity_results['p_gini']:.3f} | {'Yes' if specificity_results['p_gini'] > 0.05 else 'No'} |
| h0 I > E (p < 0.05) | p=0.002 | p={h0_results['h0_ei_pval']:.3f} | {'Yes' if h0_results['h0_ei_pval'] < 0.05 and h0_results['h0_I_mean'] > h0_results['h0_E_mean'] else 'No'} |

---

*Report generated by run_replication_animal1.py*
"""

    report_path = BASE_DIR / "specs" / f"{dataset_name}_connectivity_report.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Report saved to: {report_path}")

    # Save training report
    training_report = f"""# Training Report: {dataset_name}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Training Summary

| Metric | Value |
|--------|-------|
| Total epochs trained | {epochs_trained} |
| Best validation correlation | {best_val_corr:.4f} |
| Best epoch | {best_epoch} |
| Early stopping | {'Yes' if epochs_without_improvement >= config['patience'] else 'No'} |
| Training time | {str(timedelta(seconds=int(total_time)))} |

## Final Metrics

| Metric | Value |
|--------|-------|
| Overall PSTH correlation | {per_neuron_corr.mean():.4f} |
| E neuron mean correlation | {per_neuron_corr[:n_recorded_e].mean():.4f} |
| I neuron mean correlation | {per_neuron_corr[n_recorded_e:].mean():.4f} |

## Model Configuration

Same as train_final_model.py (see config.json)
"""

    with open(output_dir / 'training_report.md', 'w') as f:
        f.write(training_report)

    print(f"\n{'='*80}")
    print(f"COMPLETED: {dataset_name}")
    print(f"  Val correlation: {best_val_corr:.4f}")
    print(f"  Gini p-value: {specificity_results['p_gini']:.4f}")
    print(f"  h0 E/I p-value: {h0_results['h0_ei_pval']:.4f}")
    print(f"  Output: {output_dir}")
    print(f"{'='*80}")

    return summary


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run replication study for Animal 1')
    parser.add_argument('--test', action='store_true', help='Quick test run (5 epochs)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Process single dataset (e.g., Newton_08_14_2025_SC)')
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 6: CROSS-SESSION REPLICATION - ANIMAL 1 (NEWTON)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    datasets_to_process = [args.dataset] if args.dataset else DATASETS

    print(f"\nDatasets to process: {datasets_to_process}")
    print(f"Test mode: {args.test}")

    results = []
    for dataset in datasets_to_process:
        try:
            result = train_and_analyze(dataset, CONFIG, test_mode=args.test)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\nERROR processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save combined results
    if results:
        combined_df = pd.DataFrame(results)
        combined_path = RESULTS_DIR / 'all_sessions_summary.csv'
        combined_df.to_csv(combined_path, index=False)
        print(f"\nCombined results saved to: {combined_path}")

    print("\n" + "=" * 80)
    print("REPLICATION COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nProcessed {len(results)}/{len(datasets_to_process)} datasets successfully")

    if results:
        print("\nSummary:")
        for r in results:
            print(f"  {r['dataset']}: val_corr={r['best_val_corr']:.4f}, "
                  f"Gini_p={r['gini_vs_shuffle_p']:.3f}, h0_p={r['h0_ei_pval']:.3f}")


if __name__ == '__main__':
    main()
