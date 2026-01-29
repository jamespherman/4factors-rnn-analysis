#!/usr/bin/env python3
"""
Final Model Training Script

Trains the E-I RNN with the best Phase 5 configuration (attention embedding + learnable h0)
and produces all data needed for scientific analysis of the learned weights.

Usage:
    python scripts/train_final_model.py          # Full 2000-epoch run
    python scripts/train_final_model.py --test   # Quick 5-epoch test

Output directory: results/final_model/
"""

import argparse
import json
import time
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EIRNN, create_model_from_data
from src.losses import compute_L_neuron, compute_L_trial, compute_L_reg
from src.data_loader import load_session, train_val_split


# ==============================================================================
# CONFIGURATION
# ==============================================================================

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

    # Paths
    'data_path': 'data/rnn_export_Newton_08_15_2025_SC.mat',
    'output_dir': 'results/final_model',
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_psth_correlation(
    model: EIRNN,
    data: Dict,
    device: str
) -> float:
    """Compute mean PSTH correlation across neurons."""
    model.eval()
    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        model_rates, _ = model(inputs)

        # Trial-average
        model_psth = model_rates.mean(dim=0).cpu().numpy()
        target_psth = targets.mean(dim=0).cpu().numpy()

        # Correlation per neuron
        n_recorded = target_psth.shape[1]
        correlations = []
        for i in range(n_recorded):
            r = np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1]
            if not np.isnan(r):
                correlations.append(r)

        return np.mean(correlations) if correlations else 0.0


def compute_per_neuron_correlations(
    model: EIRNN,
    data: Dict,
    device: str
) -> np.ndarray:
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


def compute_fano_factors(
    model: EIRNN,
    data: Dict,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Fano factors for model and real data per neuron."""
    model.eval()
    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        model_rates, _ = model(inputs)

        # Limit model rates to recorded neurons only
        n_recorded = targets.shape[2]
        model_rates = model_rates[:, :, :n_recorded]

        # Variance and mean across trials
        model_var = model_rates.var(dim=0).cpu().numpy()  # [time, neurons]
        model_mean = model_rates.mean(dim=0).cpu().numpy()
        target_var = targets.var(dim=0).cpu().numpy()
        target_mean = targets.mean(dim=0).cpu().numpy()

        # Time-averaged Fano factor per neuron
        model_fano = np.where(model_mean.mean(axis=0) > 0.1,
                              model_var.mean(axis=0) / (model_mean.mean(axis=0) + 1e-8), 0)
        target_fano = np.where(target_mean.mean(axis=0) > 0.1,
                               target_var.mean(axis=0) / (target_mean.mean(axis=0) + 1e-8), 0)

        return model_fano, target_fano


def get_model_outputs(
    model: EIRNN,
    data: Dict,
    device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get model outputs for saving."""
    model.eval()
    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        model_rates, _ = model(inputs)

        model_rates_np = model_rates.cpu().numpy()
        target_rates_np = targets.cpu().numpy()

        # Limit to recorded neurons
        n_recorded = target_rates_np.shape[2]
        model_rates_np = model_rates_np[:, :, :n_recorded]

        model_psth = model_rates_np.mean(axis=0)
        target_psth = target_rates_np.mean(axis=0)

        return model_rates_np, target_rates_np, model_psth, target_psth


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_training_curves(history: Dict, save_path: Path):
    """Plot training curves: loss, correlation, learning rate."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = np.arange(1, len(history['train_loss']) + 1)

    # Train/Val Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', alpha=0.7, label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-', alpha=0.7, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Validation Correlation
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

    # Learning Rate
    ax = axes[1, 0]
    ax.plot(epochs, history['learning_rate'], 'purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Loss Components
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
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_per_neuron_correlation_histogram(
    correlations: np.ndarray,
    neuron_info: Dict,
    save_path: Path
):
    """Histogram of correlations colored by E/I."""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_exc = neuron_info['n_exc']
    n_recorded = len(correlations)
    e_indices = np.arange(min(n_exc, n_recorded))
    i_indices = np.arange(n_exc, n_recorded)

    bins = np.linspace(-0.2, 1.0, 25)

    ax.hist(correlations[e_indices], bins=bins, alpha=0.6, color='blue',
            label=f'E neurons (n={len(e_indices)}, mean={correlations[e_indices].mean():.3f})')
    ax.hist(correlations[i_indices], bins=bins, alpha=0.6, color='red',
            label=f'I neurons (n={len(i_indices)}, mean={correlations[i_indices].mean():.3f})')

    ax.axvline(x=correlations.mean(), color='black', linestyle='--', linewidth=2,
               label=f'Overall mean: {correlations.mean():.3f}')

    ax.set_xlabel('PSTH Correlation')
    ax.set_ylabel('Count')
    ax.set_title('Per-Neuron PSTH Correlation Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_per_neuron_correlation_scatter(
    correlations: np.ndarray,
    mean_rates: np.ndarray,
    neuron_info: Dict,
    save_path: Path
):
    """Scatter plot of correlation vs mean firing rate."""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_exc = neuron_info['n_exc']
    n_recorded = len(correlations)
    e_indices = np.arange(min(n_exc, n_recorded))
    i_indices = np.arange(n_exc, n_recorded)

    ax.scatter(mean_rates[e_indices], correlations[e_indices],
               c='blue', alpha=0.6, s=50, label='E neurons')
    ax.scatter(mean_rates[i_indices], correlations[i_indices],
               c='red', alpha=0.6, s=50, label='I neurons')

    ax.set_xlabel('Mean Firing Rate (sp/s)')
    ax.set_ylabel('PSTH Correlation')
    ax.set_title('PSTH Correlation vs Mean Firing Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_best_worst_neurons(
    model_psth: np.ndarray,
    target_psth: np.ndarray,
    correlations: np.ndarray,
    bin_size_ms: float,
    save_path: Path
):
    """Plot PSTHs for 4 best and 4 worst neurons."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Sort by correlation
    sorted_indices = np.argsort(correlations)
    best_4 = sorted_indices[-4:][::-1]
    worst_4 = sorted_indices[:4]

    time_axis = np.arange(model_psth.shape[0]) * bin_size_ms / 1000  # seconds

    # Best neurons
    for i, idx in enumerate(best_4):
        ax = axes[0, i]
        ax.plot(time_axis, target_psth[:, idx], 'b-', linewidth=2, label='Target')
        ax.plot(time_axis, model_psth[:, idx], 'r--', linewidth=2, label='Model')
        ax.set_title(f'Best #{i+1}: Neuron {idx} (r={correlations[idx]:.3f})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate')
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

    # Worst neurons
    for i, idx in enumerate(worst_4):
        ax = axes[1, i]
        ax.plot(time_axis, target_psth[:, idx], 'b-', linewidth=2, label='Target')
        ax.plot(time_axis, model_psth[:, idx], 'r--', linewidth=2, label='Model')
        ax.set_title(f'Worst #{i+1}: Neuron {idx} (r={correlations[idx]:.3f})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate')
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Best and Worst Fitting Neurons', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_weight_matrices(model: EIRNN, neuron_info: Dict, save_path: Path):
    """Heatmaps of W_rec and W_in sorted by E/I."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    W_rec = model.W_rec.detach().cpu().numpy()
    W_in = model.W_in.detach().cpu().numpy()

    n_exc = model.n_exc
    n_total = model.n_total

    # W_rec heatmap
    ax = axes[0]
    vmax = np.abs(W_rec).max()
    im = ax.imshow(W_rec, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.axhline(y=n_exc-0.5, color='white', linewidth=2)
    ax.axvline(x=n_exc-0.5, color='white', linewidth=2)
    ax.set_xlabel('From Neuron')
    ax.set_ylabel('To Neuron')
    ax.set_title(f'Recurrent Weights W_rec\n(E: 0-{n_exc-1}, I: {n_exc}-{n_total-1})')
    plt.colorbar(im, ax=ax, label='Weight')

    # W_in heatmap
    ax = axes[1]
    vmax = np.abs(W_in).max()
    im = ax.imshow(W_in, aspect='auto', cmap='viridis')
    ax.axhline(y=n_exc-0.5, color='white', linewidth=2)
    ax.set_xlabel('Input Dimension')
    ax.set_ylabel('Neuron')
    ax.set_title(f'Input Weights W_in\n(embed_dim={W_in.shape[1]})')
    plt.colorbar(im, ax=ax, label='Weight')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_h0_distribution(model: EIRNN, neuron_info: Dict, save_path: Path):
    """Histogram of learned h0 values colored by E/I."""
    if model.h0 is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'h0 not learnable', ha='center', va='center', fontsize=14)
        plt.savefig(save_path, dpi=150)
        plt.close()
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    h0 = model.h0.detach().cpu().numpy()
    n_exc = model.n_exc

    bins = np.linspace(h0.min() - 0.1, h0.max() + 0.1, 25)

    ax.hist(h0[:n_exc], bins=bins, alpha=0.6, color='blue',
            label=f'E neurons (mean={h0[:n_exc].mean():.3f})')
    ax.hist(h0[n_exc:], bins=bins, alpha=0.6, color='red',
            label=f'I neurons (mean={h0[n_exc:].mean():.3f})')

    ax.axvline(x=h0.mean(), color='black', linestyle='--', linewidth=2,
               label=f'Overall mean: {h0.mean():.3f}')

    ax.set_xlabel('Initial State h0')
    ax.set_ylabel('Count')
    ax.set_title('Learned Initial State Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_attention_patterns(model: EIRNN, save_path: Path):
    """Visualize learned attention weights."""
    if model.input_embed is None or model.input_embed.embed_type != 'attention':
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No attention embedding', ha='center', va='center', fontsize=14)
        plt.savefig(save_path, dpi=150)
        plt.close()
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    embed = model.input_embed
    qkv = embed.qkv_proj.weight.detach().cpu().numpy()  # [3*heads, 1]
    out_proj = embed.out_proj.weight.detach().cpu().numpy()  # [embed_dim, n_inputs*heads]

    n_heads = embed.attention_heads

    # QKV weights
    ax = axes[0]
    ax.bar(range(qkv.shape[0]), qkv.squeeze())
    ax.set_xlabel('QKV Index')
    ax.set_ylabel('Weight')
    ax.set_title(f'QKV Projection Weights (heads={n_heads})')
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.grid(True, alpha=0.3)

    # Output projection heatmap
    ax = axes[1]
    im = ax.imshow(out_proj, aspect='auto', cmap='viridis')
    ax.set_xlabel('Input Dimension x Heads')
    ax.set_ylabel('Embedding Dimension')
    ax.set_title('Output Projection Weights')
    plt.colorbar(im, ax=ax, label='Weight')

    # Weight statistics
    ax = axes[2]
    ax.hist(qkv.flatten(), bins=20, alpha=0.6, label='QKV', color='blue')
    ax.hist(out_proj.flatten(), bins=20, alpha=0.6, label='Out Proj', color='orange')
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Count')
    ax.set_title('Weight Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pca_trajectories(
    pca_real: np.ndarray,
    pca_model: np.ndarray,
    bin_size_ms: float,
    save_path: Path
):
    """Plot PC1-3 over time."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    time_axis = np.arange(pca_real.shape[0]) * bin_size_ms / 1000  # seconds

    for i, ax in enumerate(axes):
        ax.plot(time_axis, pca_real[:, i], 'b-', linewidth=2, label='Real')
        ax.plot(time_axis, pca_model[:, i], 'r--', linewidth=2, label='Model')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'PC{i+1}')
        ax.set_title(f'PC{i+1} Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pca_state_space(
    pca_real: np.ndarray,
    pca_model: np.ndarray,
    save_path: Path
):
    """PC1 vs PC2 phase plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PC1 vs PC2
    ax = axes[0]
    ax.plot(pca_real[:, 0], pca_real[:, 1], 'b-', linewidth=2, label='Real', alpha=0.7)
    ax.plot(pca_model[:, 0], pca_model[:, 1], 'r--', linewidth=2, label='Model', alpha=0.7)
    ax.scatter([pca_real[0, 0]], [pca_real[0, 1]], c='blue', s=100, marker='o', zorder=5, label='Start')
    ax.scatter([pca_real[-1, 0]], [pca_real[-1, 1]], c='blue', s=100, marker='x', zorder=5, label='End')
    ax.scatter([pca_model[0, 0]], [pca_model[0, 1]], c='red', s=100, marker='o', zorder=5)
    ax.scatter([pca_model[-1, 0]], [pca_model[-1, 1]], c='red', s=100, marker='x', zorder=5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Neural State Space: PC1 vs PC2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PC2 vs PC3
    ax = axes[1]
    ax.plot(pca_real[:, 1], pca_real[:, 2], 'b-', linewidth=2, label='Real', alpha=0.7)
    ax.plot(pca_model[:, 1], pca_model[:, 2], 'r--', linewidth=2, label='Model', alpha=0.7)
    ax.set_xlabel('PC2')
    ax.set_ylabel('PC3')
    ax.set_title('Neural State Space: PC2 vs PC3')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_fano_factor_comparison(
    fano_model: np.ndarray,
    fano_target: np.ndarray,
    neuron_info: Dict,
    save_path: Path
):
    """Model vs real Fano factors with log scale."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n_exc = neuron_info['n_exc']
    n_recorded = len(fano_model)
    # Ensure indices don't exceed array bounds
    e_indices = np.arange(min(n_exc, n_recorded))
    i_indices = np.arange(min(n_exc, n_recorded), n_recorded)

    # Scatter plot
    ax = axes[0]
    ax.scatter(fano_target[e_indices], fano_model[e_indices],
               c='blue', alpha=0.6, s=50, label='E neurons')
    ax.scatter(fano_target[i_indices], fano_model[i_indices],
               c='red', alpha=0.6, s=50, label='I neurons')

    max_val = max(fano_model.max(), fano_target.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Identity')
    ax.set_xlabel('Real Fano Factor')
    ax.set_ylabel('Model Fano Factor')
    ax.set_title('Fano Factor: Model vs Real')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Histogram comparison
    ax = axes[1]
    # Use log scale bins
    min_val = min(fano_model[fano_model > 0].min() if (fano_model > 0).any() else 0.1,
                  fano_target[fano_target > 0].min() if (fano_target > 0).any() else 0.1)
    max_val = max(fano_model.max(), fano_target.max())
    bins = np.logspace(np.log10(max(min_val, 0.01)), np.log10(max_val + 0.1), 20)

    ax.hist(fano_target[fano_target > 0], bins=bins, alpha=0.6, color='blue', label='Real')
    ax.hist(fano_model[fano_model > 0], bins=bins, alpha=0.6, color='red', label='Model')
    ax.set_xlabel('Fano Factor')
    ax.set_ylabel('Count')
    ax.set_title(f'Fano Factor Distribution\n(Real mean={fano_target.mean():.2f}, Model mean={fano_model.mean():.2f})')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train_model(config: Dict, test_mode: bool = False) -> Dict:
    """Main training function."""

    # Override config for test mode
    if test_mode:
        config = config.copy()
        config['max_epochs'] = 5
        config['patience'] = 3
        config['checkpoint_every'] = 2
        config['detailed_log_every'] = 2
        config['print_every'] = 1

    # Setup
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / 'weights').mkdir(exist_ok=True)
    (output_dir / 'outputs').mkdir(exist_ok=True)
    (output_dir / 'metrics').mkdir(exist_ok=True)
    (output_dir / 'population').mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)

    # Load data
    print(f"Loading data from {config['data_path']}...")
    dataset = load_session(config['data_path'])
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

    # Create model
    print("\nCreating model...")
    model = create_model_from_data(
        n_classic=neuron_info['n_exc'],
        n_interneuron=neuron_info['n_inh'],
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

    # Verify constraints
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
        'detailed_logs': [],
    }

    # Interrupt handler
    interrupted = False
    def signal_handler(sig, frame):
        nonlocal interrupted
        print("\n\nInterrupt received. Saving current state...")
        interrupted = True
    signal.signal(signal.SIGINT, signal_handler)

    # Training loop
    print(f"\nStarting training for up to {config['max_epochs']} epochs...")
    print(f"Patience: {config['patience']}, Checkpoint every: {config['checkpoint_every']}")
    print("=" * 80)

    start_time = time.time()

    pbar = tqdm(range(config['max_epochs']), desc='Training')
    for epoch in pbar:
        if interrupted:
            break

        epoch_start = time.time()

        # Training step
        model.train()
        inputs = train_data['inputs'].to(device)
        targets = train_data['targets'].to(device)
        mask = train_data['mask'].to(device)

        optimizer.zero_grad()
        model_rates, _ = model(inputs)
        n_recorded = targets.shape[2]
        model_rates_recorded = model_rates[:, :, :n_recorded]

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
            model_rates_val_recorded = model_rates_val[:, :, :n_recorded]

            L_neuron_val = compute_L_neuron(
                model_rates_val_recorded, targets_val, bin_size_ms,
                mask=mask_val, lambda_scale=config['lambda_scale'], lambda_var=config['lambda_var']
            )
            val_loss = L_neuron_val.item()

        val_corr = compute_psth_correlation(model, val_data, device)

        # Update scheduler
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

        # Update progress bar
        pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val_corr': f'{val_corr:.4f}',
            'best': f'{best_val_corr:.4f}',
            'lr': f'{current_lr:.1e}'
        })

        # Print every N epochs
        if (epoch + 1) % config['print_every'] == 0:
            tqdm.write(f"Epoch {epoch+1:4d} | train_loss: {train_loss:.4f} | val_corr: {val_corr:.4f} | "
                       f"lr: {current_lr:.1e} | time: {epoch_time:.1f}s")

        # Detailed log every N epochs
        if (epoch + 1) % config['detailed_log_every'] == 0:
            per_neuron_corr = compute_per_neuron_correlations(model, val_data, device)
            fano_model, fano_target = compute_fano_factors(model, val_data, device)

            e_mean = per_neuron_corr[:neuron_info['n_exc']].mean()
            i_mean = per_neuron_corr[neuron_info['n_exc']:].mean() if neuron_info['n_inh'] > 0 else 0

            h0_stats = {}
            if model.h0 is not None:
                h0 = model.h0.detach().cpu().numpy()
                h0_stats = {
                    'mean': float(h0.mean()),
                    'std': float(h0.std()),
                    'min': float(h0.min()),
                    'max': float(h0.max())
                }

            detailed_log = {
                'epoch': epoch + 1,
                'per_neuron_correlations': per_neuron_corr.tolist(),
                'E_mean_corr': float(e_mean),
                'I_mean_corr': float(i_mean),
                'fano_model_mean': float(fano_model.mean()),
                'fano_target_mean': float(fano_target.mean()),
                'h0_stats': h0_stats
            }
            history['detailed_logs'].append(detailed_log)

            tqdm.write(f"\n=== Epoch {epoch+1} Detailed Report ===")
            tqdm.write(f"  E neurons: mean_corr={e_mean:.4f} (n={neuron_info['n_exc']})")
            tqdm.write(f"  I neurons: mean_corr={i_mean:.4f} (n={neuron_info['n_inh']})")
            tqdm.write(f"  Fano factor: model={fano_model.mean():.2f}, real={fano_target.mean():.2f}")
            if h0_stats:
                tqdm.write(f"  h0: mean={h0_stats['mean']:.3f}, std={h0_stats['std']:.3f}")

            # Save checkpoint
            checkpoint_path = output_dir / 'checkpoints' / f'model_epoch{epoch+1}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            tqdm.write(f"  Saved checkpoint: {checkpoint_path.name}\n")

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
            tqdm.write(f"\nEarly stopping at epoch {epoch+1} (no improvement for {config['patience']} epochs)")
            break

    total_time = time.time() - start_time
    epochs_trained = epoch + 1

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total epochs: {epochs_trained}" +
          (f" (early stopped at patience={config['patience']})" if epochs_without_improvement >= config['patience'] else ""))
    print(f"Best validation correlation: {best_val_corr:.4f} (epoch {best_epoch})")
    print(f"Final validation correlation: {val_corr:.4f}")
    print(f"Training time: {str(timedelta(seconds=int(total_time)))}")

    # Load best model for saving
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # =========================================================================
    # SAVE ALL OUTPUTS
    # =========================================================================

    print("\nSaving outputs...")

    # 1. Model checkpoints
    print("  Saving model checkpoints...")
    torch.save(best_model_state, output_dir / 'model_best.pt')
    torch.save(model.state_dict(), output_dir / 'model_final.pt')

    # 2. Weight matrices
    print("  Saving weight matrices...")
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
            'qkv_bias': embed.qkv_proj.bias.detach().cpu().numpy() if embed.qkv_proj.bias is not None else None,
            'out_proj': embed.out_proj.weight.detach().cpu().numpy(),
            'out_proj_bias': embed.out_proj.bias.detach().cpu().numpy() if embed.out_proj.bias is not None else None,
        }
        np.save(output_dir / 'weights' / 'attention_weights.npy', attention_weights)
        np.save(output_dir / 'weights' / 'input_embed_linear.npy', embed.out_proj.weight.detach().cpu().numpy())

    # E/I masks
    n_total = model.n_total
    n_exc = model.n_exc
    E_mask = np.zeros(n_total, dtype=bool)
    E_mask[:n_exc] = True
    I_mask = ~E_mask
    np.save(output_dir / 'weights' / 'E_mask.npy', E_mask)
    np.save(output_dir / 'weights' / 'I_mask.npy', I_mask)

    # 3. Model outputs
    print("  Saving model outputs...")
    model_rates, target_rates, model_psth, target_psth = get_model_outputs(model, val_data, device)

    np.save(output_dir / 'outputs' / 'val_model_rates.npy', model_rates)
    np.save(output_dir / 'outputs' / 'val_target_rates.npy', target_rates)
    np.save(output_dir / 'outputs' / 'val_model_psth.npy', model_psth)
    np.save(output_dir / 'outputs' / 'val_target_psth.npy', target_psth)

    # Trial conditions (using reward as condition label)
    val_trial_conditions = dataset.trial_reward[val_idx]
    np.save(output_dir / 'outputs' / 'val_trial_conditions.npy', val_trial_conditions)

    # 4. Per-neuron metrics
    print("  Saving per-neuron metrics...")
    per_neuron_corr = compute_per_neuron_correlations(model, val_data, device)
    fano_model, fano_target = compute_fano_factors(model, val_data, device)

    np.save(output_dir / 'metrics' / 'per_neuron_correlation.npy', per_neuron_corr)
    np.save(output_dir / 'metrics' / 'per_neuron_mean_rate_model.npy', model_psth.mean(axis=0))
    np.save(output_dir / 'metrics' / 'per_neuron_mean_rate_target.npy', target_psth.mean(axis=0))
    np.save(output_dir / 'metrics' / 'per_neuron_variance_model.npy', model_psth.var(axis=0))
    np.save(output_dir / 'metrics' / 'per_neuron_variance_target.npy', target_psth.var(axis=0))
    np.save(output_dir / 'metrics' / 'per_neuron_fano_model.npy', fano_model)
    np.save(output_dir / 'metrics' / 'per_neuron_fano_target.npy', fano_target)

    # E/I labels for recorded neurons
    n_recorded = target_rates.shape[2]
    neuron_ei_labels = np.zeros(n_recorded, dtype=int)
    neuron_ei_labels[neuron_info['n_exc']:] = 1  # 0=E, 1=I
    np.save(output_dir / 'metrics' / 'neuron_ei_labels.npy', neuron_ei_labels)

    # 5. Population analysis (PCA)
    print("  Computing and saving population analysis...")
    n_pcs = min(10, n_recorded)
    pca = PCA(n_components=n_pcs)
    pca_real = pca.fit_transform(target_psth)
    pca_model = pca.transform(model_psth)

    np.save(output_dir / 'population' / 'pca_real.npy', pca_real)
    np.save(output_dir / 'population' / 'pca_model.npy', pca_model)
    np.save(output_dir / 'population' / 'pca_components.npy', pca.components_)
    np.save(output_dir / 'population' / 'pca_explained_variance.npy', pca.explained_variance_ratio_)

    # 6. Training history
    print("  Saving training history...")
    with open(output_dir / 'training_log.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_json = {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in history.items()}
        json.dump(history_json, f, indent=2)

    # 7. Configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # =========================================================================
    # GENERATE VISUALIZATIONS
    # =========================================================================

    print("\nGenerating visualizations...")
    figures_dir = output_dir / 'figures'

    print("  Training curves...")
    plot_training_curves(history, figures_dir / 'training_curves.png')

    print("  Per-neuron correlation histogram...")
    plot_per_neuron_correlation_histogram(per_neuron_corr, neuron_info,
                                          figures_dir / 'per_neuron_correlation_histogram.png')

    print("  Per-neuron correlation scatter...")
    plot_per_neuron_correlation_scatter(per_neuron_corr, target_psth.mean(axis=0), neuron_info,
                                        figures_dir / 'per_neuron_correlation_scatter.png')

    print("  Best/worst neurons...")
    plot_best_worst_neurons(model_psth, target_psth, per_neuron_corr, bin_size_ms,
                           figures_dir / 'best_worst_neurons.png')

    print("  Weight matrices...")
    plot_weight_matrices(model, neuron_info, figures_dir / 'weight_matrices.png')

    print("  h0 distribution...")
    plot_h0_distribution(model, neuron_info, figures_dir / 'h0_distribution.png')

    print("  Attention patterns...")
    plot_attention_patterns(model, figures_dir / 'attention_patterns.png')

    print("  PCA trajectories...")
    plot_pca_trajectories(pca_real, pca_model, bin_size_ms, figures_dir / 'pca_trajectories.png')

    print("  PCA state space...")
    plot_pca_state_space(pca_real, pca_model, figures_dir / 'pca_state_space.png')

    print("  Fano factor comparison...")
    plot_fano_factor_comparison(fano_model, fano_target, neuron_info,
                                figures_dir / 'fano_factor_comparison.png')

    # =========================================================================
    # GENERATE REPORT
    # =========================================================================

    print("\nGenerating training report...")

    e_mean_corr = per_neuron_corr[:neuron_info['n_exc']].mean()
    i_mean_corr = per_neuron_corr[neuron_info['n_exc']:].mean() if neuron_info['n_inh'] > 0 else 0

    report = f"""# Final Model Training Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Training Summary

| Metric | Value |
|--------|-------|
| Total epochs trained | {epochs_trained} |
| Best validation correlation | {best_val_corr:.4f} |
| Best epoch | {best_epoch} |
| Final validation correlation | {val_corr:.4f} |
| Early stopping | {'Yes' if epochs_without_improvement >= config['patience'] else 'No'} |
| Total training time | {str(timedelta(seconds=int(total_time)))} |

## Final Metrics

| Metric | Value |
|--------|-------|
| Overall PSTH correlation | {per_neuron_corr.mean():.4f} |
| E neuron mean correlation | {e_mean_corr:.4f} |
| I neuron mean correlation | {i_mean_corr:.4f} |
| Model mean Fano factor | {fano_model.mean():.2f} |
| Real mean Fano factor | {fano_target.mean():.2f} |

## Model Configuration

| Parameter | Value |
|-----------|-------|
| max_epochs | {config['max_epochs']} |
| patience | {config['patience']} |
| learning_rate | {config['lr']} |
| tau | {config['tau']} |
| noise_scale | {config['noise_scale']} |
| input_embed_dim | {config['input_embed_dim']} |
| input_embed_type | {config['input_embed_type']} |
| attention_heads | {config['attention_heads']} |
| learnable_h0 | {config['learnable_h0']} |
| h0_init | {config['h0_init']} |

## Data Configuration

| Parameter | Value |
|-----------|-------|
| Data file | {config['data_path']} |
| Total neurons | {neuron_info['n_total']} ({neuron_info['n_exc']} E, {neuron_info['n_inh']} I) |
| Train trials | {len(train_idx)} |
| Validation trials | {len(val_idx)} |
| Time bins | {dataset.n_bins} |
| Bin size | {bin_size_ms} ms |

## Saved Files

### Model Checkpoints
- `model_best.pt` - Best validation correlation checkpoint
- `model_final.pt` - Final epoch checkpoint
- `checkpoints/model_epoch*.pt` - Periodic checkpoints

### Weight Matrices (weights/)
- `W_rec.npy` - Recurrent weights [{model.n_total} x {model.n_total}]
- `W_in.npy` - Input weights [{model.n_total} x {model.actual_n_inputs}]
- `W_out.npy` - Output weights [{model.n_outputs} x {model.n_exc}]
- `h0.npy` - Learned initial state [{model.n_total}]
- `attention_weights.npy` - Attention embedding weights
- `E_mask.npy` - Boolean mask for E neurons
- `I_mask.npy` - Boolean mask for I neurons

### Model Outputs (outputs/)
- `val_model_rates.npy` - Model firing rates [{len(val_idx)} x {dataset.n_bins} x {n_recorded}]
- `val_target_rates.npy` - Target firing rates [{len(val_idx)} x {dataset.n_bins} x {n_recorded}]
- `val_model_psth.npy` - Trial-averaged model PSTH [{dataset.n_bins} x {n_recorded}]
- `val_target_psth.npy` - Trial-averaged target PSTH [{dataset.n_bins} x {n_recorded}]
- `val_trial_conditions.npy` - Condition labels [{len(val_idx)}]

### Per-Neuron Metrics (metrics/)
- `per_neuron_correlation.npy` - PSTH correlation per neuron
- `per_neuron_mean_rate_model.npy` - Mean firing rate (model)
- `per_neuron_mean_rate_target.npy` - Mean firing rate (target)
- `per_neuron_variance_model.npy` - Temporal variance (model)
- `per_neuron_variance_target.npy` - Temporal variance (target)
- `per_neuron_fano_model.npy` - Fano factor (model)
- `per_neuron_fano_target.npy` - Fano factor (target)
- `neuron_ei_labels.npy` - E/I label (0=E, 1=I)

### Population Analysis (population/)
- `pca_real.npy` - PCA projections of real PSTH [{dataset.n_bins} x {n_pcs}]
- `pca_model.npy` - PCA projections of model PSTH [{dataset.n_bins} x {n_pcs}]
- `pca_components.npy` - PCA loading vectors [{n_pcs} x {n_recorded}]
- `pca_explained_variance.npy` - Variance explained per PC

### Training History
- `training_log.json` - Complete epoch-by-epoch training log
- `config.json` - Full configuration used

### Visualizations (figures/)
- `training_curves.png` - Loss and correlation over training
- `per_neuron_correlation_histogram.png` - Correlation distribution by E/I
- `per_neuron_correlation_scatter.png` - Correlation vs firing rate
- `best_worst_neurons.png` - PSTHs for best/worst fitting neurons
- `weight_matrices.png` - Heatmaps of W_rec and W_in
- `h0_distribution.png` - Learned initial state distribution
- `attention_patterns.png` - Attention embedding weights
- `pca_trajectories.png` - PC1-3 over time
- `pca_state_space.png` - PC1 vs PC2 phase plot
- `fano_factor_comparison.png` - Model vs real Fano factors

## Code Snippets for Loading Data

```python
import numpy as np
import torch

# Load weights
W_rec = np.load('results/final_model/weights/W_rec.npy')
W_in = np.load('results/final_model/weights/W_in.npy')
W_out = np.load('results/final_model/weights/W_out.npy')
h0 = np.load('results/final_model/weights/h0.npy')
E_mask = np.load('results/final_model/weights/E_mask.npy')
I_mask = np.load('results/final_model/weights/I_mask.npy')

# Load model outputs
model_rates = np.load('results/final_model/outputs/val_model_rates.npy')
target_rates = np.load('results/final_model/outputs/val_target_rates.npy')
model_psth = np.load('results/final_model/outputs/val_model_psth.npy')
target_psth = np.load('results/final_model/outputs/val_target_psth.npy')

# Load per-neuron metrics
correlations = np.load('results/final_model/metrics/per_neuron_correlation.npy')
ei_labels = np.load('results/final_model/metrics/neuron_ei_labels.npy')

# Load PCA results
pca_real = np.load('results/final_model/population/pca_real.npy')
pca_model = np.load('results/final_model/population/pca_model.npy')
pca_explained = np.load('results/final_model/population/pca_explained_variance.npy')

# Load trained model
from src.model import create_model_from_data
model = create_model_from_data(
    n_classic={neuron_info['n_exc']},
    n_interneuron={neuron_info['n_inh']},
    n_inputs={n_inputs},
    enforce_ratio=True,
    input_embed_dim=56,
    input_embed_type='attention',
    learnable_h0=True,
    device='cpu'
)
model.load_state_dict(torch.load('results/final_model/model_best.pt'))
model.eval()
```
"""

    with open(output_dir / 'training_report.md', 'w') as f:
        f.write(report)

    print("\n" + "=" * 80)
    print(f"All files saved to: {output_dir}")
    print("See training_report.md for full summary.")
    print("=" * 80)

    return {
        'best_val_corr': best_val_corr,
        'best_epoch': best_epoch,
        'final_val_corr': val_corr,
        'epochs_trained': epochs_trained,
        'training_time': total_time,
        'output_dir': str(output_dir)
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train final E-I RNN model')
    parser.add_argument('--test', action='store_true', help='Quick test run (5 epochs)')
    args = parser.parse_args()

    result = train_model(CONFIG, test_mode=args.test)

    print(f"\nFinal result: best_val_corr={result['best_val_corr']:.4f} at epoch {result['best_epoch']}")


if __name__ == '__main__':
    main()
