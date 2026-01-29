#!/usr/bin/env python3
"""
Condition-Specific Loss Training Script

Trains the E-I RNN with condition-specific PSTH loss to preserve factor selectivity.

Key differences from train_final_model.py:
- Uses compute_L_neuron_conditioned() instead of compute_L_neuron()
- Computes separate PSTHs for each of 16 conditions (4 loc × 2 reward × 2 salience)
- Tracks per-condition losses and selectivity metrics

Usage:
    python scripts/train_conditioned_loss.py          # Full run
    python scripts/train_conditioned_loss.py --test   # Quick 5-epoch test

Output directory: results/conditioned_loss_08_15/
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
from scipy import stats
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EIRNN, create_model_from_data
from src.losses import (
    compute_L_neuron, compute_L_neuron_conditioned,
    compute_L_trial, compute_L_reg, compute_selectivity_index
)
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

    # Model parameters (same as final model)
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
    'min_trials_per_condition': 3,

    # Logging parameters
    'checkpoint_every': 100,
    'detailed_log_every': 100,
    'print_every': 10,

    # Paths
    'data_path': 'data/rnn_export_Newton_08_15_2025_SC.mat',
    'output_dir': 'results/conditioned_loss_08_15',
    'original_model_dir': 'results/final_model',
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_psth_correlation(
    model: EIRNN,
    data: Dict,
    device: str
) -> float:
    """Compute mean PSTH correlation across neurons (grand average)."""
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


def compute_per_condition_correlations(
    model: EIRNN,
    data: Dict,
    device: str
) -> Dict[int, float]:
    """Compute mean PSTH correlation for each condition."""
    model.eval()
    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        conditions = data['trial_conditions'].to(device)
        model_rates, _ = model(inputs)

        n_recorded = targets.shape[2]
        model_rates = model_rates[:, :, :n_recorded]

        per_condition_corr = {}
        for cond in torch.unique(conditions):
            mask = conditions == cond
            if mask.sum() < 3:
                continue

            model_cond = model_rates[mask].mean(dim=0).cpu().numpy()
            target_cond = targets[mask].mean(dim=0).cpu().numpy()

            correlations = []
            for i in range(n_recorded):
                r = np.corrcoef(model_cond[:, i], target_cond[:, i])[0, 1]
                if not np.isnan(r):
                    correlations.append(r)

            per_condition_corr[int(cond.item())] = (
                np.mean(correlations) if correlations else 0.0
            )

        return per_condition_corr


def compute_factor_selectivity(
    rates: np.ndarray,
    factor_values: np.ndarray,
    method: str = 'dprime'
) -> np.ndarray:
    """
    Compute selectivity index for each neuron for a binary factor.

    Args:
        rates: [n_trials, n_time, n_neurons] or [n_trials, n_neurons]
        factor_values: [n_trials] binary factor (0/1)
        method: 'dprime' or 'auroc'

    Returns:
        [n_neurons] selectivity values
    """
    # Get mean rates per trial
    if rates.ndim == 3:
        mean_rates = rates.mean(axis=1)  # [n_trials, n_neurons]
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

        if method == 'dprime':
            # d-prime
            mean_diff = high_rates.mean() - low_rates.mean()
            pooled_var = (low_rates.var() + high_rates.var()) / 2
            pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 1e-6
            selectivity[i] = mean_diff / pooled_std
        elif method == 'auroc':
            # Area under ROC
            try:
                from sklearn.metrics import roc_auc_score
                y_true = np.concatenate([np.zeros(len(low_rates)), np.ones(len(high_rates))])
                y_score = np.concatenate([low_rates, high_rates])
                selectivity[i] = roc_auc_score(y_true, y_score)
            except:
                selectivity[i] = 0.5

    return selectivity


def compute_selectivity_correlation(
    model: EIRNN,
    data: Dict,
    device: str
) -> Dict[str, float]:
    """
    Compute correlation between recorded and RNN selectivity for each factor.

    Returns dict with keys: 'reward_corr', 'salience_corr', 'location_corr'
    """
    model.eval()
    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        model_rates, _ = model(inputs)

        n_recorded = targets.shape[2]
        model_rates_np = model_rates[:, :, :n_recorded].cpu().numpy()
        target_rates_np = targets.cpu().numpy()

        results = {}

        # Reward selectivity
        if 'trial_reward' in data:
            reward = data['trial_reward'].numpy()
            model_sel = compute_factor_selectivity(model_rates_np, reward)
            target_sel = compute_factor_selectivity(target_rates_np, reward)
            r, p = stats.pearsonr(model_sel, target_sel)
            results['reward_corr'] = r if not np.isnan(r) else 0.0
            results['reward_p'] = p if not np.isnan(p) else 1.0

        # Salience selectivity
        if 'trial_salience' in data:
            salience = data['trial_salience'].numpy()
            if len(np.unique(salience)) > 1:  # Need variation
                model_sel = compute_factor_selectivity(model_rates_np, salience)
                target_sel = compute_factor_selectivity(target_rates_np, salience)
                r, p = stats.pearsonr(model_sel, target_sel)
                results['salience_corr'] = r if not np.isnan(r) else 0.0
                results['salience_p'] = p if not np.isnan(p) else 1.0
            else:
                results['salience_corr'] = 0.0
                results['salience_p'] = 1.0

        # Location selectivity (compare each location vs others)
        if 'trial_location' in data:
            location = data['trial_location'].numpy()
            location_sels_model = []
            location_sels_target = []

            for loc in np.unique(location):
                loc_binary = (location == loc).astype(int)
                model_sel = compute_factor_selectivity(model_rates_np, loc_binary)
                target_sel = compute_factor_selectivity(target_rates_np, loc_binary)
                location_sels_model.append(model_sel)
                location_sels_target.append(target_sel)

            # Average selectivity magnitude across locations
            model_loc_sel = np.abs(np.stack(location_sels_model)).mean(axis=0)
            target_loc_sel = np.abs(np.stack(location_sels_target)).mean(axis=0)
            r, p = stats.pearsonr(model_loc_sel, target_loc_sel)
            results['location_corr'] = r if not np.isnan(r) else 0.0
            results['location_p'] = p if not np.isnan(p) else 1.0

        return results


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

        n_recorded = target_rates_np.shape[2]
        model_rates_np = model_rates_np[:, :, :n_recorded]

        model_psth = model_rates_np.mean(axis=0)
        target_psth = target_rates_np.mean(axis=0)

        return model_rates_np, target_rates_np, model_psth, target_psth


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_training_curves(history: Dict, save_path: Path):
    """Plot training curves including L_trial."""
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

    # L_trial trajectory (key metric!)
    ax = axes[1, 0]
    ax.plot(epochs, history['L_trial'], 'purple', linewidth=2, label='L_trial')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L_trial')
    ax.set_title('Trial-Matching Loss (Should Decrease!)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss Components
    ax = axes[1, 1]
    ax.plot(epochs, history['L_neuron_cond'], 'b-', alpha=0.7, label='L_neuron_cond')
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


def plot_selectivity_comparison(
    selectivity_history: list,
    save_path: Path
):
    """Plot selectivity correlation over training."""
    if not selectivity_history:
        return

    epochs = [s['epoch'] for s in selectivity_history]

    fig, ax = plt.subplots(figsize=(10, 6))

    for factor in ['reward', 'salience', 'location']:
        key = f'{factor}_corr'
        if key in selectivity_history[0]:
            values = [s[key] for s in selectivity_history]
            ax.plot(epochs, values, '-o', label=f'{factor.capitalize()}', markersize=4)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Selectivity Correlation (recorded vs RNN)')
    ax.set_title('Factor Selectivity Matching Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_per_condition_psth_examples(
    model: EIRNN,
    data: Dict,
    device: str,
    neuron_idx: int,
    save_path: Path,
    bin_size_ms: float = 25.0
):
    """Plot PSTHs for different conditions for a single neuron."""
    model.eval()
    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        conditions = data['trial_conditions'].to(device)
        model_rates, _ = model(inputs)

        n_recorded = targets.shape[2]
        model_rates = model_rates[:, :, :n_recorded]

        time_axis = np.arange(targets.shape[1]) * bin_size_ms / 1000

        # Get unique conditions and select a subset
        unique_conds = torch.unique(conditions).cpu().numpy()
        n_show = min(8, len(unique_conds))
        show_conds = unique_conds[:n_show]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i, cond in enumerate(show_conds):
            ax = axes[i]
            mask = conditions.cpu() == cond

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

        plt.suptitle(f'Neuron {neuron_idx}: Condition-Specific PSTHs', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()


def plot_ltrial_comparison(
    history: Dict,
    original_history: Optional[Dict],
    save_path: Path
):
    """Compare L_trial trajectories between conditioned and original model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = np.arange(1, len(history['L_trial']) + 1)
    ax.plot(epochs, history['L_trial'], 'b-', linewidth=2, label='Conditioned Loss')

    if original_history and 'L_trial' in original_history:
        orig_epochs = np.arange(1, len(original_history['L_trial']) + 1)
        ax.plot(orig_epochs, original_history['L_trial'], 'r--', linewidth=2,
                label='Original (Grand Average)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('L_trial')
    ax.set_title('L_trial Comparison: Conditioned vs Original Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train_model(config: Dict, test_mode: bool = False) -> Dict:
    """Main training function with condition-specific loss."""

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
    for subdir in ['weights', 'outputs', 'metrics', 'population', 'figures', 'checkpoints']:
        (output_dir / subdir).mkdir(exist_ok=True)

    # Load data
    print(f"Loading data from {config['data_path']}...")
    dataset = load_session(config['data_path'])
    train_idx, val_idx = train_val_split(dataset, val_fraction=0.2, seed=config['seed'])

    # Get data with condition labels
    all_data = dataset.get_all_trials(include_conditions=True)
    train_data = {
        'inputs': all_data['inputs'][train_idx],
        'targets': all_data['targets'][train_idx],
        'mask': all_data['mask'][train_idx],
        'trial_conditions': all_data['trial_conditions'][train_idx],
        'trial_reward': all_data['trial_reward'][train_idx],
        'trial_location': all_data['trial_location'][train_idx],
        'trial_salience': all_data['trial_salience'][train_idx],
    }
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
    condition_info = dataset.get_condition_info()
    n_inputs = dataset.get_input_dim()
    bin_size_ms = dataset.bin_size_ms

    print(f"Train trials: {len(train_idx)}, Val trials: {len(val_idx)}")
    print(f"Neurons: {neuron_info['n_total']} ({neuron_info['n_exc']} E, {neuron_info['n_inh']} I)")
    print(f"Input dim: {n_inputs}, Time bins: {dataset.n_bins}, Bin size: {bin_size_ms}ms")
    print(f"\nCondition structure:")
    print(f"  Number of conditions: {condition_info['n_conditions']}")
    print(f"  Trials per condition: {condition_info['min_trials_per_condition']}-{condition_info['max_trials_per_condition']} (mean: {condition_info['mean_trials_per_condition']:.1f})")

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
        'L_neuron_cond': [],
        'L_trial': [],
        'L_reg': [],
        'per_condition_loss': [],
        'selectivity_history': [],
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
    print(f"Using CONDITION-SPECIFIC PSTH loss (16 conditions)")
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
        conditions = train_data['trial_conditions'].to(device)

        optimizer.zero_grad()
        model_rates, _ = model(inputs)
        n_recorded = targets.shape[2]
        model_rates_recorded = model_rates[:, :, :n_recorded]

        # Compute condition-specific PSTH loss
        L_neuron_cond, per_cond_loss = compute_L_neuron_conditioned(
            model_rates_recorded, targets, conditions, bin_size_ms,
            mask=mask,
            lambda_scale=config['lambda_scale'],
            lambda_var=config['lambda_var'],
            min_trials_per_condition=config['min_trials_per_condition']
        )

        # Compute trial-matching loss
        L_trial_raw = compute_L_trial(
            model_rates_recorded, targets, bin_size_ms,
            mask=mask, sinkhorn_iters=20, sinkhorn_epsilon=0.1
        )

        L_reg = compute_L_reg(model, config['lambda_reg'])

        # Gradient balancing
        if config['use_grad_balancing']:
            with torch.no_grad():
                loss_ema['L_neuron'] = ema_decay * loss_ema['L_neuron'] + (1 - ema_decay) * L_neuron_cond.item()
                loss_ema['L_trial'] = ema_decay * loss_ema['L_trial'] + (1 - ema_decay) * L_trial_raw.item()

            L_neuron_norm = L_neuron_cond / (loss_ema['L_neuron'] + 1e-8)
            L_trial_norm = L_trial_raw / (loss_ema['L_trial'] + 1e-8)
            loss = L_neuron_norm + config['ltrial_scale'] * L_trial_norm + L_reg
        else:
            loss = L_neuron_cond + config['ltrial_scale'] * L_trial_raw + L_reg

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
            conditions_val = val_data['trial_conditions'].to(device)

            model_rates_val, _ = model(inputs_val)
            model_rates_val_recorded = model_rates_val[:, :, :n_recorded]

            L_neuron_val, _ = compute_L_neuron_conditioned(
                model_rates_val_recorded, targets_val, conditions_val, bin_size_ms,
                mask=mask_val,
                lambda_scale=config['lambda_scale'],
                lambda_var=config['lambda_var'],
                min_trials_per_condition=config['min_trials_per_condition']
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
        history['L_neuron_cond'].append(L_neuron_cond.item())
        history['L_trial'].append(L_trial_raw.item())
        history['L_reg'].append(L_reg.item())
        history['per_condition_loss'].append(per_cond_loss)

        # Update progress bar
        pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val_corr': f'{val_corr:.4f}',
            'L_trial': f'{L_trial_raw.item():.4f}',
            'best': f'{best_val_corr:.4f}',
            'lr': f'{current_lr:.1e}'
        })

        # Print every N epochs
        if (epoch + 1) % config['print_every'] == 0:
            tqdm.write(f"Epoch {epoch+1:4d} | train: {train_loss:.4f} | val_corr: {val_corr:.4f} | "
                       f"L_trial: {L_trial_raw.item():.4f} | lr: {current_lr:.1e}")

        # Detailed log every N epochs
        if (epoch + 1) % config['detailed_log_every'] == 0:
            per_neuron_corr = compute_per_neuron_correlations(model, val_data, device)
            per_cond_corr = compute_per_condition_correlations(model, val_data, device)
            selectivity = compute_selectivity_correlation(model, val_data, device)

            selectivity['epoch'] = epoch + 1
            history['selectivity_history'].append(selectivity)

            e_mean = per_neuron_corr[:neuron_info['n_exc']].mean()
            i_mean = per_neuron_corr[neuron_info['n_exc']:].mean() if neuron_info['n_inh'] > 0 else 0

            detailed_log = {
                'epoch': epoch + 1,
                'per_neuron_correlations': per_neuron_corr.tolist(),
                'per_condition_correlations': per_cond_corr,
                'E_mean_corr': float(e_mean),
                'I_mean_corr': float(i_mean),
                'selectivity': selectivity
            }
            history['detailed_logs'].append(detailed_log)

            tqdm.write(f"\n=== Epoch {epoch+1} Detailed Report ===")
            tqdm.write(f"  E neurons: mean_corr={e_mean:.4f}")
            tqdm.write(f"  I neurons: mean_corr={i_mean:.4f}")
            tqdm.write(f"  Selectivity correlations:")
            for key, val in selectivity.items():
                if 'corr' in key:
                    tqdm.write(f"    {key}: {val:.4f}")

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
            tqdm.write(f"\nEarly stopping at epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    epochs_trained = epoch + 1

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total epochs: {epochs_trained}")
    print(f"Best validation correlation: {best_val_corr:.4f} (epoch {best_epoch})")
    print(f"Final L_trial: {history['L_trial'][-1]:.4f}")
    print(f"Training time: {str(timedelta(seconds=int(total_time)))}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================

    print("\nSaving outputs...")

    # Model checkpoints
    torch.save(best_model_state, output_dir / 'model_best.pt')
    torch.save(model.state_dict(), output_dir / 'model_final.pt')

    # Weights
    np.save(output_dir / 'weights' / 'W_rec.npy', model.W_rec.detach().cpu().numpy())
    np.save(output_dir / 'weights' / 'W_in.npy', model.W_in.detach().cpu().numpy())
    np.save(output_dir / 'weights' / 'W_out.npy', model.W_out.detach().cpu().numpy())
    if model.h0 is not None:
        np.save(output_dir / 'weights' / 'h0.npy', model.h0.detach().cpu().numpy())

    # Model outputs
    model_rates_np, target_rates_np, model_psth, target_psth = get_model_outputs(model, val_data, device)
    np.save(output_dir / 'outputs' / 'val_model_rates.npy', model_rates_np)
    np.save(output_dir / 'outputs' / 'val_target_rates.npy', target_rates_np)
    np.save(output_dir / 'outputs' / 'val_model_psth.npy', model_psth)
    np.save(output_dir / 'outputs' / 'val_target_psth.npy', target_psth)
    np.save(output_dir / 'outputs' / 'val_trial_conditions.npy', val_data['trial_conditions'].numpy())

    # Per-neuron metrics
    per_neuron_corr = compute_per_neuron_correlations(model, val_data, device)
    np.save(output_dir / 'metrics' / 'per_neuron_correlation.npy', per_neuron_corr)

    # Final selectivity
    final_selectivity = compute_selectivity_correlation(model, val_data, device)

    # Training history
    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)

    # Config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Condition info
    with open(output_dir / 'condition_info.json', 'w') as f:
        json.dump(condition_info, f, indent=2)

    # =========================================================================
    # GENERATE VISUALIZATIONS
    # =========================================================================

    print("\nGenerating visualizations...")
    figures_dir = output_dir / 'figures'

    plot_training_curves(history, figures_dir / 'training_curves.png')

    if history['selectivity_history']:
        plot_selectivity_comparison(history['selectivity_history'],
                                    figures_dir / 'selectivity_over_training.png')

    # Per-condition PSTH examples (select a neuron with high correlation)
    best_neuron = np.argmax(per_neuron_corr)
    plot_per_condition_psth_examples(model, val_data, device, best_neuron,
                                     figures_dir / 'per_condition_psth_examples.png',
                                     bin_size_ms)

    # Try to load original model history for comparison
    original_history = None
    orig_path = Path(config['original_model_dir']) / 'training_log.json'
    if orig_path.exists():
        try:
            with open(orig_path) as f:
                original_history = json.load(f)
        except:
            pass

    plot_ltrial_comparison(history, original_history, figures_dir / 'ltrial_comparison.png')

    print(f"\nAll files saved to: {output_dir}")

    return {
        'best_val_corr': best_val_corr,
        'best_epoch': best_epoch,
        'final_val_corr': val_corr,
        'final_ltrial': history['L_trial'][-1],
        'epochs_trained': epochs_trained,
        'training_time': total_time,
        'selectivity': final_selectivity,
        'output_dir': str(output_dir)
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train E-I RNN with condition-specific loss')
    parser.add_argument('--test', action='store_true', help='Quick test run (5 epochs)')
    args = parser.parse_args()

    result = train_model(CONFIG, test_mode=args.test)

    print(f"\nFinal result:")
    print(f"  Best validation correlation: {result['best_val_corr']:.4f} (epoch {result['best_epoch']})")
    print(f"  Final L_trial: {result['final_ltrial']:.4f}")
    print(f"\nSelectivity correlations:")
    for key, val in result['selectivity'].items():
        if 'corr' in key:
            print(f"  {key}: {val:.4f}")


if __name__ == '__main__':
    main()
