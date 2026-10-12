#!/usr/bin/env python3
"""
Biological Input Experiments - Main Training Script

Tests the hypothesis that E->E recurrence propagates spatial/value factors
while I->E mediates spatial competition, not factor competition.

Usage:
    python scripts/run_biological_input_experiments.py --experiment baseline
    python scripts/run_biological_input_experiments.py --experiment biological
    python scripts/run_biological_input_experiments.py --experiment all
    python scripts/run_biological_input_experiments.py --test  # Quick test run

Output: results/biological_input_experiments/
"""

import argparse
import json
import time
import copy
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EIRNN, create_model_from_data
from src.losses import compute_L_neuron, compute_L_trial, compute_L_reg
from src.data_loader import (
    load_session, train_val_split, load_mat_file, validate_data,
    infer_rf_locations, construct_restructured_input_tensor, find_target_onset_bin
)
from src.input_assignment import (
    InputAssignment, create_biological_assignment,
    create_baseline_assignment, compute_assignment_statistics
)
from src.analysis.ablation import create_ablated_model, run_ablation_analysis
from src.analysis.emergent_selectivity import compute_emergent_selectivity

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_CONFIG = {
    # Training parameters
    'max_epochs': 2000,
    'patience': 300,
    'lr': 1e-3,
    'lr_scheduler_patience': 50,
    'lr_scheduler_factor': 0.5,
    'min_lr': 1e-5,
    'gradient_clip': 1.0,
    'seed': 42,

    # Model parameters (best from audit)
    'tau': 50.0,
    'dt': 25.0,
    'noise_scale': 0.1,
    'spectral_radius': 0.9,

    # Input embedding
    'input_embed_dim': 56,
    'input_embed_type': 'attention',
    'attention_heads': 4,

    # Initial state
    'learnable_h0': True,
    'h0_init': 0.1,

    # Loss parameters
    'use_grad_balancing': True,
    'ltrial_scale': 0.5,
    'lambda_reg': 1e-4,
    'lambda_scale': 0.1,
    'lambda_var': 0.05,
    'sinkhorn_epsilon': 0.1,

    # Paths
    'data_path': 'data/rnn_export_Newton_08_15_2025_SC.mat',
    'output_dir': 'results/biological_input_experiments',
}

# Biological fractions based on SC anatomy
BIOLOGICAL_FRACTIONS = {
    'e_fractions': {
        'reward': 0.30,       # SNr pathway (~28%)
        'probability': 0.30,  # SNr pathway
        'salience': 0.75,     # V1 pathway (67-85%)
        'identity': 0.75      # V1 pathway
    },
    'i_fractions': {
        'reward': 0.40,       # SNr pathway (~39%)
        'probability': 0.40,  # SNr pathway
        'salience': 0.25,     # V1 pathway (11-33%)
        'identity': 0.25      # V1 pathway
    }
}


# ==============================================================================
# DATA PREPARATION
# ==============================================================================

def prepare_biological_inputs(
    data: dict,
    assignment: InputAssignment,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Prepare inputs based on biological assignment.

    For biological experiments, we construct per-neuron inputs based on
    which factors each neuron receives.

    Args:
        data: Loaded session data
        assignment: InputAssignment specifying what each neuron receives
        device: Compute device

    Returns:
        inputs: [n_trials, n_bins, n_input_channels] tensor
        targets: [n_trials, n_bins, n_neurons] tensor
        trial_factors: Dict of factor values per trial
    """
    n_trials = int(data['n_trials'])
    n_bins = int(data['n_time_bins'])
    n_neurons = int(data['n_neurons'])

    # Use construct_restructured_input_tensor if it exists
    # Otherwise, we'll construct inputs ourselves
    inputs = construct_restructured_input_tensor(data, assignment)

    # Convert to torch
    inputs = torch.tensor(inputs, dtype=torch.float32)

    # Get targets
    firing_rates = data['firing_rates']  # [n_neurons, n_bins, n_trials]
    targets = np.transpose(firing_rates, (2, 1, 0))  # [n_trials, n_bins, n_neurons]
    targets = np.nan_to_num(targets, nan=0.0)
    targets = torch.tensor(targets, dtype=torch.float32)

    # Get trial factors
    trial_factors = {
        'reward': data['trial_reward'].astype(float),
        'probability': data.get('trial_probability', np.zeros(n_trials)).astype(float),
        'salience': data.get('trial_salience', np.zeros(n_trials)).astype(float),
        'identity': data.get('trial_identity', np.zeros(n_trials)).astype(float),
        'location': data['trial_location'].astype(float)
    }

    return inputs, targets, trial_factors


def prepare_standard_inputs(data: dict) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Prepare standard (full) inputs - all neurons receive all factors.

    Uses the standard 14-channel input structure.
    """
    from src.data_loader import construct_input_tensor, construct_target_tensor

    inputs = construct_input_tensor(data)
    targets = construct_target_tensor(data)

    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    n_trials = int(data['n_trials'])
    trial_factors = {
        'reward': data['trial_reward'].astype(float),
        'probability': data.get('trial_probability', np.zeros(n_trials)).astype(float),
        'salience': data.get('trial_salience', np.zeros(n_trials)).astype(float),
        'identity': data.get('trial_identity', np.zeros(n_trials)).astype(float),
        'location': data['trial_location'].astype(float)
    }

    return inputs, targets, trial_factors


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def compute_psth_correlation(
    model: EIRNN,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: str
) -> Tuple[float, np.ndarray]:
    """Compute mean PSTH correlation across neurons."""
    model.eval()
    with torch.no_grad():
        inputs_d = inputs.to(device)
        model_rates, _ = model(inputs_d)

        model_psth = model_rates.mean(dim=0).cpu().numpy()
        target_psth = targets.mean(dim=0).cpu().numpy()

        n_recorded = target_psth.shape[1]
        model_psth = model_psth[:, :n_recorded]

        correlations = np.zeros(n_recorded)
        for i in range(n_recorded):
            r = np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1]
            correlations[i] = r if not np.isnan(r) else 0.0

        return float(np.mean(correlations)), correlations


def train_model(
    model: EIRNN,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    config: Dict,
    device: str,
    output_dir: Path,
    experiment_name: str
) -> Dict:
    """
    Train model with given configuration.

    Returns:
        Dictionary with training results and metrics
    """
    # Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=config['lr_scheduler_factor'],
        patience=config['lr_scheduler_patience'],
        min_lr=config['min_lr']
    )

    # Loss EMA for gradient balancing
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
        'L_neuron': [],
        'L_trial': [],
    }

    n_recorded = train_targets.shape[2]
    bin_size_ms = config.get('bin_size_ms', 25.0)

    print(f"\nTraining {experiment_name}...")
    print(f"  Max epochs: {config['max_epochs']}, Patience: {config['patience']}")

    start_time = time.time()

    pbar = tqdm(range(config['max_epochs']), desc=experiment_name)
    for epoch in pbar:
        # Training step
        model.train()
        inputs = train_inputs.to(device)
        targets = train_targets.to(device)

        optimizer.zero_grad()
        model_rates, _ = model(inputs)
        model_rates_recorded = model_rates[:, :, :n_recorded]

        # Compute losses
        L_neuron = compute_L_neuron(
            model_rates_recorded, targets, bin_size_ms
        )

        L_trial_raw = compute_L_trial(
            model_rates_recorded, targets, bin_size_ms
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
        val_corr, _ = compute_psth_correlation(model, val_inputs, val_targets, device)

        # Compute val loss
        model.eval()
        with torch.no_grad():
            val_inputs_d = val_inputs.to(device)
            val_targets_d = val_targets.to(device)
            val_rates, _ = model(val_inputs_d)
            val_rates_recorded = val_rates[:, :, :n_recorded]
            L_neuron_val = compute_L_neuron(
                val_rates_recorded, val_targets_d, bin_size_ms
            )
            val_loss = L_neuron_val.item()

        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_correlation'].append(val_corr)
        history['learning_rate'].append(current_lr)
        history['L_neuron'].append(L_neuron.item())
        history['L_trial'].append(L_trial_raw.item())

        # Update progress bar
        pbar.set_postfix({
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
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    epochs_trained = epoch + 1

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Get final per-neuron correlations
    _, per_neuron_corr = compute_psth_correlation(model, val_inputs, val_targets, device)

    result = {
        'experiment_name': experiment_name,
        'best_val_corr': best_val_corr,
        'best_epoch': best_epoch,
        'epochs_trained': epochs_trained,
        'training_time': total_time,
        'per_neuron_correlations': per_neuron_corr,
        'history': history,
        'model_state': best_model_state
    }

    print(f"\n  Best: {best_val_corr:.4f} at epoch {best_epoch}")
    print(f"  Time: {timedelta(seconds=int(total_time))}")

    return result


# ==============================================================================
# ABLATION ANALYSIS
# ==============================================================================

def run_ablation_suite(
    model: EIRNN,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    trial_factors: Dict,
    assignment: InputAssignment,
    device: str,
    output_dir: Path,
    experiment_name: str
) -> Dict:
    """
    Run ablation analyses on trained model.

    Tests:
    - full_recurrence: All W_rec = 0
    - ie_only: I->E connections = 0
    - ee_only: E->E connections = 0
    - ei_only: E->I connections = 0
    """
    print(f"\nRunning ablation analyses for {experiment_name}...")

    ablation_types = ['full_recurrence', 'ie_only', 'ee_only', 'ei_only']
    results = {}

    n_recorded = targets.shape[2]

    for abl_type in ablation_types:
        print(f"  Ablation: {abl_type}")

        # Create ablated model
        ablated_model = create_ablated_model(model, abl_type)
        ablated_model.to(device)

        # Compute correlation with ablated model
        val_corr, per_neuron = compute_psth_correlation(
            ablated_model, inputs, targets, device
        )

        # Get model rates for selectivity analysis
        ablated_model.eval()
        with torch.no_grad():
            inputs_d = inputs.to(device)
            model_rates, _ = ablated_model(inputs_d)
            model_rates = model_rates[:, :, :n_recorded].cpu().numpy()

        # Compute emergent selectivity for each factor
        selectivity_results = {}
        for factor in ['reward', 'probability', 'salience', 'identity']:
            if factor in trial_factors:
                sel_result = compute_emergent_selectivity(
                    model_rates,
                    trial_factors[factor],
                    assignment,
                    factor
                )
                selectivity_results[factor] = {
                    'mean_abs_emergent': sel_result.mean_abs_emergent,
                    'n_significant': sel_result.n_significant_emergent,
                    'frac_significant': sel_result.fraction_significant
                }

        results[abl_type] = {
            'val_correlation': val_corr,
            'per_neuron_correlations': per_neuron,
            'selectivity': selectivity_results
        }

    return results


# ==============================================================================
# EMERGENT SELECTIVITY ANALYSIS
# ==============================================================================

def compute_full_selectivity_analysis(
    model: EIRNN,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    trial_factors: Dict,
    assignment: InputAssignment,
    neuron_type: np.ndarray,
    device: str
) -> Dict:
    """
    Compute comprehensive emergent selectivity analysis.

    Separates results by:
    - Factor (reward, probability, salience, identity)
    - Neuron type (E vs I)
    - Whether neuron receives factor as input
    """
    model.eval()
    n_recorded = targets.shape[2]

    with torch.no_grad():
        inputs_d = inputs.to(device)
        model_rates, _ = model(inputs_d)
        model_rates = model_rates[:, :, :n_recorded].cpu().numpy()

    results = {}

    for factor in ['reward', 'probability', 'salience', 'identity']:
        if factor not in trial_factors:
            continue

        factor_values = trial_factors[factor]

        # Get which neurons receive this factor
        receives = assignment.get_input_matrix(factor)[:n_recorded]

        # Compute selectivity for all neurons
        n_neurons = model_rates.shape[2]
        selectivity = np.zeros(n_neurons)

        # Use mean activity in response epoch
        epoch_rates = model_rates.mean(axis=1)  # [trials, neurons]

        for i in range(n_neurons):
            r = np.corrcoef(epoch_rates[:, i], factor_values)[0, 1]
            selectivity[i] = r if not np.isnan(r) else 0.0

        # Separate by receiver status and E/I type
        is_e = neuron_type[:n_recorded] == 1
        is_i = neuron_type[:n_recorded] == 2

        results[factor] = {
            # Overall
            'all_selectivity': selectivity,
            'mean_selectivity': float(np.mean(np.abs(selectivity))),

            # By receiver status
            'receivers_mean': float(np.mean(np.abs(selectivity[receives]))) if receives.sum() > 0 else 0.0,
            'non_receivers_mean': float(np.mean(np.abs(selectivity[~receives]))) if (~receives).sum() > 0 else 0.0,
            'n_receivers': int(receives.sum()),
            'n_non_receivers': int((~receives).sum()),

            # By E/I type
            'e_mean': float(np.mean(np.abs(selectivity[is_e]))) if is_e.sum() > 0 else 0.0,
            'i_mean': float(np.mean(np.abs(selectivity[is_i]))) if is_i.sum() > 0 else 0.0,

            # E neurons: receivers vs non-receivers
            'e_receivers': int(receives[is_e].sum()) if is_e.sum() > 0 else 0,
            'e_non_receivers': int((~receives[is_e]).sum()) if is_e.sum() > 0 else 0,
            'e_receivers_mean': float(np.mean(np.abs(selectivity[is_e & receives]))) if (is_e & receives).sum() > 0 else 0.0,
            'e_non_receivers_mean': float(np.mean(np.abs(selectivity[is_e & ~receives]))) if (is_e & ~receives).sum() > 0 else 0.0,

            # I neurons: receivers vs non-receivers
            'i_receivers': int(receives[is_i].sum()) if is_i.sum() > 0 else 0,
            'i_non_receivers': int((~receives[is_i]).sum()) if is_i.sum() > 0 else 0,
            'i_receivers_mean': float(np.mean(np.abs(selectivity[is_i & receives]))) if (is_i & receives).sum() > 0 else 0.0,
            'i_non_receivers_mean': float(np.mean(np.abs(selectivity[is_i & ~receives]))) if (is_i & ~receives).sum() > 0 else 0.0,

            # Significant emergent (non-receivers with |r| > 0.1)
            'n_significant_emergent': int((np.abs(selectivity[~receives]) > 0.1).sum()) if (~receives).sum() > 0 else 0,
            'frac_significant': float((np.abs(selectivity[~receives]) > 0.1).mean()) if (~receives).sum() > 0 else 0.0,
        }

    return results


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_baseline_experiment(config: Dict, device: str) -> Dict:
    """
    Run E0: Baseline with full input (all neurons receive all factors).
    Uses attention embedding - the proven best configuration.
    """
    output_dir = Path(config['output_dir']) / 'experiments' / 'E0_baseline'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    data = load_mat_file(config['data_path'])
    validate_data(data)

    neuron_type = data['neuron_type']
    n_neurons = int(data['n_neurons'])
    n_exc = (neuron_type == 1).sum()
    n_inh = (neuron_type == 2).sum()

    print(f"Dataset: {n_neurons} neurons ({n_exc} E, {n_inh} I)")

    # Create standard inputs
    inputs, targets, trial_factors = prepare_standard_inputs(data)

    # Train/val split
    n_trials = inputs.shape[0]
    np.random.seed(config['seed'])
    indices = np.arange(n_trials)
    np.random.shuffle(indices)
    n_val = int(n_trials * 0.2)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_inputs = inputs[train_idx]
    train_targets = targets[train_idx]
    val_inputs = inputs[val_idx]
    val_targets = targets[val_idx]

    # Create baseline assignment (all neurons receive all factors)
    rf_locations = infer_rf_locations(data)
    assignment = create_baseline_assignment(n_neurons, neuron_type, rf_locations)

    print(f"Input shape: {inputs.shape}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Create model
    model = create_model_from_data(
        n_classic=n_exc,
        n_interneuron=n_inh,
        n_inputs=inputs.shape[2],  # 14 standard channels
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

    # Train
    config['bin_size_ms'] = float(data['bin_size_ms'])
    result = train_model(
        model, train_inputs, train_targets, val_inputs, val_targets,
        config, device, output_dir, 'E0_baseline'
    )

    # Load best model state
    model.load_state_dict(result['model_state'])

    # Run selectivity analysis
    print("\nComputing selectivity analysis...")
    selectivity = compute_full_selectivity_analysis(
        model, val_inputs, val_targets, {k: v[val_idx] for k, v in trial_factors.items()},
        assignment, neuron_type, device
    )
    result['selectivity'] = selectivity

    # Run ablation analysis
    print("\nRunning ablation suite...")
    ablation_results = run_ablation_suite(
        model, val_inputs, val_targets, {k: v[val_idx] for k, v in trial_factors.items()},
        assignment, device, output_dir, 'E0_baseline'
    )
    result['ablations'] = ablation_results

    # Save results
    result_save = {k: v for k, v in result.items() if k != 'model_state'}
    result_save['per_neuron_correlations'] = result['per_neuron_correlations'].tolist()
    result_save['history'] = {k: v for k, v in result['history'].items()}

    # Convert numpy arrays in ablation results
    for abl_type, abl_data in result_save['ablations'].items():
        abl_data['per_neuron_correlations'] = abl_data['per_neuron_correlations'].tolist()

    # Convert numpy arrays in selectivity
    for factor, factor_data in result_save['selectivity'].items():
        factor_data['all_selectivity'] = factor_data['all_selectivity'].tolist()

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(result_save, f, indent=2)

    # Save model
    torch.save(result['model_state'], output_dir / 'model_best.pt')

    # Save assignment
    np.save(output_dir / 'assignment.npy', assignment.to_dict(), allow_pickle=True)

    print(f"\nE0 Baseline complete: {result['best_val_corr']:.4f}")

    return result


def run_biological_experiment(config: Dict, device: str) -> Dict:
    """
    Run E1: Biological input fractions.

    E neurons: 30% receive reward/prob, 75% receive salience/identity
    I neurons: 40% receive reward/prob, 25% receive salience/identity
    """
    output_dir = Path(config['output_dir']) / 'experiments' / 'E1_biological'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    data = load_mat_file(config['data_path'])
    validate_data(data)

    neuron_type = data['neuron_type']
    n_neurons = int(data['n_neurons'])
    n_exc = (neuron_type == 1).sum()
    n_inh = (neuron_type == 2).sum()

    print(f"Dataset: {n_neurons} neurons ({n_exc} E, {n_inh} I)")

    # Create biological assignment
    rf_locations = infer_rf_locations(data)
    assignment = create_biological_assignment(
        n_neurons=n_neurons,
        neuron_type=neuron_type,
        rf_locations=rf_locations,
        e_fractions=BIOLOGICAL_FRACTIONS['e_fractions'],
        i_fractions=BIOLOGICAL_FRACTIONS['i_fractions'],
        seed=config['seed'],
        experiment_type='E1_BIO'
    )

    print(f"\nBiological assignment:")
    print(assignment.summary())
    stats = compute_assignment_statistics(assignment)
    for k, v in stats.items():
        if 'n_' in k or 'frac' in k:
            print(f"  {k}: {v}")

    # Prepare biological inputs
    inputs, targets, trial_factors = prepare_biological_inputs(data, assignment, device)

    # Train/val split
    n_trials = inputs.shape[0]
    np.random.seed(config['seed'])
    indices = np.arange(n_trials)
    np.random.shuffle(indices)
    n_val = int(n_trials * 0.2)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_inputs = inputs[train_idx]
    train_targets = targets[train_idx]
    val_inputs = inputs[val_idx]
    val_targets = targets[val_idx]

    print(f"Input shape: {inputs.shape}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # For biological inputs, we need per-neuron input processing
    # The input is [n_trials, n_bins, n_neurons, 6]
    # We need to flatten or adapt for the model

    # Reshape inputs: flatten neuron dimension into input channels
    # New shape: [n_trials, n_bins, n_neurons * 6]
    n_input_channels = inputs.shape[3]
    inputs_flat = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
    train_inputs_flat = train_inputs.reshape(train_inputs.shape[0], train_inputs.shape[1], -1)
    val_inputs_flat = val_inputs.reshape(val_inputs.shape[0], val_inputs.shape[1], -1)

    print(f"Flattened input shape: {inputs_flat.shape}")

    # Create model with expanded input dimension
    # For biological experiment, we use linear embedding (not attention)
    # because the input structure is different
    bio_config = config.copy()
    bio_config['input_embed_dim'] = 56  # Keep same embedding dimension
    bio_config['input_embed_type'] = 'linear'  # Use linear for compatibility

    model = create_model_from_data(
        n_classic=n_exc,
        n_interneuron=n_inh,
        n_inputs=inputs_flat.shape[2],  # n_neurons * 6
        enforce_ratio=True,
        dt=bio_config['dt'],
        tau=bio_config['tau'],
        noise_scale=bio_config['noise_scale'],
        spectral_radius=bio_config['spectral_radius'],
        input_embed_dim=bio_config['input_embed_dim'],
        input_embed_type=bio_config['input_embed_type'],
        attention_heads=bio_config['attention_heads'],
        learnable_h0=bio_config['learnable_h0'],
        h0_init=bio_config['h0_init'],
        device=device
    )

    # Train
    bio_config['bin_size_ms'] = float(data['bin_size_ms'])
    result = train_model(
        model, train_inputs_flat, train_targets, val_inputs_flat, val_targets,
        bio_config, device, output_dir, 'E1_biological'
    )

    # Load best model state
    model.load_state_dict(result['model_state'])

    # Run selectivity analysis
    print("\nComputing selectivity analysis...")
    selectivity = compute_full_selectivity_analysis(
        model, val_inputs_flat, val_targets, {k: v[val_idx] for k, v in trial_factors.items()},
        assignment, neuron_type, device
    )
    result['selectivity'] = selectivity

    # Run ablation analysis
    print("\nRunning ablation suite...")
    ablation_results = run_ablation_suite(
        model, val_inputs_flat, val_targets, {k: v[val_idx] for k, v in trial_factors.items()},
        assignment, device, output_dir, 'E1_biological'
    )
    result['ablations'] = ablation_results

    # Save results
    result_save = {k: v for k, v in result.items() if k != 'model_state'}
    result_save['per_neuron_correlations'] = result['per_neuron_correlations'].tolist()
    result_save['biological_fractions'] = BIOLOGICAL_FRACTIONS

    # Convert numpy arrays
    for abl_type, abl_data in result_save['ablations'].items():
        abl_data['per_neuron_correlations'] = abl_data['per_neuron_correlations'].tolist()

    for factor, factor_data in result_save['selectivity'].items():
        factor_data['all_selectivity'] = factor_data['all_selectivity'].tolist()

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(result_save, f, indent=2)

    # Save model and assignment
    torch.save(result['model_state'], output_dir / 'model_best.pt')
    np.save(output_dir / 'assignment.npy', assignment.to_dict(), allow_pickle=True)

    print(f"\nE1 Biological complete: {result['best_val_corr']:.4f}")

    return result


# ==============================================================================
# FIGURE GENERATION
# ==============================================================================

def generate_comparison_figures(
    baseline_result: Dict,
    biological_result: Dict,
    output_dir: Path
):
    """Generate comparison figures for hypothesis testing."""

    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Figure 1: Fit Quality Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    experiments = ['E0 Baseline', 'E1 Biological']
    correlations = [baseline_result['best_val_corr'], biological_result['best_val_corr']]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax.bar(experiments, correlations, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Validation PSTH Correlation', fontsize=12)
    ax.set_title('Fit Quality: Baseline vs Biological Input Constraints', fontsize=14)
    ax.set_ylim(0, 0.5)

    for bar, val in zip(bars, correlations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')

    # Add threshold line at 0.35 (85% of expected baseline)
    ax.axhline(y=0.35, color='orange', linestyle='--', linewidth=2, label='Success threshold (0.35)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / 'fit_quality_comparison.png', dpi=150)
    plt.close()

    # Figure 2: Emergent Selectivity by Factor
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    factors = ['reward', 'probability', 'salience', 'identity']

    for idx, (result, title, ax) in enumerate([
        (baseline_result, 'E0 Baseline', axes[0]),
        (biological_result, 'E1 Biological', axes[1])
    ]):
        receivers = [result['selectivity'][f]['receivers_mean'] for f in factors]
        non_receivers = [result['selectivity'][f]['non_receivers_mean'] for f in factors]

        x = np.arange(len(factors))
        width = 0.35

        bars1 = ax.bar(x - width/2, receivers, width, label='Receivers', color='#3498db')
        bars2 = ax.bar(x + width/2, non_receivers, width, label='Non-receivers (Emergent)', color='#e74c3c')

        ax.set_ylabel('Mean |Selectivity|')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(factors)
        ax.legend()
        ax.set_ylim(0, 0.5)

    plt.suptitle('Emergent Selectivity: Receivers vs Non-Receivers', fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / 'emergent_selectivity_by_factor.png', dpi=150)
    plt.close()

    # Figure 3: Ablation Effects - I->E vs E->E
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (result, title, ax) in enumerate([
        (baseline_result, 'E0 Baseline', axes[0]),
        (biological_result, 'E1 Biological', axes[1])
    ]):
        # Get original selectivity
        original = {f: result['selectivity'][f]['non_receivers_mean'] for f in factors}

        # Get IE ablation selectivity
        ie_ablated = {f: result['ablations']['ie_only']['selectivity'][f]['mean_abs_emergent']
                     for f in factors if f in result['ablations']['ie_only']['selectivity']}

        # Get EE ablation selectivity
        ee_ablated = {f: result['ablations']['ee_only']['selectivity'][f]['mean_abs_emergent']
                     for f in factors if f in result['ablations']['ee_only']['selectivity']}

        x = np.arange(len(factors))
        width = 0.25

        bars1 = ax.bar(x - width, [original[f] for f in factors], width, label='Original', color='#2ecc71')
        bars2 = ax.bar(x, [ie_ablated.get(f, 0) for f in factors], width, label='I->E Ablated', color='#e74c3c')
        bars3 = ax.bar(x + width, [ee_ablated.get(f, 0) for f in factors], width, label='E->E Ablated', color='#3498db')

        ax.set_ylabel('Emergent Selectivity')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(factors)
        ax.legend()
        ax.set_ylim(0, 0.6)

    plt.suptitle('Ablation Effects: I->E vs E->E on Emergent Selectivity', fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / 'ablation_by_factor.png', dpi=150)
    plt.close()

    # Figure 4: E vs I Neuron Selectivity
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (result, title, ax) in enumerate([
        (baseline_result, 'E0 Baseline', axes[0]),
        (biological_result, 'E1 Biological', axes[1])
    ]):
        e_selectivity = [result['selectivity'][f]['e_non_receivers_mean'] for f in factors]
        i_selectivity = [result['selectivity'][f]['i_non_receivers_mean'] for f in factors]

        x = np.arange(len(factors))
        width = 0.35

        bars1 = ax.bar(x - width/2, e_selectivity, width, label='E neurons', color='#3498db')
        bars2 = ax.bar(x + width/2, i_selectivity, width, label='I neurons', color='#e74c3c')

        ax.set_ylabel('Emergent Selectivity')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(factors)
        ax.legend()
        ax.set_ylim(0, 0.5)

    plt.suptitle('Emergent Selectivity: E vs I Neurons (Non-Receivers Only)', fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / 'emergent_selectivity_e_vs_i.png', dpi=150)
    plt.close()

    print(f"\nFigures saved to {figures_dir}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run biological input experiments')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['baseline', 'biological', 'all'],
                       help='Which experiment to run')
    parser.add_argument('--test', action='store_true',
                       help='Quick test run (5 epochs)')
    args = parser.parse_args()

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Config
    config = BASE_CONFIG.copy()

    if args.test:
        print("\n=== TEST MODE ===")
        config['max_epochs'] = 5
        config['patience'] = 3

    # Output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    results = {}

    # Run experiments
    if args.experiment in ['baseline', 'all']:
        results['baseline'] = run_baseline_experiment(config, device)

    if args.experiment in ['biological', 'all']:
        results['biological'] = run_biological_experiment(config, device)

    # Generate comparison figures if both experiments completed
    if 'baseline' in results and 'biological' in results:
        generate_comparison_figures(
            results['baseline'],
            results['biological'],
            output_dir
        )

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Best val correlation: {result['best_val_corr']:.4f}")
        print(f"  Best epoch: {result['best_epoch']}")
        print(f"  Training time: {timedelta(seconds=int(result['training_time']))}")

        if 'selectivity' in result:
            print("  Emergent selectivity (non-receivers):")
            for factor in ['reward', 'probability', 'salience', 'identity']:
                if factor in result['selectivity']:
                    sel = result['selectivity'][factor]
                    print(f"    {factor}: {sel['non_receivers_mean']:.3f} "
                          f"({sel['frac_significant']*100:.1f}% significant)")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
