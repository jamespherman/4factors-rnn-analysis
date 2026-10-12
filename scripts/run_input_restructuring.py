#!/usr/bin/env python3
"""
Input Restructuring Experiments Runner

Trains E-I RNN models with restructured inputs as specified in INPUT_RESTRUCTURING_SPEC.md.
Tests whether factor selectivity arises from recurrent dynamics or direct input drive.

Experiments:
    E0: Baseline (all neurons, all factors, at target onset)
    E1: Temporal fix (all neurons, spatial from trial start)
    E2: Primary test (RF-matched spatial, sparse features p=0.5) - 5 seeds
    E3a: Sparsity test p=0.25 - 3 seeds
    E3b: Sparsity test p=0.75 - 3 seeds
    E4: Replication with different seeds - 5 seeds

Usage:
    python scripts/run_input_restructuring.py                      # Full run (18 models)
    python scripts/run_input_restructuring.py --priority-only      # E0, E1, E2_seed0 only
    python scripts/run_input_restructuring.py --experiment E2      # Specific experiment
    python scripts/run_input_restructuring.py --experiment E2 --seed 0  # Single run
    python scripts/run_input_restructuring.py --test               # Quick test (5 epochs)

Output: results/input_restructuring/
"""

import argparse
import json
import time
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass

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
    load_mat_file, validate_data, construct_input_tensor,
    construct_target_tensor, construct_mask, train_val_split,
    RNNDataset, infer_rf_locations, find_target_onset_bin,
    construct_restructured_input_tensor
)
from src.input_assignment import (
    InputAssignment, create_assignment_for_experiment,
    save_assignment, compute_assignment_statistics
)
from src.analysis.emergent_selectivity import (
    compute_all_factor_selectivities, summarize_emergent_selectivity,
    compute_selectivity_by_neuron_type
)
from src.analysis.ablation import run_ablation_analysis, print_ablation_report


# =============================================================================
# EXPERIMENTAL MATRIX DEFINITION
# =============================================================================

EXPERIMENTS = {
    'E0': {
        'description': 'Baseline: all neurons, all factors, at target onset',
        'seeds': [0],
        'priority': True,  # Run in priority mode
    },
    'E1': {
        'description': 'Temporal fix: all neurons, spatial from trial start',
        'seeds': [0],
        'priority': True,
    },
    'E2': {
        'description': 'Primary test: RF-matched spatial (E only), sparse features (p=0.5)',
        'seeds': [0, 1, 2, 3, 4],
        'priority': True,  # At least seed 0
    },
    'E3a': {
        'description': 'Sparsity sensitivity: p=0.25',
        'seeds': [0, 1, 2],
        'priority': False,
    },
    'E3b': {
        'description': 'Sparsity sensitivity: p=0.75',
        'seeds': [0, 1, 2],
        'priority': False,
    },
    'E4': {
        'description': 'Replication with different seeds',
        'seeds': [100, 101, 102, 103, 104],
        'priority': False,
    }
}


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Training parameters (extended for better convergence)
    'max_epochs': 2000,
    'patience': 300,
    'lr': 1e-3,
    'lr_scheduler_patience': 50,
    'lr_scheduler_factor': 0.5,
    'min_lr': 1e-5,
    'gradient_clip': 1.0,

    # Model parameters (best Phase 5 configuration)
    'tau': 50.0,
    'dt': 25.0,
    'noise_scale': 0.1,
    'spectral_radius': 0.9,

    # Input embedding (Phase 3 improvement: +2.1%)
    'input_embed_dim': 56,
    'input_embed_type': 'attention',
    'attention_heads': 4,

    # Learnable initial state (Phase 4 improvement: +4.6%)
    'learnable_h0': True,
    'h0_init': 0.1,

    # Loss parameters
    'use_grad_balancing': True,
    'ltrial_scale': 0.5,
    'lambda_reg': 1e-4,

    # Logging
    'checkpoint_every': 100,
    'print_every': 20,

    # Paths
    # Feb 2026 results used the stale 06:31 export (14 E / 28 I); the canonical
    # export is data/rnn_export_Newton_08_15_2025_SC.mat (41 E / 13 I)
    'data_path': 'data/export_2026-01-22_0631/rnn_export_Newton_08_15_2025_SC.mat',
    'output_dir': 'results/fit_quality_improvement',  # New output directory
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_psth_correlation(model_rates: np.ndarray, target_rates: np.ndarray) -> float:
    """Compute mean PSTH correlation across neurons."""
    model_psth = model_rates.mean(axis=0)   # [time, neurons]
    target_psth = target_rates.mean(axis=0)

    n_neurons = min(model_psth.shape[1], target_psth.shape[1])
    correlations = []
    for i in range(n_neurons):
        r = np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1]
        if not np.isnan(r):
            correlations.append(r)

    return float(np.mean(correlations)) if correlations else 0.0


def prepare_trial_factors(data: dict) -> Dict[str, np.ndarray]:
    """Extract trial-level factor values from data."""
    n_trials = int(data['n_trials'])
    return {
        'reward': data['trial_reward'].astype(float),
        'probability': data.get('trial_probability', np.zeros(n_trials)).astype(float),
        'salience': data.get('trial_salience', np.zeros(n_trials)).astype(float),
        'identity': data.get('trial_identity', np.zeros(n_trials)).astype(float)
    }


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_single_experiment(
    experiment_type: str,
    seed: int,
    data: dict,
    config: dict,
    output_dir: Path,
    test_mode: bool = False
) -> Dict[str, Any]:
    """
    Train a single model with specified input assignment.

    Args:
        experiment_type: E0, E1, E2, E3a, E3b, or E4
        seed: Random seed for input assignment and training
        data: Loaded session data
        config: Training configuration
        output_dir: Base output directory
        test_mode: If True, run quick test (5 epochs)

    Returns:
        Dictionary with training results and metrics
    """
    run_name = f"{experiment_type}_seed{seed}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {run_name}")
    print(f"  {EXPERIMENTS[experiment_type]['description']}")
    print(f"{'='*70}")

    # Override config for test mode
    if test_mode:
        config = config.copy()
        config['max_epochs'] = 5
        config['patience'] = 3
        config['print_every'] = 1

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Extract metadata
    n_neurons = int(data['n_neurons'])
    n_trials = int(data['n_trials'])
    n_bins = int(data['n_time_bins'])
    bin_size_ms = float(data['bin_size_ms'])
    neuron_type = data['neuron_type']

    n_exc = (neuron_type == 1).sum()
    n_inh = (neuron_type == 2).sum()
    print(f"Neurons: {n_neurons} ({n_exc} E, {n_inh} I)")
    print(f"Trials: {n_trials}, Time bins: {n_bins}")

    # ==========================================================================
    # STEP 1: Infer RF locations
    # ==========================================================================
    print("\nInferring RF locations...")
    rf_locations = infer_rf_locations(data)
    print(f"  E units with RF: {(rf_locations > 0).sum()}")
    for loc in range(1, 5):
        print(f"    Location {loc}: {(rf_locations == loc).sum()} neurons")

    # ==========================================================================
    # STEP 2: Create input assignment
    # ==========================================================================
    print("\nCreating input assignment...")
    assignment = create_assignment_for_experiment(
        experiment_type=experiment_type,
        n_neurons=n_neurons,
        neuron_type=neuron_type,
        rf_locations=rf_locations,
        seed=seed
    )
    print(assignment.summary())

    # Save assignment
    save_assignment(assignment, str(run_dir / 'input_assignment.npy'))
    with open(run_dir / 'assignment_stats.json', 'w') as f:
        json.dump(compute_assignment_statistics(assignment), f, indent=2)

    # ==========================================================================
    # STEP 3: Construct inputs based on assignment
    # ==========================================================================
    print("\nConstructing inputs...")

    if experiment_type == 'E0':
        # Baseline: use standard input construction
        inputs = construct_input_tensor(data)
        n_input_channels = inputs.shape[2]
    else:
        # Restructured inputs
        inputs = construct_restructured_input_tensor(data, assignment)
        n_input_channels = inputs.shape[3]  # [trials, bins, neurons, channels]

        # Reshape for model: [trials, bins, neurons * channels]
        # This allows neuron-specific inputs through the weight matrix
        inputs = inputs.reshape(n_trials, n_bins, n_neurons * n_input_channels)

    print(f"  Input shape: {inputs.shape}")

    # Targets
    targets = construct_target_tensor(data)  # [trials, bins, neurons]
    mask = construct_mask(data)

    # Train/val split
    indices = np.arange(n_trials)
    np.random.shuffle(indices)
    n_val = int(0.2 * n_trials)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_inputs = torch.tensor(inputs[train_idx], dtype=torch.float32)
    train_targets = torch.tensor(targets[train_idx], dtype=torch.float32)
    train_mask = torch.tensor(mask[train_idx], dtype=torch.float32)

    val_inputs = torch.tensor(inputs[val_idx], dtype=torch.float32)
    val_targets = torch.tensor(targets[val_idx], dtype=torch.float32)
    val_mask = torch.tensor(mask[val_idx], dtype=torch.float32)

    print(f"  Train trials: {len(train_idx)}, Val trials: {len(val_idx)}")

    # ==========================================================================
    # STEP 4: Create model
    # ==========================================================================
    print("\nCreating model...")

    # Input dimension depends on assignment type
    if experiment_type == 'E0':
        model_n_inputs = n_input_channels  # Standard 14 channels
        # For E0, use full attention embedding (14→56)
        use_embed_dim = config.get('input_embed_dim')
        use_embed_type = config.get('input_embed_type', 'attention')
    else:
        model_n_inputs = n_neurons * n_input_channels  # Neuron-specific inputs
        # For restructured inputs, use linear embedding (too many dims for attention)
        # Or no embedding if input dim is already large
        if model_n_inputs > 100:
            # Large input space - use linear embedding to moderate size
            use_embed_dim = config.get('input_embed_dim', 56)
            use_embed_type = 'linear'  # Attention over 300+ tokens is expensive
        else:
            use_embed_dim = config.get('input_embed_dim')
            use_embed_type = config.get('input_embed_type', 'attention')

    model = create_model_from_data(
        arch_version='feb2026',  # model code these results were trained with
        n_classic=n_exc,
        n_interneuron=n_inh,
        n_inputs=model_n_inputs,
        enforce_ratio=True,
        dt=config['dt'],
        tau=config['tau'],
        noise_scale=config['noise_scale'],
        spectral_radius=config['spectral_radius'],
        input_embed_dim=use_embed_dim,
        input_embed_type=use_embed_type,
        attention_heads=config.get('attention_heads', 4),
        learnable_h0=config.get('learnable_h0', False),
        h0_init=config.get('h0_init', 0.1),
        device=device
    )

    print(f"  Model: {model.n_total} units ({model.n_exc} E, {model.n_inh} I)")
    print(f"  Input dim: {model_n_inputs}")
    print(f"  Input embedding: {use_embed_type if use_embed_dim else 'None'} -> {model.actual_n_inputs}")
    print(f"  Learnable h0: {model.h0 is not None}")

    # ==========================================================================
    # STEP 5: Training loop
    # ==========================================================================
    print("\nStarting training...")

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
        'L_reg': []
    }

    start_time = time.time()

    for epoch in range(config['max_epochs']):
        # Training step
        model.train()
        train_inputs_gpu = train_inputs.to(device)
        train_targets_gpu = train_targets.to(device)
        train_mask_gpu = train_mask.to(device)

        optimizer.zero_grad()
        model_rates, _ = model(train_inputs_gpu)

        # Only compare recorded neurons
        n_recorded = train_targets_gpu.shape[2]
        model_rates_recorded = model_rates[:, :, :n_recorded]

        # Compute losses
        L_neuron = compute_L_neuron(
            model_rates_recorded, train_targets_gpu, bin_size_ms,
            mask=train_mask_gpu
        )
        L_trial = compute_L_trial(
            model_rates_recorded, train_targets_gpu, bin_size_ms,
            mask=train_mask_gpu
        )
        L_reg = compute_L_reg(model, config['lambda_reg'])

        # Gradient balancing
        if config['use_grad_balancing']:
            with torch.no_grad():
                loss_ema['L_neuron'] = ema_decay * loss_ema['L_neuron'] + (1 - ema_decay) * L_neuron.item()
                loss_ema['L_trial'] = ema_decay * loss_ema['L_trial'] + (1 - ema_decay) * L_trial.item()

            L_neuron_norm = L_neuron / (loss_ema['L_neuron'] + 1e-8)
            L_trial_norm = L_trial / (loss_ema['L_trial'] + 1e-8)
            loss = L_neuron_norm + config['ltrial_scale'] * L_trial_norm + L_reg
        else:
            loss = L_neuron + config['ltrial_scale'] * L_trial + L_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
        optimizer.step()

        train_loss = loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_inputs_gpu = val_inputs.to(device)
            val_targets_gpu = val_targets.to(device)
            val_mask_gpu = val_mask.to(device)

            val_rates, _ = model(val_inputs_gpu)
            val_rates_recorded = val_rates[:, :, :n_recorded]

            L_neuron_val = compute_L_neuron(
                val_rates_recorded, val_targets_gpu, bin_size_ms,
                mask=val_mask_gpu
            )
            val_loss = L_neuron_val.item()

            # PSTH correlation
            val_corr = compute_psth_correlation(
                val_rates_recorded.cpu().numpy(),
                val_targets_gpu.cpu().numpy()
            )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_correlation'].append(val_corr)
        history['learning_rate'].append(current_lr)
        history['L_neuron'].append(L_neuron.item())
        history['L_trial'].append(L_trial.item())
        history['L_reg'].append(L_reg.item())

        # Print progress
        if (epoch + 1) % config['print_every'] == 0:
            print(f"  Epoch {epoch+1:4d} | loss: {train_loss:.4f} | val_corr: {val_corr:.4f} | "
                  f"best: {best_val_corr:.4f} | lr: {current_lr:.1e}")

        # Track best
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= config['patience']:
            print(f"  Early stopping at epoch {epoch+1}")
            break

        # Checkpoint
        if (epoch + 1) % config['checkpoint_every'] == 0:
            torch.save(model.state_dict(), run_dir / f'model_epoch{epoch+1}.pt')

    training_time = time.time() - start_time
    epochs_trained = epoch + 1

    print(f"\nTraining complete:")
    print(f"  Epochs: {epochs_trained}")
    print(f"  Best val correlation: {best_val_corr:.4f} (epoch {best_epoch})")
    print(f"  Time: {timedelta(seconds=int(training_time))}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ==========================================================================
    # STEP 6: Save model and compute final metrics
    # ==========================================================================
    print("\nSaving outputs...")

    # Save model
    torch.save(best_model_state, run_dir / 'model_best.pt')

    # Save weights
    np.save(run_dir / 'W_rec.npy', model.W_rec.detach().cpu().numpy())
    np.save(run_dir / 'W_in.npy', model.W_in.detach().cpu().numpy())
    if model.h0 is not None:
        np.save(run_dir / 'h0.npy', model.h0.detach().cpu().numpy())
    if model.input_embed is not None:
        np.save(run_dir / 'input_embed_type.npy', {'type': model.input_embed.embed_type})

    # Save history
    with open(run_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # ==========================================================================
    # STEP 7: Compute emergent selectivity
    # ==========================================================================
    print("\nComputing emergent selectivity...")

    # Get model rates on validation set
    model.eval()
    with torch.no_grad():
        val_inputs_gpu = val_inputs.to(device)
        val_rates, _ = model(val_inputs_gpu)
        val_model_rates = val_rates[:, :, :n_recorded].cpu().numpy()

    # Find visual epoch
    target_onset = find_target_onset_bin(data['input_target_loc'])
    epoch_start = target_onset + int(50 / bin_size_ms)
    epoch_end = target_onset + int(250 / bin_size_ms)
    epoch_bins = (epoch_start, epoch_end)

    # Prepare trial factors for validation trials
    all_factors = prepare_trial_factors(data)
    val_factors = {k: v[val_idx] for k, v in all_factors.items()}

    # Compute emergent selectivity for each factor
    selectivity_results = compute_all_factor_selectivities(
        val_model_rates, val_factors, assignment, epoch_bins
    )

    # Save selectivity results
    selectivity_dict = {
        factor: result.to_dict()
        for factor, result in selectivity_results.items()
    }
    with open(run_dir / 'emergent_selectivity.json', 'w') as f:
        json.dump(selectivity_dict, f, indent=2)

    # Summary
    selectivity_summary = summarize_emergent_selectivity(selectivity_results)
    print("\nEmergent selectivity summary:")
    for factor, result in selectivity_results.items():
        print(f"  {factor}: {result.n_significant_emergent}/{result.n_non_receivers} significant "
              f"(mean |sel|={result.mean_abs_emergent:.4f})")

    # ==========================================================================
    # STEP 8: Run ablation analysis (for primary test conditions)
    # ==========================================================================
    if experiment_type in ['E2', 'E3a', 'E3b', 'E4']:
        print("\nRunning ablation analysis...")

        # Note: We need to limit to recorded neurons for ablation analysis
        # since the assignment only covers recorded neurons
        ablation_results = run_ablation_analysis(
            model=model,
            inputs=val_inputs,
            trial_factors=val_factors,
            assignment=assignment,
            epoch_bins=epoch_bins,
            ablation_types=['full_recurrence', 'ie_only'],
            device=device,
            n_recorded=n_recorded  # Limit to recorded neurons
        )

        # Save ablation results
        ablation_dict = {}
        for factor, factor_results in ablation_results.items():
            ablation_dict[factor] = {
                abl_type: result.to_dict()
                for abl_type, result in factor_results.items()
            }
        with open(run_dir / 'ablation_results.json', 'w') as f:
            json.dump(ablation_dict, f, indent=2)

        print(print_ablation_report(ablation_results))

    # ==========================================================================
    # STEP 9: Compile results
    # ==========================================================================
    results = {
        'experiment_type': experiment_type,
        'seed': seed,
        'epochs_trained': epochs_trained,
        'best_epoch': best_epoch,
        'best_val_correlation': float(best_val_corr),
        'training_time_seconds': training_time,
        'selectivity_summary': selectivity_summary,
        'config': config,
    }

    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run input restructuring experiments')
    parser.add_argument('--priority-only', action='store_true',
                       help='Only run priority experiments (E0, E1, E2_seed0)')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Run specific experiment (E0, E1, E2, E3a, E3b, E4)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Run specific seed only')
    parser.add_argument('--test', action='store_true',
                       help='Quick test run (5 epochs)')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Override data path')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory')
    args = parser.parse_args()

    # Setup configuration
    config = DEFAULT_CONFIG.copy()
    if args.data_path:
        config['data_path'] = args.data_path
    if args.output_dir:
        config['output_dir'] = args.output_dir

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine experiments to run
    experiments_to_run = []

    if args.experiment:
        exp_config = EXPERIMENTS[args.experiment]
        if args.seed is not None:
            experiments_to_run = [(args.experiment, args.seed)]
        else:
            experiments_to_run = [(args.experiment, s) for s in exp_config['seeds']]
    elif args.priority_only:
        for exp_name, exp_config in EXPERIMENTS.items():
            if exp_config.get('priority', False):
                if exp_name in ['E2', 'E3a', 'E3b', 'E4']:
                    experiments_to_run.append((exp_name, exp_config['seeds'][0]))
                else:
                    for s in exp_config['seeds']:
                        experiments_to_run.append((exp_name, s))
    else:
        # Full matrix
        for exp_name, exp_config in EXPERIMENTS.items():
            for s in exp_config['seeds']:
                experiments_to_run.append((exp_name, s))

    print(f"="*70)
    print(f"INPUT RESTRUCTURING EXPERIMENTS")
    print(f"="*70)
    print(f"Total experiments: {len(experiments_to_run)}")
    print(f"Output directory: {output_dir}")
    print(f"Test mode: {args.test}")
    print()

    # Load data once
    print(f"Loading data from {config['data_path']}...")
    data = load_mat_file(config['data_path'])
    validate_data(data)

    # Run experiments
    all_results = {}
    start_time = time.time()

    for i, (exp_name, seed) in enumerate(experiments_to_run):
        print(f"\n[{i+1}/{len(experiments_to_run)}] Running {exp_name}_seed{seed}...")

        try:
            results = train_single_experiment(
                experiment_type=exp_name,
                seed=seed,
                data=data,
                config=config,
                output_dir=output_dir,
                test_mode=args.test
            )
            all_results[f"{exp_name}_seed{seed}"] = results

        except Exception as e:
            print(f"ERROR in {exp_name}_seed{seed}: {e}")
            import traceback
            traceback.print_exc()
            all_results[f"{exp_name}_seed{seed}"] = {'error': str(e)}

        # Save intermediate summary
        with open(output_dir / 'all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    total_time = time.time() - start_time

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"\nResults:")

    for run_name, results in all_results.items():
        if 'error' in results:
            print(f"  {run_name}: ERROR - {results['error']}")
        else:
            print(f"  {run_name}: val_corr={results['best_val_correlation']:.4f}, "
                  f"epochs={results['epochs_trained']}")

    print(f"\nAll results saved to: {output_dir}")
    print("Run 'python scripts/analyze_input_restructuring.py' for full analysis.")


if __name__ == '__main__':
    main()
